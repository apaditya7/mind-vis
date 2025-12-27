"""
Stage B LDM Fine-tuning with Text Auxiliary Supervision

This extends the baseline Stage B training with text auxiliary supervision:
- During training: Uses text CLIP embeddings to guide fMRI feature learning via contrastive loss
- During inference: Only uses fMRI conditioning (no text needed)

Key differences from baseline:
1. Loads text CLIP embeddings alongside fMRI data
2. Adds contrastive loss between fMRI features and text CLIP embeddings
3. Total loss = LDM loss + λ * contrastive loss

This addresses the cross-modal gap issue while avoiding the dual CLIP prediction bottleneck.
"""

import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import datetime
import wandb
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import copy

# own code
from config import Config_Generative_Model
from dataset import create_Kamitani_dataset, create_BOLD5000_dataset
from dc_ldm.ldm_for_fmri import fLDM
from eval_metrics import get_similarity_metric


def wandb_init(config, output_path):
    wandb.init( project='mind-vis',
                group="stageB_text_aux",
                anonymous="allow",
                config=config,
                reinit=True)
    create_readme(config, output_path)

def wandb_finish():
    wandb.finish()

def to_image(img):
    if img.shape[-1] != 3:
        img = rearrange(img, 'c h w -> h w c')
    img = 255. * img
    return Image.fromarray(img.astype(np.uint8))

def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')

def get_eval_metric(samples, avg=True):
    metric_list = ['mse', 'pcc', 'ssim', 'psm']
    res_list = []

    gt_images = [img[0] for img in samples]
    gt_images = rearrange(np.stack(gt_images), 'n c h w -> n h w c')
    samples_to_run = np.arange(1, len(samples[0])) if avg else [1]
    for m in metric_list:
        res_part = []
        for s in samples_to_run:
            pred_images = [img[s] for img in samples]
            pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
            res = get_similarity_metric(pred_images, gt_images, method='pair-wise', metric_name=m)
            res_part.append(np.mean(res))
        res_list.append(np.mean(res_part))
    res_part = []
    for s in samples_to_run:
        pred_images = [img[s] for img in samples]
        pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
        res = get_similarity_metric(pred_images, gt_images, 'class', None,
                        n_way=50, num_trials=50, top_k=1, device='cuda')
        res_part.append(np.mean(res))
    res_list.append(np.mean(res_part))
    res_list.append(np.max(res_part))
    metric_list.append('top-1-class')
    metric_list.append('top-1-class (max)')
    return res_list, metric_list

def generate_images(generative_model, fmri_latents_dataset_train, fmri_latents_dataset_test, config):
    grid, _ = generative_model.generate(fmri_latents_dataset_train, config.num_samples,
                config.ddim_steps, config.HW, 10) # generate 10 instances
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path, 'samples_train.png'))
    wandb.log({'summary/samples_train': wandb.Image(grid_imgs)})

    grid, samples = generative_model.generate(fmri_latents_dataset_test, config.num_samples,
                config.ddim_steps, config.HW)
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path,f'./samples_test.png'))
    for sp_idx, imgs in enumerate(samples):
        for copy_idx, img in enumerate(imgs[1:]):
            img = rearrange(img, 'c h w -> h w c')
            Image.fromarray(img).save(os.path.join(config.output_path,
                            f'./test{sp_idx}-{copy_idx}.png'))

    wandb.log({f'summary/samples_test': wandb.Image(grid_imgs)})

    metric, metric_list = get_eval_metric(samples, avg=config.eval_avg)
    metric_dict = {f'summary/pair-wise_{k}':v for k, v in zip(metric_list[:-2], metric[:-2])}
    metric_dict[f'summary/{metric_list[-2]}'] = metric[-2]
    metric_dict[f'summary/{metric_list[-1]}'] = metric[-1]
    wandb.log(metric_dict)

def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img

def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)


def load_text_clip_embeddings(clip_embeddings_path):
    """
    Load text CLIP embeddings for text auxiliary supervision.

    Returns:
        text_clip_embeddings: numpy array of text embeddings
        text_clip_image_ids: list of image IDs
        text_start_idx: dict mapping image_id -> (start_idx, end_idx) in text_clip_embeddings
    """
    data = np.load(clip_embeddings_path)

    text_embeddings = data['text_clip_embeddings']  # Shape: (total_captions, 512)
    image_ids = data['text_clip_image_ids']  # Image IDs corresponding to captions
    lengths = data['text_clip_lengths']  # Number of captions per image

    # Build index: image_id -> (start, end) in text_embeddings array
    text_start_idx = {}
    current_idx = 0
    for img_id, length in zip(image_ids, lengths):
        text_start_idx[img_id] = (current_idx, current_idx + length)
        current_idx += length

    return text_embeddings, list(image_ids), text_start_idx


def create_text_clip_lookup(clip_embeddings_path, dataset):
    """
    Create a lookup table mapping dataset index -> text CLIP embedding.

    Args:
        clip_embeddings_path: path to CLIP embeddings .npz file
        dataset: Kamitani dataset

    Returns:
        text_clip_lookup: dict mapping dataset index -> text CLIP embedding (512,)
    """
    import csv

    # Load text CLIP embeddings
    text_embeddings, image_ids, text_start_idx = load_text_clip_embeddings(clip_embeddings_path)

    # Load image ID mapping from CSV (maps dataset index -> image ID)
    csv_path = os.path.join(os.path.dirname(clip_embeddings_path), '..', 'Kamitani', 'npz', 'imagenet_training_label.csv')
    with open(csv_path, 'r') as f:
        csvreader = csv.reader(f)
        img_ids_ordered = [row[1].strip('"').replace('.JPEG', '') for row in csvreader]

    # Build lookup: dataset_index -> text CLIP embedding
    text_clip_lookup = {}

    for idx in range(len(dataset)):
        # Map dataset index to unique image index (handling repeats across subjects)
        img_idx = idx % len(img_ids_ordered)
        img_id = img_ids_ordered[img_idx]

        if img_id in text_start_idx:
            start, end = text_start_idx[img_id]
            # Use last caption (most descriptive)
            if end > start:
                caption_idx = end - 1
                text_clip_lookup[idx] = text_embeddings[caption_idx]

    return text_clip_lookup


def compute_text_auxiliary_loss(fmri_features, text_clip_embeddings, temperature=0.07):
    """
    Compute contrastive loss between fMRI features and text CLIP embeddings.

    This encourages the fMRI encoder to learn features that align with text semantics,
    providing auxiliary supervision during training.

    Args:
        fmri_features: (batch, dim) - features from fMRI encoder
        text_clip_embeddings: (batch, 512) - text CLIP embeddings
        temperature: temperature for InfoNCE loss

    Returns:
        loss: scalar contrastive loss
    """
    # Normalize features
    fmri_features = F.normalize(fmri_features, dim=-1)
    text_embeddings = F.normalize(text_clip_embeddings, dim=-1)

    # Compute similarity matrix
    logits = torch.matmul(fmri_features, text_embeddings.T) / temperature

    # Labels: diagonal elements are positive pairs
    batch_size = fmri_features.size(0)
    labels = torch.arange(batch_size, device=fmri_features.device)

    # Bidirectional contrastive loss (fMRI->text and text->fMRI)
    loss_fmri_to_text = F.cross_entropy(logits, labels)
    loss_text_to_fmri = F.cross_entropy(logits.T, labels)

    loss = (loss_fmri_to_text + loss_text_to_fmri) / 2

    return loss


def main(config):
    # project setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    crop_pix = int(config.crop_ratio*config.img_size)
    img_transform_train = transforms.Compose([
        normalize,
        random_crop(config.img_size-crop_pix, p=0.5),
        transforms.Resize((256, 256)),
        channel_last
    ])
    img_transform_test = transforms.Compose([
        normalize, transforms.Resize((256, 256)),
        channel_last
    ])
    if config.dataset == 'GOD':
        fmri_latents_dataset_train, fmri_latents_dataset_test = create_Kamitani_dataset(config.kam_path, config.roi, config.patch_size,
                fmri_transform=fmri_transform, image_transform=[img_transform_train, img_transform_test],
                subjects=config.kam_subs)
        num_voxels = fmri_latents_dataset_train.num_voxels
    elif config.dataset == 'BOLD5000':
        fmri_latents_dataset_train, fmri_latents_dataset_test = create_BOLD5000_dataset(config.bold5000_path, config.patch_size,
                fmri_transform=fmri_transform, image_transform=[img_transform_train, img_transform_test],
                subjects=config.bold5000_subs)
        num_voxels = fmri_latents_dataset_train.num_voxels
    else:
        raise NotImplementedError

    # Load text CLIP embeddings for auxiliary supervision
    if hasattr(config, 'clip_embeddings_path') and config.clip_embeddings_path and config.text_aux_weight > 0:
        print("\n" + "="*60)
        print("LOADING TEXT CLIP EMBEDDINGS FOR AUXILIARY SUPERVISION")
        print("="*60)
        text_clip_lookup = create_text_clip_lookup(config.clip_embeddings_path, fmri_latents_dataset_train)
        print(f"Loaded text embeddings for {len(text_clip_lookup)}/{len(fmri_latents_dataset_train)} training samples")
        print(f"Text auxiliary loss weight: {config.text_aux_weight}")
        print("="*60 + "\n")
    else:
        text_clip_lookup = None
        print("\nText auxiliary supervision disabled (set --clip_embeddings_path and --text_aux_weight > 0 to enable)\n")

    # prepare pretrained mbm
    pretrain_mbm_metafile = torch.load(config.pretrain_mbm_path, map_location='cpu')

    # create generateive model with text auxiliary supervision support
    generative_model = fLDM(pretrain_mbm_metafile, num_voxels,
                device=device, pretrain_root=config.pretrain_gm_path, logger=config.logger,
                ddim_steps=config.ddim_steps, global_pool=config.global_pool, use_time_cond=config.use_time_cond,
                use_shape_conditioning=getattr(config, 'use_shape_conditioning', False),
                shape_predictor_path=getattr(config, 'shape_predictor_path', None))

    # Attach text auxiliary supervision components to model
    if text_clip_lookup is not None:
        generative_model.model.text_clip_lookup = text_clip_lookup
        generative_model.model.text_aux_weight = config.text_aux_weight
        generative_model.model.text_aux_temperature = getattr(config, 'text_aux_temperature', 0.07)

        # Add projection layer to map fMRI features to CLIP dimension
        cond_dim = generative_model.model.cond_stage_model.fmri_latent_dim
        generative_model.model.text_aux_projector = torch.nn.Linear(cond_dim, 512).to(device)

        print(f"Text auxiliary projector: {cond_dim} -> 512")

    # resume training if applicable
    if config.checkpoint_path is not None:
        model_meta = torch.load(config.checkpoint_path, map_location='cpu')
        generative_model.model.load_state_dict(model_meta['model_state_dict'], strict=False)
        print('model resumed')

    # finetune the model
    trainer = create_trainer(config.num_epoch, config.precision, config.accumulate_grad, logger, check_val_every_n_epoch=5)
    generative_model.finetune(trainer, fmri_latents_dataset_train, fmri_latents_dataset_test,
                config.batch_size, config.lr, config.output_path, config=config)

    # generate images
    generate_images(generative_model, fmri_latents_dataset_train, fmri_latents_dataset_test, config)

    return

def get_args_parser():
    parser = argparse.ArgumentParser('Stage B LDM with Text Auxiliary Supervision', add_help=False)
    # project parameters
    parser.add_argument('--seed', type=int)
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--kam_path', type=str)
    parser.add_argument('--bold5000_path', type=str)
    parser.add_argument('--pretrain_mbm_path', type=str)
    parser.add_argument('--crop_ratio', type=float)
    parser.add_argument('--dataset', type=str)

    # finetune parameters
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--precision', type=int)
    parser.add_argument('--accumulate_grad', type=int)
    parser.add_argument('--global_pool', type=bool)

    # diffusion sampling parameters
    parser.add_argument('--pretrain_gm_path', type=str)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--ddim_steps', type=int)
    parser.add_argument('--use_time_cond', type=bool)
    parser.add_argument('--eval_avg', type=bool)

    # shape conditioning parameters
    parser.add_argument('--use_shape_conditioning', action='store_true', help='Enable shape conditioning')
    parser.add_argument('--shape_predictor_path', type=str, default=None,
                       help='Path to trained shape predictor model')

    # text auxiliary supervision parameters
    parser.add_argument('--clip_embeddings_path', type=str, default=None,
                       help='Path to CLIP embeddings .npz file for text auxiliary supervision')
    parser.add_argument('--text_aux_weight', type=float, default=0.1,
                       help='Weight for text auxiliary contrastive loss (0 to disable)')
    parser.add_argument('--text_aux_temperature', type=float, default=0.07,
                       help='Temperature for text auxiliary contrastive loss')

    return parser

def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))

    # Add text auxiliary parameters
    if hasattr(args, 'clip_embeddings_path'):
        config.clip_embeddings_path = args.clip_embeddings_path
    if hasattr(args, 'text_aux_weight'):
        config.text_aux_weight = args.text_aux_weight
    if hasattr(args, 'text_aux_temperature'):
        config.text_aux_temperature = args.text_aux_temperature
    if hasattr(args, 'use_shape_conditioning'):
        config.use_shape_conditioning = args.use_shape_conditioning
    if hasattr(args, 'shape_predictor_path'):
        config.shape_predictor_path = args.shape_predictor_path

    return config

def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)


def create_trainer(num_epoch, precision=32, accumulate_grad_batches=2,logger=None,check_val_every_n_epoch=0):
    acc = 'gpu' if torch.cuda.is_available() else 'cpu'
    return pl.Trainer(accelerator=acc, max_epochs=num_epoch, logger=logger,
            precision=precision, accumulate_grad_batches=accumulate_grad_batches,
            enable_checkpointing=False, enable_model_summary=False, gradient_clip_val=0.5,
            check_val_every_n_epoch=check_val_every_n_epoch)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    config = Config_Generative_Model()
    config = update_config(args, config)

    if config.checkpoint_path is not None:
        model_meta = torch.load(config.checkpoint_path, map_location='cpu')
        ckp = config.checkpoint_path
        config = model_meta['config']
        config.checkpoint_path = ckp
        print('Resuming from checkpoint: {}'.format(config.checkpoint_path))

    output_path = os.path.join(config.root_path, 'results', 'stageB_text_aux',  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    config.output_path = output_path
    os.makedirs(output_path, exist_ok=True)

    wandb_init(config, output_path)

    logger = WandbLogger()
    config.logger = logger
    main(config)
    wandb_finish()
