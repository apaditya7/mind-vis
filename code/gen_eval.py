import os, sys
import numpy as np
import torch
from eval_metrics import get_similarity_metric
from dataset import create_Kamitani_dataset, create_BOLD5000_dataset
from dc_ldm.ldm_for_fmri import fLDM
from einops import rearrange
from PIL import Image
import torchvision.transforms as transforms
from config import *
import wandb
import datetime
import argparse

def to_image(img):
    if img.shape[-1] != 3:
        img = rearrange(img, 'c h w -> h w c')
    img = 255. * img
    return Image.fromarray(img.astype(np.uint8))

def channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, 'c h w -> h w c')

def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

def wandb_init(config):
    wandb.init( project="mind-vis",
                group='eval',
                anonymous="allow",
                config=config,
                reinit=True)

def get_eval_metric(samples, avg=True, device='cpu'):
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

    fid_res = []
    for s in samples_to_run:
        pred_images = [img[s] for img in samples]
        pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
        fid_score = get_similarity_metric(pred_images, gt_images, method='metrics-only', metric_name='fid')
        fid_res.append(fid_score)
    res_list.append(np.mean(fid_res))
    metric_list.append('fid')

    res_part = []
    for s in samples_to_run:
        pred_images = [img[s] for img in samples]
        pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
        res = get_similarity_metric(pred_images, gt_images, 'class', None,
                        n_way=50, num_trials=1000, top_k=1, device=device)
        res_part.append(np.mean(res))
    res_list.append(np.mean(res_part))
    res_list.append(np.max(res_part))
    metric_list.append('top-1-class')
    metric_list.append('top-1-class (max)')
    return res_list, metric_list

def get_args_parser():
    parser = argparse.ArgumentParser('Double Conditioning LDM Finetuning', add_help=False)
    # project parameters
    parser.add_argument('--root', type=str, default='.')
    parser.add_argument('--dataset', type=str, default='GOD')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit number of samples to process')
    # shape conditioning parameters
    parser.add_argument('--use_shape_conditioning', action='store_true', help='Enable shape conditioning')
    parser.add_argument('--shape_predictor_path', type=str, default=None,
                       help='Path to trained shape predictor model')

    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    # Validate shape conditioning arguments
    if args.use_shape_conditioning and not args.shape_predictor_path:
        raise ValueError("--shape_predictor_path must be provided when --use_shape_conditioning is enabled")

    if args.use_shape_conditioning and not os.path.exists(args.shape_predictor_path):
        raise FileNotFoundError(f"Shape predictor not found: {args.shape_predictor_path}")

    root = args.root
    target = args.dataset
    model_path = os.path.join(root, 'pretrains', f'{target}', 'finetuned.pth')
  
    sd = torch.load(model_path, map_location='cpu', weights_only=False)
    config = sd['config']
    # update paths
    config.root_path = root
    config.kam_path = os.path.join(root, 'data/Kamitani/npz')
    config.bold5000_path = os.path.join(root, 'data/BOLD5000')
    config.pretrain_mbm_path = os.path.join(root, 'pretrains', f'{target}', 'fmri_encoder.pth')
    config.pretrain_gm_path = os.path.join(root, 'pretrains/ldm/label2img')
    print(config.__dict__)

    output_path = os.path.join(config.root_path, 'results', 'eval',  
                    '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_transform_test = transforms.Compose([
        normalize, transforms.Resize((256, 256)), 
        channel_last
    ])

    if target == 'GOD':
        _, dataset_test = create_Kamitani_dataset(config.kam_path, config.roi, config.patch_size, 
                fmri_transform=torch.FloatTensor, image_transform=img_transform_test, 
                subjects=config.kam_subs, test_category=config.test_category)
    elif target == 'BOLD5000':
        _, dataset_test = create_BOLD5000_dataset(config.bold5000_path, config.patch_size, 
                fmri_transform=torch.FloatTensor, image_transform=img_transform_test, 
                subjects=config.bold5000_subs)
    else:
        raise NotImplementedError
 
    num_voxels = dataset_test.num_voxels
    total_samples = len(dataset_test)
    if args.max_samples:
        total_samples = min(total_samples, args.max_samples)
        print(f"Processing limited to {total_samples} samples")
    
    print(f"Total samples to process: {total_samples}")
    

    pretrain_mbm_metafile = torch.load(config.pretrain_mbm_path, map_location='cpu', weights_only=False)
    # create generateive model
    generative_model = fLDM(pretrain_mbm_metafile, num_voxels,
                device=device, pretrain_root=config.pretrain_gm_path, logger=config.logger,
                ddim_steps=config.ddim_steps, global_pool=config.global_pool, use_time_cond=config.use_time_cond,
                use_shape_conditioning=args.use_shape_conditioning,
                shape_predictor_path=args.shape_predictor_path)
    generative_model.model.load_state_dict(sd['model_state_dict'], strict=False)
    print('load ldm successfully')
    state = sd['state']
    
    # Use the original generate method but with limit parameter
    grid, samples = generative_model.generate(dataset_test, config.num_samples, 
                config.ddim_steps, config.HW, limit=total_samples, state=state)
    grid_imgs = Image.fromarray(grid.astype(np.uint8))

    os.makedirs(output_path, exist_ok=True)
    grid_imgs.save(os.path.join(output_path,f'./samples_test.png'))

    wandb_init(config)
    wandb.log({f'summary/samples_test': wandb.Image(grid_imgs)})
    metric, metric_list = get_eval_metric(samples, avg=True, device=device)
    metric_dict = {f'summary/pair-wise_{k}':v for k, v in zip(metric_list[:-2], metric[:-2])}
    metric_dict[f'summary/{metric_list[-2]}'] = metric[-2]
    metric_dict[f'summary/{metric_list[-1]}'] = metric[-1]
    print(metric_dict)
    wandb.log(metric_dict)
    
    print(f"Results saved to: {output_path}")