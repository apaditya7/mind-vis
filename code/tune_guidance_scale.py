"""
Guidance Scale Tuning Script

Tests different classifier-free guidance scales to find the optimal value
for fMRI-conditioned image generation. This is an inference-only modification
that doesn't require retraining.

Usage:
    python code/tune_guidance_scale.py --checkpoint <path> --dataset GOD

The script will:
1. Test guidance scales: [1.0, 3.0, 5.0, 7.5, 10.0]
2. Generate images for each scale
3. Compute evaluation metrics (SSIM, PCC, FID, Top-1 classification)
4. Recommend the best guidance scale
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from eval_metrics import get_similarity_metric
from dataset import create_Kamitani_dataset, create_BOLD5000_dataset
from dc_ldm.ldm_for_fmri import fLDM
from einops import rearrange
from PIL import Image
import torchvision.transforms as transforms
from config import *

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
    img = img * 2.0 - 1.0  # to -1 ~ 1
    return img

def get_eval_metric(samples, avg=True, device='cpu'):
    """Compute evaluation metrics for generated samples."""
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

    # FID score
    fid_res = []
    for s in samples_to_run:
        pred_images = [img[s] for img in samples]
        pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
        fid_score = get_similarity_metric(pred_images, gt_images, method='metrics-only', metric_name='fid')
        fid_res.append(fid_score)
    res_list.append(np.mean(fid_res))
    metric_list.append('fid')

    # Top-1 classification accuracy
    res_part = []
    for s in samples_to_run:
        pred_images = [img[s] for img in samples]
        pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
        res = get_similarity_metric(pred_images, gt_images, 'class', None,
                        n_way=50, num_trials=1000, top_k=1, device=device)
        res_part.append(np.mean(res))
    res_list.append(np.mean(res_part))
    metric_list.append('top-1-class')

    return res_list, metric_list

def generate_with_guidance(model, dataset, guidance_scale, num_samples, ddim_steps,
                          HW, limit, state, device):
    """
    Generate images with specified guidance scale.

    This implements classifier-free guidance by:
    1. Creating unconditional (null) embeddings
    2. Computing both conditional and unconditional predictions
    3. Extrapolating: e_t = e_t_uncond + scale * (e_t_cond - e_t_uncond)
    """
    from dc_ldm.models.diffusion.plms import PLMSSampler
    from einops import repeat
    from torchvision.utils import make_grid

    all_samples = []

    if HW is None:
        shape = (model.ldm_config.model.params.channels,
            model.ldm_config.model.params.image_size,
            model.ldm_config.model.params.image_size)
    else:
        num_resolutions = len(model.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
        shape = (model.ldm_config.model.params.channels,
            HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

    model_nn = model.model.to(device)
    sampler = PLMSSampler(model_nn)

    if state is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(state)
            else:
                torch.set_rng_state(state)
        except (RuntimeError, ValueError) as e:
            print(f"Warning: Could not restore RNG state: {e}. Using fresh random state.")
            state = None

    with model_nn.ema_scope():
        model_nn.eval()

        with torch.no_grad():
            for count, item in enumerate(dataset):
                if limit is not None:
                    if count >= limit:
                        break

                latent = item['fmri']
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w')

                # Get conditional embedding
                c = model_nn.get_learned_conditioning(
                    repeat(latent, 'h w -> c h w', c=num_samples).to(device)
                )

                # Create unconditional embedding (zeros)
                uc = torch.zeros_like(c)

                # Sample with classifier-free guidance
                samples_ddim, _ = sampler.sample(
                    S=ddim_steps,
                    conditioning=c,
                    batch_size=num_samples,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=guidance_scale,
                    unconditional_conditioning=uc
                )

                x_samples_ddim = model_nn.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)

                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0))

    # Create grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=num_samples+1)
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()

    model_nn = model_nn.to('cpu')

    return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Tune guidance scale for fMRI-to-image generation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint (checkpoint.pth)')
    parser.add_argument('--dataset', type=str, default='GOD', choices=['GOD', 'BOLD5000'],
                       help='Dataset to use')
    parser.add_argument('--root', type=str, default='.',
                       help='Root directory')
    parser.add_argument('--max_samples', type=int, default=50,
                       help='Number of test samples to use (default: 50)')
    parser.add_argument('--num_generated_per_sample', type=int, default=1,
                       help='Number of images to generate per fMRI sample')
    parser.add_argument('--guidance_scales', type=float, nargs='+',
                       default=[1.0, 3.0, 5.0, 7.5, 10.0],
                       help='Guidance scales to test')
    parser.add_argument('--use_shape_conditioning', action='store_true',
                       help='Enable shape conditioning if model was trained with it')
    parser.add_argument('--shape_predictor_path', type=str, default=None,
                       help='Path to shape predictor checkpoint')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if args.use_shape_conditioning and not args.shape_predictor_path:
        raise ValueError("--shape_predictor_path required when --use_shape_conditioning is set")

    print("=" * 80)
    print("GUIDANCE SCALE TUNING")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Test samples: {args.max_samples}")
    print(f"Guidance scales to test: {args.guidance_scales}")
    print("=" * 80)

    # Load checkpoint
    print("\nLoading checkpoint...")
    sd = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = sd['config']

    # Update paths
    config.root_path = args.root
    config.kam_path = os.path.join(args.root, 'data/Kamitani/npz')
    config.bold5000_path = os.path.join(args.root, 'data/BOLD5000')
    config.pretrain_mbm_path = os.path.join(args.root, 'pretrains', args.dataset, 'fmri_encoder.pth')
    config.pretrain_gm_path = os.path.join(args.root, 'pretrains/ldm/label2img')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare dataset
    img_transform_test = transforms.Compose([
        normalize, transforms.Resize((256, 256)), channel_last
    ])

    if args.dataset == 'GOD':
        _, dataset_test = create_Kamitani_dataset(
            config.kam_path, config.roi, config.patch_size,
            fmri_transform=torch.FloatTensor, image_transform=img_transform_test,
            subjects=config.kam_subs, test_category=getattr(config, 'test_category', None)
        )
    elif args.dataset == 'BOLD5000':
        _, dataset_test = create_BOLD5000_dataset(
            config.bold5000_path, config.patch_size,
            fmri_transform=torch.FloatTensor, image_transform=img_transform_test,
            subjects=config.bold5000_subs
        )
    else:
        raise NotImplementedError

    num_voxels = dataset_test.num_voxels
    total_samples = min(len(dataset_test), args.max_samples)
    print(f"Test samples: {total_samples}\n")

    # Load model
    print("Loading model...")
    pretrain_mbm_metafile = torch.load(config.pretrain_mbm_path, map_location='cpu', weights_only=False)
    generative_model = fLDM(
        pretrain_mbm_metafile, num_voxels,
        device=device, pretrain_root=config.pretrain_gm_path,
        logger=None, ddim_steps=config.ddim_steps,
        global_pool=config.global_pool, use_time_cond=config.use_time_cond,
        use_shape_conditioning=args.use_shape_conditioning,
        shape_predictor_path=args.shape_predictor_path
    )
    generative_model.model.load_state_dict(sd['model_state_dict'], strict=False)
    print("Model loaded successfully\n")

    state = sd.get('state', None)

    # Create output directory
    output_dir = os.path.join(args.root, 'results', 'guidance_tuning')
    os.makedirs(output_dir, exist_ok=True)

    # Test each guidance scale
    results = {}

    for scale in args.guidance_scales:
        print("=" * 80)
        print(f"Testing guidance_scale = {scale}")
        print("=" * 80)

        # Generate images
        print("Generating images...")
        grid, samples = generate_with_guidance(
            generative_model, dataset_test, scale,
            args.num_generated_per_sample, config.ddim_steps,
            config.HW, total_samples, state, device
        )

        # Save grid
        scale_dir = os.path.join(output_dir, f"scale_{scale}")
        os.makedirs(scale_dir, exist_ok=True)
        grid_img = Image.fromarray(grid.astype(np.uint8))
        grid_img.save(os.path.join(scale_dir, 'samples.png'))

        # Compute metrics
        print("Computing metrics...")
        metric, metric_list = get_eval_metric(samples, avg=True, device=device)

        # Store results
        results[scale] = {name: value for name, value in zip(metric_list, metric)}

        # Print metrics
        print(f"\nResults for guidance_scale = {scale}:")
        print("-" * 60)
        for name, value in results[scale].items():
            if 'fid' in name.lower():
                print(f"  {name:20s}: {value:.2f} (lower is better)")
            elif 'class' in name.lower():
                print(f"  {name:20s}: {value*100:.2f}%")
            else:
                print(f"  {name:20s}: {value:.4f}")
        print()

    # Summary and recommendation
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nComparison across guidance scales:\n")

    # Print comparison table
    print(f"{'Metric':<20} | " + " | ".join([f"scale={s:>4.1f}" for s in args.guidance_scales]))
    print("-" * 100)

    for metric_name in results[args.guidance_scales[0]].keys():
        values = [results[scale][metric_name] for scale in args.guidance_scales]

        if 'fid' in metric_name.lower():
            # Lower is better for FID
            best_idx = np.argmin(values)
            print(f"{metric_name:<20} | " + " | ".join([
                f"{v:>9.2f}{'*' if i == best_idx else ' '}" for i, v in enumerate(values)
            ]))
        elif 'class' in metric_name.lower():
            # Higher is better for classification
            best_idx = np.argmax(values)
            print(f"{metric_name:<20} | " + " | ".join([
                f"{v*100:>8.2f}%{'*' if i == best_idx else ' '}" for i, v in enumerate(values)
            ]))
        else:
            # Higher is better for correlation metrics
            best_idx = np.argmax(values)
            print(f"{metric_name:<20} | " + " | ".join([
                f"{v:>9.4f}{'*' if i == best_idx else ' '}" for i, v in enumerate(values)
            ]))

    print("\n(* indicates best value for each metric)")

    # Compute aggregate score (normalized metrics)
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Normalize metrics and compute weighted score
    scores = {}
    for scale in args.guidance_scales:
        score = 0
        # SSIM (higher better)
        ssim_values = [results[s]['ssim'] for s in args.guidance_scales]
        score += (results[scale]['ssim'] - min(ssim_values)) / (max(ssim_values) - min(ssim_values) + 1e-8)

        # PCC (higher better)
        pcc_values = [results[s]['pcc'] for s in args.guidance_scales]
        score += (results[scale]['pcc'] - min(pcc_values)) / (max(pcc_values) - min(pcc_values) + 1e-8)

        # FID (lower better - invert)
        fid_values = [results[s]['fid'] for s in args.guidance_scales]
        score += (max(fid_values) - results[scale]['fid']) / (max(fid_values) - min(fid_values) + 1e-8)

        # Top-1 (higher better)
        top1_values = [results[s]['top-1-class'] for s in args.guidance_scales]
        score += (results[scale]['top-1-class'] - min(top1_values)) / (max(top1_values) - min(top1_values) + 1e-8)

        scores[scale] = score / 4  # Average

    best_scale = max(scores, key=scores.get)

    print(f"\nAggregate scores (normalized average of SSIM, PCC, FID, Top-1):")
    for scale in args.guidance_scales:
        marker = " <-- RECOMMENDED" if scale == best_scale else ""
        print(f"  guidance_scale = {scale:>4.1f}: {scores[scale]:.4f}{marker}")

    print(f"\nRecommended guidance scale: {best_scale}")
    print(f"\nTo use this guidance scale in generation, modify ldm_for_fmri.py:191-195")
    print(f"to pass unconditional_guidance_scale={best_scale} and unconditional_conditioning")
    print(f"to the sampler.sample() call.")

    print(f"\nResults saved to: {output_dir}")

    # Save results to file
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write("GUIDANCE SCALE TUNING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Test samples: {total_samples}\n\n")

        for scale in args.guidance_scales:
            f.write(f"\nGuidance scale = {scale}:\n")
            for name, value in results[scale].items():
                if 'fid' in name.lower():
                    f.write(f"  {name:20s}: {value:.2f}\n")
                elif 'class' in name.lower():
                    f.write(f"  {name:20s}: {value*100:.2f}%\n")
                else:
                    f.write(f"  {name:20s}: {value:.4f}\n")

        f.write(f"\n\nRECOMMENDED: guidance_scale = {best_scale}\n")


if __name__ == '__main__':
    main()
