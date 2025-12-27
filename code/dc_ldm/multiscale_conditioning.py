"""
Multi-scale fMRI conditioning for LDM.

Extracts features from multiple layers of the fMRI encoder to capture
different levels of abstraction, then injects them at appropriate U-Net scales.
"""

import torch
import torch.nn as nn


class MultiScaleConditioner(nn.Module):
    """
    Extracts multi-scale features from fMRI encoder and prepares them
    for injection into U-Net at different resolution levels.
    """

    def __init__(self, fmri_encoder, extract_layers=[8, 16, 23], cond_dim=1280):
        """
        Args:
            fmri_encoder: Pre-trained fMRI MAE encoder
            extract_layers: Which transformer layers to extract features from
                          Early layers (0-8): Low-level features
                          Mid layers (9-16): Mid-level features
                          Late layers (17-23): High-level features
            cond_dim: Target conditioning dimension
        """
        super().__init__()
        self.fmri_encoder = fmri_encoder
        self.extract_layers = sorted(extract_layers)
        self.num_scales = len(extract_layers)

        # Feature dimension from encoder
        self.fmri_dim = fmri_encoder.embed_dim

        # Projection layers for each scale
        self.projectors = nn.ModuleList([
            nn.Linear(self.fmri_dim, cond_dim)
            for _ in range(self.num_scales)
        ])

        # Register hooks to extract intermediate features
        self.features = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to extract intermediate layer outputs."""
        def get_hook(layer_idx):
            def hook(module, input, output):
                self.features[layer_idx] = output
            return hook

        # Register hooks on transformer blocks
        for layer_idx in self.extract_layers:
            self.fmri_encoder.blocks[layer_idx].register_forward_hook(
                get_hook(layer_idx)
            )

    def forward(self, x):
        """
        Extract multi-scale features from fMRI encoder.

        Args:
            x: (batch, 1, num_voxels) fMRI input

        Returns:
            multi_scale_features: list of (batch, seq_len, cond_dim) tensors,
                                 one for each scale, from coarse to fine
        """
        self.features = {}

        # Forward pass through encoder (hooks will capture intermediate features)
        _ = self.fmri_encoder(x)

        # Project each scale to conditioning dimension
        multi_scale_cond = []
        for i, layer_idx in enumerate(self.extract_layers):
            feat = self.features[layer_idx]  # (batch, seq_len, fmri_dim)

            # Global pool if single token needed
            if hasattr(self.fmri_encoder, 'global_pool') and self.fmri_encoder.global_pool:
                feat = feat.mean(dim=1, keepdim=True)  # (batch, 1, fmri_dim)

            # Project to conditioning dim
            cond = self.projectors[i](feat)  # (batch, seq_len/1, cond_dim)
            multi_scale_cond.append(cond)

        return multi_scale_cond


class MultiScaleInjector(nn.Module):
    """
    Injects multi-scale conditioning into U-Net at appropriate resolution levels.

    Strategy:
    - Coarse features (early encoder layers) → Low-res U-Net blocks
    - Fine features (late encoder layers) → High-res U-Net blocks
    """

    def __init__(self, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.current_scale_idx = 0

    def get_conditioning_for_resolution(self, resolution_level):
        """
        Get the appropriate conditioning scale for a given U-Net resolution.

        Args:
            resolution_level: 0 (lowest res) to N (highest res)

        Returns:
            scale_idx: Which conditioning scale to use
        """
        # Map resolution level to conditioning scale
        # Lower resolution → use coarse features (early layers)
        # Higher resolution → use fine features (late layers)
        if resolution_level <= 1:
            return 0  # Coarse features
        elif resolution_level <= 3:
            return min(1, self.num_scales - 1)  # Mid features
        else:
            return self.num_scales - 1  # Fine features


def create_multiscale_cond_stage_model(metafile, num_voxels, cond_dim=1280,
                                       global_pool=True, extract_layers=[8, 16, 23]):
    """
    Create multi-scale conditioning stage model.

    Args:
        metafile: fMRI encoder checkpoint
        num_voxels: Number of voxels
        cond_dim: Conditioning dimension
        global_pool: Whether to use global pooling
        extract_layers: Which layers to extract features from

    Returns:
        multiscale_conditioner: MultiScaleConditioner module
    """
    from sc_mbm.mae_for_fmri import fmri_encoder

    config = metafile['config']
    model = fmri_encoder(
        num_voxels=num_voxels,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        global_pool=global_pool
    )
    model.load_checkpoint(metafile['model'])

    multiscale_conditioner = MultiScaleConditioner(
        fmri_encoder=model,
        extract_layers=extract_layers,
        cond_dim=cond_dim
    )

    return multiscale_conditioner
