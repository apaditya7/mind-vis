"""
Advanced fusion modules for combining shape and semantic conditioning.

Provides multiple fusion strategies:
1. Weighted sum (baseline)
2. Gated fusion (learnable weights)
3. Concat + projection
4. Adaptive fusion (timestep-dependent)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSumFusion(nn.Module):
    """
    Simple weighted sum fusion (current baseline).

    out = semantic + weight * shape
    """

    def __init__(self, shape_weight=0.05):
        super().__init__()
        self.shape_weight = shape_weight

    def forward(self, semantic, shape):
        """
        Args:
            semantic: (batch, seq_len, dim) or (batch, 1, dim)
            shape: (batch, 1, dim) or (batch, seq_len, dim)

        Returns:
            fused: same shape as semantic
        """
        return semantic + self.shape_weight * shape


class GatedFusion(nn.Module):
    """
    Gated fusion with learnable, input-dependent weights.

    Learns to dynamically balance shape vs semantic based on the input features.

    gate = sigmoid(W @ [semantic; shape] + b)
    out = gate * semantic + (1 - gate) * shape
    """

    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, semantic, shape):
        """
        Args:
            semantic: (batch, seq_len, dim) or (batch, 1, dim)
            shape: (batch, 1, dim) or (batch, seq_len, dim)

        Returns:
            fused: same shape as semantic
        """
        # Concatenate for gating decision
        if semantic.size(1) != shape.size(1):
            # Expand shape to match semantic seq_len
            shape = shape.expand(-1, semantic.size(1), -1)

        concat = torch.cat([semantic, shape], dim=-1)  # (batch, seq_len, 2*dim)

        # Compute gate
        gate = self.gate_net(concat)  # (batch, seq_len, 1)

        # Weighted combination
        fused = gate * semantic + (1 - gate) * shape

        return fused


class ConcatProjectionFusion(nn.Module):
    """
    Concatenation followed by projection.

    Preserves both shape and semantic information, then projects to target dim.

    out = W @ [semantic; shape] + b
    """

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, semantic, shape):
        """
        Args:
            semantic: (batch, seq_len, dim) or (batch, 1, dim)
            shape: (batch, 1, dim) or (batch, seq_len, dim)

        Returns:
            fused: same shape as semantic
        """
        # Expand shape if needed
        if semantic.size(1) != shape.size(1):
            shape = shape.expand(-1, semantic.size(1), -1)

        # Concatenate
        concat = torch.cat([semantic, shape], dim=-1)  # (batch, seq_len, 2*dim)

        # Project back to original dim
        fused = self.proj(concat)

        return fused


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion with timestep-dependent weights.

    The fusion weight depends on the diffusion timestep, allowing shape
    to contribute more at early (coarse) timesteps and less at late (fine) timesteps.

    This is motivated by the idea that shape is more important for coarse structure,
    while semantic features matter more for fine details.

    gate(t) = sigmoid(MLP(t_embed))
    out = gate(t) * semantic + (1 - gate(t)) * shape
    """

    def __init__(self, dim, time_embed_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, semantic, shape, timestep_embed=None):
        """
        Args:
            semantic: (batch, seq_len, dim) or (batch, 1, dim)
            shape: (batch, 1, dim) or (batch, seq_len, dim)
            timestep_embed: (batch, time_embed_dim) - timestep embeddings from U-Net

        Returns:
            fused: same shape as semantic
        """
        # Expand shape if needed
        if semantic.size(1) != shape.size(1):
            shape = shape.expand(-1, semantic.size(1), -1)

        if timestep_embed is not None:
            # Compute timestep-dependent gate
            gate = self.time_mlp(timestep_embed)  # (batch, 1)
            gate = gate.unsqueeze(1)  # (batch, 1, 1)
        else:
            # Fallback to fixed weight if timestep not provided
            gate = 0.5

        # Weighted combination
        fused = gate * semantic + (1 - gate) * shape

        return fused


class MultiModalAttention(nn.Module):
    """
    Cross-attention based fusion.

    Uses semantic features as queries and shape features as keys/values.
    This allows semantic features to selectively attend to shape information.

    This is the most expressive fusion mechanism but also most computationally expensive.
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, semantic, shape):
        """
        Args:
            semantic: (batch, seq_len, dim) - queries
            shape: (batch, 1, dim) - keys/values

        Returns:
            fused: (batch, seq_len, dim)
        """
        # Expand shape if needed
        if shape.size(1) == 1 and semantic.size(1) > 1:
            shape = shape.expand(-1, semantic.size(1), -1)

        # Cross-attention: semantic attends to shape
        attended, _ = self.attention(semantic, shape, shape)

        # Residual connection + norm
        fused = self.norm(semantic + attended)

        return fused


def create_fusion_module(fusion_type, dim, **kwargs):
    """
    Factory function to create fusion modules.

    Args:
        fusion_type: one of ['weighted_sum', 'gated', 'concat', 'adaptive', 'attention']
        dim: feature dimension
        **kwargs: additional arguments for specific fusion types

    Returns:
        fusion_module: nn.Module
    """
    if fusion_type == 'weighted_sum':
        return WeightedSumFusion(shape_weight=kwargs.get('shape_weight', 0.05))
    elif fusion_type == 'gated':
        return GatedFusion(dim, hidden_dim=kwargs.get('hidden_dim', 256))
    elif fusion_type == 'concat':
        return ConcatProjectionFusion(dim)
    elif fusion_type == 'adaptive':
        return AdaptiveFusion(dim, time_embed_dim=kwargs.get('time_embed_dim', 256))
    elif fusion_type == 'attention':
        return MultiModalAttention(dim, num_heads=kwargs.get('num_heads', 8))
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
