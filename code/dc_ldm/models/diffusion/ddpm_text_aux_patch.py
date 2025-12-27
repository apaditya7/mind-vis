"""
Patch for DDPM training_step to add text auxiliary supervision.

This file contains the modified training_step method that should replace
the one in dc_ldm/models/diffusion/ddpm.py when using text auxiliary supervision.

To apply:
1. Import this function in stageB_ldm_text_aux.py
2. Replace model.training_step with this patched version after model creation

The patch adds:
1. Text auxiliary contrastive loss during training
2. Loss logging for monitoring
3. Graceful handling when text embeddings are not available
"""

import torch
import torch.nn.functional as F


def text_aux_training_step(self, batch, batch_idx):
    """
    Modified training_step that includes text auxiliary supervision.

    Computes two losses:
    1. Main diffusion loss (reconstruction)
    2. Text auxiliary loss (contrastive alignment between fMRI features and text CLIP)

    Total loss = diffusion_loss + text_aux_weight * text_aux_loss
    """
    self.train()
    self.cond_stage_model.train()

    # Compute main diffusion loss
    loss_diffusion, loss_dict = self.shared_step(batch)

    # Add text auxiliary loss if enabled
    text_aux_loss = 0.0
    if hasattr(self, 'text_clip_lookup') and hasattr(self, 'text_aux_weight') and self.text_aux_weight > 0:
        try:
            # Get batch indices to look up text embeddings
            # Assuming batch contains 'index' or we can track it
            if 'index' in batch:
                batch_indices = batch['index']
            else:
                # Fallback: try to get indices from dataloader
                # This requires the dataloader to provide indices
                batch_indices = None

            if batch_indices is not None:
                # Get fMRI features from conditional encoder
                fmri_input = batch['fmri']
                with torch.no_grad() if not self.cond_stage_trainable else torch.enable_grad():
                    fmri_features = self.cond_stage_model.mae(fmri_input)

                    # If using global pooling, features are (batch, embed_dim)
                    # Otherwise, take mean across sequence: (batch, seq, embed_dim) -> (batch, embed_dim)
                    if fmri_features.dim() == 3:
                        fmri_features = fmri_features.mean(dim=1)

                # Project to CLIP space
                fmri_features_proj = self.text_aux_projector(fmri_features)

                # Get corresponding text CLIP embeddings
                text_embeds = []
                for idx in batch_indices.cpu().numpy():
                    idx = int(idx)
                    if idx in self.text_clip_lookup:
                        text_embeds.append(self.text_clip_lookup[idx])

                if len(text_embeds) > 0:
                    text_embeds = torch.FloatTensor(text_embeds).to(fmri_features.device)

                    # Compute contrastive loss
                    text_aux_loss = compute_contrastive_loss(
                        fmri_features_proj[:len(text_embeds)],
                        text_embeds,
                        temperature=self.text_aux_temperature
                    )

                    loss_dict['train/text_aux_loss'] = text_aux_loss.item()

        except Exception as e:
            # Gracefully handle errors (e.g., missing indices, shape mismatches)
            print(f"Warning: Text auxiliary loss computation failed: {e}")
            text_aux_loss = 0.0

    # Total loss
    total_loss = loss_diffusion + self.text_aux_weight * text_aux_loss if isinstance(text_aux_loss, torch.Tensor) else loss_diffusion
    loss_dict['train/total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

    self.log_dict(loss_dict, prog_bar=True,
                logger=True, on_step=False, on_epoch=True)

    if self.use_scheduler:
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    return total_loss


def compute_contrastive_loss(fmri_features, text_embeddings, temperature=0.07):
    """
    Compute bidirectional InfoNCE contrastive loss.

    Args:
        fmri_features: (batch, 512) - projected fMRI features
        text_embeddings: (batch, 512) - text CLIP embeddings
        temperature: temperature for softmax

    Returns:
        loss: scalar contrastive loss
    """
    # Normalize
    fmri_features = F.normalize(fmri_features, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    # Compute similarity matrix
    logits = torch.matmul(fmri_features, text_embeddings.T) / temperature

    # Labels: diagonal are positive pairs
    batch_size = fmri_features.size(0)
    labels = torch.arange(batch_size, device=fmri_features.device)

    # Bidirectional loss
    loss_fmri_to_text = F.cross_entropy(logits, labels)
    loss_text_to_fmri = F.cross_entropy(logits.T, labels)

    loss = (loss_fmri_to_text + loss_text_to_fmri) / 2

    return loss


def apply_text_aux_patch(model):
    """
    Apply text auxiliary supervision patch to DDPM model.

    Args:
        model: DDPM model instance

    This replaces the training_step method with the patched version.
    """
    import types
    model.training_step = types.MethodType(text_aux_training_step, model)
    print("Applied text auxiliary supervision patch to model.training_step")
