import numpy as np
import wandb
import torch
from dc_ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch.nn as nn
import os
from dc_ldm.models.diffusion.plms import PLMSSampler
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sc_mbm.mae_for_fmri import fmri_encoder

def create_model_from_config(config, num_voxels, global_pool):
    model = fmri_encoder(num_voxels=num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
                depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, global_pool=global_pool)
    return model

class cond_stage_model(nn.Module):
    def __init__(self, metafile, num_voxels, cond_dim=1280, global_pool=True,
                 use_shape_conditioning=False, shape_predictor_path=None):
        super().__init__()
        # prepare pretrained fmri mae (no shape head needed)
        model = create_model_from_config(metafile['config'], num_voxels, global_pool)
        model.load_checkpoint(metafile['model'])
        self.mae = model
        self.fmri_seq_len = model.num_patches
        self.fmri_latent_dim = model.embed_dim
        if global_pool == False:
            self.channel_mapper = nn.Sequential(
                nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True),
                nn.Conv1d(self.fmri_seq_len // 2, 77, 1, bias=True)
            )
        self.dim_mapper = nn.Linear(self.fmri_latent_dim, cond_dim, bias=True)
        self.global_pool = global_pool
        self.use_shape_conditioning = use_shape_conditioning

        # External shape predictor
        self.shape_predictor = None
        if use_shape_conditioning and shape_predictor_path:
            from shape_predictor import ShapePredictor
            checkpoint = torch.load(shape_predictor_path, map_location='cpu')
            self.shape_predictor = ShapePredictor(
                fmri_dim=checkpoint['fmri_dim'],
                shape_dim=checkpoint['shape_dim']
            )
            self.shape_predictor.load_state_dict(checkpoint['model_state_dict'])
            self.shape_predictor.eval()

            # Map shape predictions to same dimension as semantic conditioning
            self.shape_mapper = nn.Linear(checkpoint['shape_dim'], cond_dim, bias=True)
            self.shape_weight = 0.05  # Weight for blending shape with semantic

    def forward(self, x):
        # n, c, w = x.shape
        # Get semantic embedding from fMRI encoder
        latent_crossattn = self.mae(x)
        if self.global_pool == False:
            latent_crossattn = self.channel_mapper(latent_crossattn)
        latent_crossattn = self.dim_mapper(latent_crossattn)

        if self.use_shape_conditioning and self.shape_predictor is not None:
            # Get shape prediction from external shape predictor
            fmri_flattened = x.flatten(1)  # Flatten fMRI for shape predictor
            with torch.no_grad():  # Don't train shape predictor during diffusion training
                shape_pred = self.shape_predictor(fmri_flattened)

            # Map shape to conditioning space (same dim as semantic)
            shape_cond = self.shape_mapper(shape_pred)

            # Reshape shape conditioning to match semantic conditioning
            if self.global_pool:
                shape_cond = shape_cond.unsqueeze(1)  # (batch, 1, cond_dim)
            else:
                shape_cond = shape_cond.unsqueeze(1).expand(-1, latent_crossattn.size(1), -1)

            # Blend semantic and shape conditioning (don't concatenate)
            out = latent_crossattn + self.shape_weight * shape_cond
            return out
        else:
            return latent_crossattn

class fLDM:

    def __init__(self, metafile, num_voxels, device=torch.device('cpu'),
                 pretrain_root='../pretrains/ldm/label2img',
                 logger=None, ddim_steps=250, global_pool=True, use_time_cond=True,
                 use_shape_conditioning=False, shape_predictor_path=None):
        self.ckp_path = os.path.join(pretrain_root, 'model.ckpt')
        self.config_path = os.path.join(pretrain_root, 'config.yaml')
        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim
        self.use_shape_conditioning = use_shape_conditioning

        # Keep original context dimension - don't change U-Net architecture

        model = instantiate_from_config(config.model)
        pl_sd = torch.load(self.ckp_path, map_location="cpu")['state_dict']

        m, u = model.load_state_dict(pl_sd, strict=False)
        model.cond_stage_trainable = True
        model.cond_stage_model = cond_stage_model(metafile, num_voxels, self.cond_dim, global_pool=global_pool,
                                                 use_shape_conditioning=use_shape_conditioning,
                                                 shape_predictor_path=shape_predictor_path)

        model.ddim_steps = ddim_steps
        model.re_init_ema()
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult

        self.device = device    
        self.model = model
        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.fmri_latent_dim = model.cond_stage_model.fmri_latent_dim
        self.metafile = metafile

    def finetune(self, trainers, dataset, test_dataset, bs1, lr1,
                output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        # self.model.train_dataset = dataset
        self.model.run_full_validation_threshold = 0.15
        # stage one: train the cond encoder with the pretrained one
      
        # # stage one: only optimize conditional encoders
        print('\n##### Stage One: only optimize conditional encoders #####')
        dataloader = DataLoader(dataset, batch_size=bs1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        self.model.unfreeze_whole_model()
        self.model.freeze_first_stage()

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = True
        self.model.eval_avg = config.eval_avg
        trainers.fit(self.model, dataloader, val_dataloaders=test_loader)

        self.model.unfreeze_whole_model()
        
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': config,
                'state': torch.random.get_rng_state()

            },
            os.path.join(output_path, 'checkpoint.pth')
        )
        

    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None, guidance_scale=5.0):
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels,
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        if state is not None:
            try:
                torch.cuda.set_rng_state(state)
            except (RuntimeError, ValueError) as e:
                print(f"Warning: Could not restore RNG state: {e}. Using fresh random state.")

        with model.ema_scope():
            model.eval()
            for count, item in enumerate(fmri_embedding):
                if limit is not None:
                    if count >= limit:
                        break
                latent = item['fmri']
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w') # h w c
                print(f"rendering {num_samples} examples in {ddim_steps} steps (guidance_scale={guidance_scale}).")
                # assert latent.shape[-1] == self.fmri_latent_dim, 'dim error'

                c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))

                # Classifier-free guidance: use unconditional embeddings when guidance_scale > 1.0
                if guidance_scale > 1.0:
                    uc = torch.zeros_like(c)
                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                    conditioning=c,
                                                    batch_size=num_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=guidance_scale,
                                                    unconditional_conditioning=uc)
                else:
                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                    conditioning=c,
                                                    batch_size=num_samples,
                                                    shape=shape,
                                                    verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)

                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first


        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')

        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)


