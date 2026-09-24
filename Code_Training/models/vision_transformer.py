# Code_Training/models/vision_transformer.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
from scipy.special import sph_harm

from Code_Training.layers import Mlp, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from Code_Training.layers.patch_embed import EEGResidualPatchEmbed

logger = logging.getLogger("incept")

def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x

def get_eeg_coords():
    """
    Standard 10-20 system coordinates for 19 channels.
    """
    coords_dict = {
        'Fp1': [-0.0294367,  0.0839171, -0.00699  ], 
        'Fp2': [ 0.0298723,  0.0848959, -0.00708  ],
        'F3':  [-0.0502438,  0.0531112,  0.042192 ], 
        'F4':  [ 0.0518362,  0.0543048,  0.040814 ],
        'C3':  [-0.0653581, -0.0116317,  0.064358 ], 
        'C4':  [ 0.0671179, -0.0109003,  0.06358  ],
        'P3':  [-0.0530073, -0.0787878,  0.05594  ], 
        'P4':  [ 0.0556667, -0.0785602,  0.056561 ],
        'O1':  [-0.0294134, -0.1124490,  0.008839 ], 
        'O2':  [ 0.0298426, -0.1121560,  0.0088   ],
        'F7':  [-0.0702629,  0.0424743, -0.01142  ], 
        'F8':  [ 0.0730431,  0.0444217, -0.012    ],
        'T3':  [-0.0841611, -0.0160187, -0.009346 ], 
        'T4':  [ 0.0850799, -0.0150203, -0.00949  ],
        'T5':  [-0.0724343, -0.0734527, -0.002487 ], 
        'T6':  [ 0.0730557, -0.0730683, -0.00254  ],
        'Fz':  [ 0.0003122,  0.0585120,  0.066462 ], 
        'Cz':  [ 0.0004009, -0.0091670,  0.100244 ],
        'Pz':  [ 0.0003247, -0.0811150,  0.082615 ]
    }
    return torch.tensor(list(coords_dict.values()), dtype=torch.float32)

def compute_spherical_harmonics(coords, degree):
    """
    Args:
        coords: (N, 3) Cartesian coordinates [x, y, z]
        degree: int, max degree L
    Returns:
        torch.Tensor: (N, (degree+1)**2) float32
    """
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # 1. Calc radius (r) for normalization
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.maximum(r, 1e-8)  # Avoid div by zero

    # 2. Cartesian -> Spherical (Physics convention)
    # theta: polar angle [0, pi], requires normalized z (z/r)
    theta = np.arccos(np.clip(z / r, -1, 1)) 
    # phi: azimuth angle [0, 2pi]
    phi = np.arctan2(y, x)

    feats = []
    # 3. Compute SH for each (l, m)
    for l in range(degree + 1):
        for m in range(-l, l + 1):
            # Scipy args: m, l, azimuth(phi), polar(theta)
            Y = sph_harm(m, l, phi, theta)
            # Complex -> Real SH
            if m < 0:
                Y_real = np.sqrt(2) * (-1)**m * np.imag(Y)
            elif m == 0:
                Y_real = np.real(Y)
            else: # m > 0
                Y_real = np.sqrt(2) * (-1)**m * np.real(Y)
            feats.append(Y_real)

    # (N, (degree+1)**2)
    return torch.from_numpy(np.stack(feats, axis=-1)).float()

class GeoResNetChannelEmbed(nn.Module):
    def __init__(self, coords, embed_dim, sh_degree=3):
        super().__init__()
        self.register_buffer("channel_coords", coords)   # (19, 3)
        sh_feats = compute_spherical_harmonics(coords, degree=sh_degree)
        self.register_buffer("channel_spherical_harmonics", sh_feats)   # (19, 16)

        # Path A: Project spherical harmonics to embedding space (only focus on angle)
        sh_dim = (sh_degree + 1) ** 2
        self.sh_proj = nn.Linear(sh_dim, embed_dim)
        
        # Path B: Residual MLP for raw coordinate
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim)
        )
        
        # Init weights: Start with weak residual to rely on SH geometry first
        nn.init.trunc_normal_(self.sh_proj.weight, std=0.02)
        nn.init.zeros_(self.sh_proj.bias)
        nn.init.trunc_normal_(self.coord_mlp[0].weight, std=0.02)
        nn.init.trunc_normal_(self.coord_mlp[2].weight, std=1e-6)
        nn.init.zeros_(self.coord_mlp[2].bias)

    def forward(self, idxs):
        """
        idxs: (B, C_crop)
        """
        # 1. Lookup spherical harmonics
        sh_input = self.channel_spherical_harmonics[idxs]  # (B, C_crop, 16)
        coord_input = self.channel_coords[idxs]  # (B, C_crop, 3)
        dtype = self.sh_proj.weight.dtype

        # 2. Learnable Projections
        # Path A: spherical harmonics
        base_embed = self.sh_proj(sh_input.to(dtype=dtype))  # (B, C_crop, 16) -> (B, C_crop, 768)
        
        # Path B: raw coordinate
        res_embed = self.coord_mlp(coord_input.to(dtype=dtype))  # (B, C_crop, 3) -> (B, C_crop, 768)
        
        # 3. Residual Fusion
        return base_embed + res_embed   # (B, C_crop, 768)


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        # EEG Specific Arguments
        patch_time_dim=250,          # T: Number of time samples per patch
        num_channels=19,             # C: Total number of EEG channels available
        num_patches_per_channel=30,  # N: Total number of time patches per channel
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=EEGResidualPatchEmbed,  # use time-freq concat patch embedding!
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        channel_embed_sh_degree=4,  # degree of spherical harmonics for channel embedding
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens

        self.patch_time_dim = patch_time_dim
        self.num_channels = num_channels
        self.num_patches_per_channel = num_patches_per_channel


        # Initialize EEGPatchEmbed
        self.patch_embed = embed_layer(
            in_chans=patch_time_dim, 
            embed_dim=embed_dim, 
        )

        # --- EEG Positional Embeddings (Geo-ResNet) ---
        # 1. Channel Embedding: Hybrid SH + MLP
        self.channel_embed_sh_degree = channel_embed_sh_degree
        coords = get_eeg_coords()
        if len(coords) != num_channels:
            raise ValueError(f"Length of pre-defined channel coordinates {len(coords)} must match number of channels {num_channels}")
        self.channel_embed = GeoResNetChannelEmbed(coords, embed_dim, sh_degree=self.channel_embed_sh_degree)
        logger.info(f"using Spherical Harmonics Channel Embedding with degree {self.channel_embed_sh_degree}")

        # 2. Time Embedding: Sinusoidal and Learnable weight
        self.time_embed_temperature = 10000.0
        self.time_embed_init_scale = 1.0
        self.time_embed_scale = nn.Parameter(
            torch.ones(embed_dim) * self.time_embed_init_scale
        )
        logger.info(f"using Learnable Sinusoidal Positional Embedding (Channel-wise gating) with temperature {self.time_embed_temperature}")

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = np.linspace(0, drop_path_rate, depth).tolist()

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        else:
            raise NotImplementedError

        layer_info = ", ".join([
            f"layer{i}:{'channel-wise' if i % 2 == 0 else 'global'}" 
            # f"layer{i}:{'global'}" 
            for i in range(depth)
        ])
        logger.info(f"Building transformer blocks - Total layers: {depth}, Layer types: [{layer_info}]")
        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
                # attn_type="global",
                attn_type="channel-wise" if i % 2 == 0 else "global",
                num_register_tokens=num_register_tokens,
            )
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.chunked_blocks = False

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        self.init_weights()

    def init_weights(self):
        # GeoResNetChannelEmbed initializes itself in its __init__, so we don't touch it here
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def build_time_embed(self, time_idxs):
        D = self.embed_dim
        div_term = torch.exp(torch.arange(0, D, 2, device=time_idxs.device).float() * (-math.log(self.time_embed_temperature) / D))
        args = time_idxs.unsqueeze(-1) * div_term 
        return torch.stack([torch.sin(args), torch.cos(args)], dim=-1).flatten(-2)

    def prepare_tokens_with_masks(self, x, masks=None, ch_idxs=None, time_idxs=None):
        # 1. Patch Embedding
        x = self.patch_embed(x)  # (B, T, C, N) -> (B, C*N, D)
        
        # ch_idxs: (B, C) - IDs of channels in this batch/crop
        # time_idxs: (B, N) - IDs of time patches in this batch/crop
        if ch_idxs is None or time_idxs is None:
             raise ValueError("ch_idxs and time_idxs must be provided for EEG positional embedding.")
        B, L, D = x.shape
        C = ch_idxs.shape[1]
        N = time_idxs.shape[1]
        assert L == C * N, f"Patch embedding output length {L} does not match C({C})*N({N})"
        x = x.view(B, C, N, D)  # (B, C*N, D) -> (B, C, N, D)

        # 2. Masking (Apply only to patch tokens, before adding CLS)
        if masks is not None:
            masks = masks.view(B, C, N).unsqueeze(-1)  # (B, C*N) -> (B, C, N) -> (B, C, N, 1)
            mask_token = self.mask_token.view(1, 1, 1, -1)  # (1, D) -> (1, 1, 1, D)
            x = torch.where(masks, mask_token.to(x.dtype), x)

        # 3. Positional Embeddings
        ch_emb = self.channel_embed(ch_idxs)  # (B, C) -> (B, C, D)
        t_emb = self.build_time_embed(time_idxs).to(dtype=x.dtype) * self.time_embed_scale  # (B, N) -> (B, N, D) -> *(D) -> (B, N, D)
        x = x + ch_emb.unsqueeze(2) + t_emb.unsqueeze(1)  # (B, C, N, D) + (B, C, 1, D) + (B, 1, N, D) -> (B, C, N, D)

        # 4. Insert CLS Tokens (One per Channel)
        # Expand generic CLS token to (B, C, 1, D)
        cls_tokens = self.cls_token.expand(B, C, 1, -1)  # (1, 1, 1, D) -> (B, C, 1, D)
        cls_tokens = cls_tokens + ch_emb.unsqueeze(2)  # (B, C, 1, D) + (B, C, 1, D) -> (B, C, 1, D)
        x = torch.cat((cls_tokens, x), dim=2)  # (B, C, 1, D) concat (B, C, N, D) -> (B, C, 1+N, D)
        x = x.flatten(1, 2)  # (B, C, 1+N, D) -> (B, C*(1+N), D)

        # 5. Register Tokens
        if self.register_tokens is not None:
            x = torch.cat((self.register_tokens.expand(B, -1, -1), x), dim=1)  # (1, N_reg, D) -> (B, N_reg, D) concat (B, C*(1+N), D) -> (B, N_reg+C*(1+N), D)

        return x

    def forward_features_list(self, x_list, masks_list, ch_idxs_list, time_idxs_list):
        x = [
            self.prepare_tokens_with_masks(x, masks, ch_idxs, time_idxs)
            for x, masks, ch_idxs, time_idxs in zip(x_list, masks_list, ch_idxs_list, time_idxs_list)
        ]   # list of (B, N_reg+C*(1+N), D)
        num_channels_list = [ch_idxs.shape[1] for ch_idxs in ch_idxs_list]
        for i, blk in enumerate(self.blocks):
            x = blk(x, num_channels=num_channels_list)

        output = []
        for x_out, masks, ch_idxs in zip(x, masks_list, ch_idxs_list):
            x_norm = self.norm(x_out)     # (B, N_reg+C*(1+N), D)
            reg_tokens = x_norm[:, :self.num_register_tokens]  # (B, N_reg, D)
            x_main = x_norm[:, self.num_register_tokens:]  # (B, C*(1+N), D)
            B, _, D = x_main.shape; C = ch_idxs.shape[1]
            x_main = x_main.view(B, C, -1, D)  # (B, C*(1+N), D) -> (B, C, 1+N, D)
            cls_tokens = x_main[:, :, 0, :]   # (B, C, D)
            patch_tokens = x_main[:, :, 1:, :].flatten(1, 2) # (B, C, N, D) -> (B, C*N, D)
            output.append(
                {
                    "x_norm_clstoken": cls_tokens,      # (B, C, D)
                    "x_norm_regtokens": reg_tokens,     # (B, N_reg, D)
                    "x_norm_patchtokens": patch_tokens, # (B, C*N, D)
                    "x_prenorm": x_out,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None, ch_idxs=None, time_idxs=None):
        if isinstance(x, list):
            masks_list = masks if masks is not None else [None] * len(x)
            ch_idxs_list = ch_idxs if ch_idxs is not None else [None] * len(x)
            time_idxs_list = time_idxs if time_idxs is not None else [None] * len(x)
            return self.forward_features_list(x, masks_list, ch_idxs_list, time_idxs_list)

        x = self.prepare_tokens_with_masks(x, masks, ch_idxs, time_idxs)  # (B, N_reg+C*(1+N), D)
        for blk in self.blocks:
            x = blk(x, num_channels=ch_idxs.shape[1])
        x_norm = self.norm(x)     # (B, N_reg+C*(1+N), D)
        reg_tokens = x_norm[:, :self.num_register_tokens]  # (B, N_reg, D)
        x_main = x_norm[:, self.num_register_tokens:]  # (B, C*(1+N), D)
        B, _, D = x_main.shape; C = ch_idxs.shape[1]
        x_main = x_main.view(B, C, -1, D)  # (B, C*(1+N), D) -> (B, C, 1+N, D)
        cls_tokens = x_main[:, :, 0, :]   # (B, C, D)
        patch_tokens = x_main[:, :, 1:, :].flatten(1, 2) # (B, C, N, D) -> (B, C*N, D)
        return {
            "x_norm_clstoken": cls_tokens,      # (B, C, D)
            "x_norm_regtokens": reg_tokens,     # (B, N_reg, D)
            "x_norm_patchtokens": patch_tokens, # (B, C*N, D)
            "x_prenorm": x,
            "masks": masks,
        }

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


    # def _get_intermediate_layers_not_chunked(self, x, n=1, ch_idxs=None, time_idxs=None):
    #     x = self.prepare_tokens_with_masks(x, ch_idxs=ch_idxs, time_idxs=time_idxs)  # x: (B, T, C, N) -> (B, N_reg+C*(1+N), D)
    #     output, total_block_len = [], len(self.blocks)
    #     blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
    #     for i, blk in enumerate(self.blocks):
    #         x = blk(x, num_channels=ch_idxs.shape[1])
    #         if i in blocks_to_take:
    #             output.append(x)
    #     assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
    #     return output
    
    # def get_intermediate_layers(
    #     self,
    #     x: torch.Tensor,
    #     n: Union[int, Sequence] = 1,
    #     return_class_token: bool = True,
    #     norm: bool = True,
    #     ch_idxs=None, 
    #     time_idxs=None
    # ):
    #     if ch_idxs is None or time_idxs is None:
    #         raise ValueError("ch_idxs and time_idxs must be provided in kwargs for get_intermediate_layers")
    #     outputs = self._get_intermediate_layers_not_chunked(x, n, ch_idxs=ch_idxs, time_idxs=time_idxs)  # x: (B, T, C, N) -> (B, N_reg+C*(1+N), D)
    #     if norm:
    #         outputs = [self.norm(out) for out in outputs]

    #     cls_tokens_list = []
    #     patch_tokens_list = []
    #     C = ch_idxs.shape[1]
    #     for out in outputs:  # for each layer
    #         x_main = out[:, self.num_register_tokens:]  # (B, N_reg+C*(1+N), D) -> (B, C*(1+N), D)
    #         B, _, D = x_main.shape
    #         x_main = x_main.view(B, C, -1, D)  # (B, C*(1+N), D) -> (B, C, 1+N, D)
    #         cls_tokens = x_main[:, :, 0, :]  # (B, C, D)
    #         cls_tokens_list.append(cls_tokens)
    #         patch_tokens = x_main[:, :, 1:, :].flatten(1, 2)  # (B, C, N, D) -> (B, C*N, D)
    #         patch_tokens_list.append(patch_tokens)

    #     # ((Patch_1layer, CLS_1layer), (Patch_2layer, CLS_2layer), ...) 正序，从中间层到最后一层
    #     # where each Patch_xlayer is (B, C*N, D) and each CLS_xlayer is (B, C, D)
    #     if return_class_token:
    #         return tuple(zip(patch_tokens_list, cls_tokens_list))  
    #     return tuple(patch_tokens_list)

    def _get_intermediate_layers_not_chunked(self, x, n=1, ch_idxs=None, time_idxs=None):
        x = self.prepare_tokens_with_masks(x, ch_idxs=ch_idxs, time_idxs=time_idxs)   # (B, T, C, N) -> (B, N_reg + C*(1+N), D)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x, num_channels=ch_idxs.shape[1])  # (B, N_reg + C*(1+N), D)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        return_class_token: bool = True,
        norm: bool = True,
        ch_idxs=None, 
        time_idxs=None
    ):
        if ch_idxs is None or time_idxs is None:
            raise ValueError("ch_idxs and time_idxs must be provided for EEG positional embedding.")
        outputs = self._get_intermediate_layers_not_chunked(x, n, ch_idxs=ch_idxs, time_idxs=time_idxs)  # list of [ (B, N_reg + C*(1+N), D) ]
        if norm:
            outputs = [self.norm(out) for out in outputs]

        cls_tokens_list = []
        patch_tokens_list = []
        C = ch_idxs.shape[1]
        for out in outputs:
            x_main = out[:, self.num_register_tokens:]   # (B, N_reg + C*(1+N), D) -> (B, C*(1+N), D)
            B, _, D = x_main.shape
            x_main = x_main.view(B, C, -1, D)  # (B, C*(1+N), D) -> (B, C, 1+N, D)
            cls_tokens = x_main[:, :, 0, :]  # (B, C, 1+N, D) -> (B, C, D)
            cls_tokens_list.append(cls_tokens)
            patch_tokens = x_main[:, :, 1:, :]  # (B, C, 1+N, D) -> (B, C, N, D)
            patch_tokens = patch_tokens.flatten(1, 2)  # (B, C, N, D) -> (B, C*N, D)
            patch_tokens_list.append(patch_tokens)

        if return_class_token:
            return tuple(zip(patch_tokens_list, cls_tokens_list))
        return tuple(patch_tokens_list)


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_base(num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model