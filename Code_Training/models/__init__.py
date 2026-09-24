# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from . import vision_transformer as vits


logger = logging.getLogger("incept")


def build_model(args, only_teacher=False, num_channels=19, num_patches_per_channel=30, patch_time_dim=250):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            num_channels=num_channels,
            num_patches_per_channel=num_patches_per_channel,
            patch_time_dim=patch_time_dim,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            channel_embed_sh_degree=args.channel_embed_sh_degree,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(
        cfg.student, only_teacher=only_teacher, 
        num_channels=cfg.dataset.num_channels, 
        num_patches_per_channel=cfg.dataset.num_patches_per_channel,
        patch_time_dim=cfg.dataset.patch_time_dim
    )
