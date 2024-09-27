# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.

from .eva_vit import EVAVisionTransformer


def build_vision_tower(config):
    if config.vision_tower == 'eva_vit':
        return EVAVisionTransformer(config.vision_output_layer)
    else:
        raise ValueError(f'Unknown vision tower: {config.vision_tower}')
