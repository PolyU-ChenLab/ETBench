# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.

import warnings

import nncore
import torch
import torch.nn as nn
from peft import PeftModel
from safetensors.torch import load_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from etchat.utils.transforms import get_transform


def build_model(model_path, device='cpu', dtype=torch.float16, **kwargs):
    adapter_path = nncore.join(model_path, 'adapter_model.safetensors')
    partial_path = nncore.join(model_path, 'pytorch_model.safetensors')

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)

    if nncore.is_file(adapter_path) or nncore.is_file(partial_path):
        print(f'Loading base model from {config.model_name_or_path}...')
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            config=config,
            attn_implementation='eager' if getattr(config, 'bi_attention', None) else None,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True,
            torch_dtype=dtype,
            **kwargs)

        try:
            model.generation_config = GenerationConfig.from_pretrained(model_path)
        except OSError:
            warnings.warn('generation_config.json not found')

        meta_state_dict = {
            n: torch.empty_like(p, device='cpu')
            for n, p in model.named_parameters() if p.device == torch.device('meta')
        }
        model.load_state_dict(meta_state_dict, strict=False, assign=True)

        if model.config.load_base_models:
            model.load_pretrained_weights()

        size = (model.model.embed_tokens.num_embeddings, model.model.embed_tokens.embedding_dim)
        if model.model.embed_tokens.weight.size() != size:
            print(f'Resizing embed_tokens to {size}...')
            model.model.embed_tokens.weight = nn.Parameter(model.model.embed_tokens.weight.new_empty(size))

        size = (model.lm_head.out_features, model.lm_head.in_features)
        if model.lm_head.weight.size() != size:
            print(f'Resizing lm_head to {size}...')
            model.lm_head.weight = nn.Parameter(model.lm_head.weight.new_empty(size))

        if nncore.is_file(adapter_path):
            print(f'Loading adapter from {model_path}...')
            model = PeftModel.from_pretrained(model, model_path, torch_dtype=dtype)

        if nncore.is_file(partial_path):
            print(f'Loading state dict from {partial_path}...')
            load_model(model, partial_path, strict=False, device=str(model.device))

        if nncore.is_file(adapter_path):
            print('Merging adapter and unloading...')
            model = model.merge_and_unload()
            model._hf_peft_config_loaded = False
    else:
        print(f'Loading full model from {model_path}...')
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            attn_implementation='eager' if getattr(config, 'bi_attention', None) else None,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
            **kwargs)

    model = model.to(device).eval()

    transform = get_transform(model.config.vision_processor, inference=True)

    return model, tokenizer, transform
