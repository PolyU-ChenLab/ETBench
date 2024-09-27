# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.

import warnings

import nncore
import torch
from deepspeed import zero
from safetensors.torch import load_model, save_file
from torch.utils.data import Sampler
from transformers import Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


def gather(param):
    if hasattr(param, 'ds_id'):
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def gather_lora_params(model, bias):
    assert bias in ('lora_only', 'all', 'none')

    if bias == 'lora_only':
        state_dict, maybe_lora_bias, lora_bias_names = dict(), dict(), set()
        for n, p in model.named_parameters():
            if 'lora_' in n:
                state_dict[n] = p
                bias_name = n.split('lora_')[0] + 'bias'
                lora_bias_names.add(bias_name)
            elif 'bias' in n:
                maybe_lora_bias[n] = p
        for n, p in maybe_lora_bias:
            if bias_name in lora_bias_names:
                state_dict[bias_name] = p
    elif bias == 'all':
        state_dict = {n: p for n, p in model.named_parameters() if 'lora_' in n or 'bias' in n}
    else:
        state_dict = {n: p for n, p in model.named_parameters() if 'lora_' in n}

    state_dict = {n: gather(p) for n, p in state_dict.items()}
    return state_dict


def gather_non_lora_params(model):
    state_dict = {n: p for n, p in model.named_parameters() if p.requires_grad and 'lora_' not in n}
    state_dict = {n: gather(p) for n, p in state_dict.items()}
    return state_dict


def split_to_chunks(inds, lengths, num_chunks):
    if len(inds) % num_chunks != 0:
        return [inds[i::num_chunks] for i in range(num_chunks)]

    chunk_size = len(inds) // num_chunks
    chunks = [[] for _ in range(num_chunks)]
    chunks_len = [0 for _ in range(num_chunks)]
    for idx in inds:
        min_idx = chunks_len.index(min(chunks_len))
        chunks[min_idx].append(idx)
        chunks_len[min_idx] += lengths[idx]
        if len(chunks[min_idx]) == chunk_size:
            chunks_len[min_idx] = float('inf')

    return chunks


def generate_length_grouped_inds(batch_size, world_size, lengths):
    inds = torch.randperm(len(lengths))
    mega_bs = world_size * batch_size
    megabatches = [inds[i:i + mega_bs].tolist() for i in range(0, len(lengths), mega_bs)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_chunks(megabatch, lengths, world_size) for megabatch in megabatches]
    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def generate_modality_grouped_inds(batch_size, world_size, lengths):
    assert all(l != 0 for l in lengths)

    m_inds, m_len = zip(*[(i, abs(l)) for i, l in enumerate(lengths) if l > 0])
    t_inds, t_len = zip(*[(i, abs(l)) for i, l in enumerate(lengths) if l < 0])

    assert len(m_inds) > 0 and len(t_inds) > 0

    m_shuffle = [m_inds[i] for i in generate_length_grouped_inds(batch_size, world_size, m_len)]
    t_shuffle = [t_inds[i] for i in generate_length_grouped_inds(batch_size, world_size, t_len)]
    mega_bs = world_size * batch_size
    m_mega_batch = [m_shuffle[i:i + mega_bs] for i in range(0, len(m_shuffle), mega_bs)]
    t_mega_batch = [t_shuffle[i:i + mega_bs] for i in range(0, len(t_shuffle), mega_bs)]

    if len(m_mega_batch[-1]) < mega_bs:
        m_mega_batch = m_mega_batch[:-1]

    if len(t_mega_batch[-1]) < mega_bs:
        t_mega_batch = t_mega_batch[:-1]

    megabatches = m_mega_batch + t_mega_batch
    megabatch_inds = torch.randperm(len(megabatches))
    megabatches = [megabatches[i] for i in megabatch_inds]

    return [i for megabatch in megabatches for i in megabatch]


class GroupSampler(Sampler):

    def __init__(self, batch_size, world_size, lengths):
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        inds = generate_modality_grouped_inds(self.batch_size, self.world_size, self.lengths)
        return iter(inds)


class ETChatTrainer(Trainer):

    def _get_train_sampler(self):
        if self.args.group_by_modality:
            return GroupSampler(self.args.train_batch_size, self.args.world_size, self.train_dataset.lengths)
        else:
            return super(ETChatTrainer, self)._get_train_sampler()

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            model = self.model

        super(ETChatTrainer, self)._load_from_checkpoint(resume_from_checkpoint, model=model)

        partial_path = nncore.join(resume_from_checkpoint, 'pytorch_model.safetensors')
        if nncore.is_file(partial_path):
            load_model(model, partial_path, strict=False, device=model.device)

    def create_optimizer(self):
        if self.optimizer is None:
            grad_params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

            decay_params = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
            decay_params = [n for n in decay_params if 'bias' not in n]

            if self.args.lora_lr is None:
                groups = [{
                    'params': [p for n, p in grad_params if (n in decay_params)],
                    'weight_decay': self.args.weight_decay
                }, {
                    'params': [p for n, p in grad_params if (n not in decay_params)],
                    'weight_decay': 0.0
                }]
            else:
                lora_params = [n for n, _ in grad_params if 'lora' in n]
                groups = [{
                    'params': [p for n, p in grad_params if (n in decay_params and n not in lora_params)],
                    'weight_decay': self.args.weight_decay
                }, {
                    'params': [p for n, p in grad_params if (n not in decay_params and n not in lora_params)],
                    'weight_decay': 0.0
                }, {
                    'params': [p for n, p in grad_params if (n in decay_params and n in lora_params)],
                    'weight_decay': self.args.weight_decay,
                    'lr': self.args.lora_lr
                }, {
                    'params': [p for n, p in grad_params if (n not in decay_params and n in lora_params)],
                    'weight_decay': 0.0,
                    'lr': self.args.lora_lr
                }]

            optim_cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optim_cls(groups, **kwargs)

        return self.optimizer

    def gather_and_save_model(self):
        deepspeed_zero3 = self.accelerator.deepspeed_config['zero_optimization']['stage'] == 3
        output_dir = self.args.output_dir

        if self.args.save_full_model and self.args.lora_enable and deepspeed_zero3:
            warnings.warn('LoRA models cannot be saved in full mode under zero3, saving adapters instead')
            self.args.save_full_model = False

        if self.args.save_full_model:
            if self.args.lora_enable:
                self.model = self.model.merge_and_unload()

            self.model.config.load_base_models = False

            if deepspeed_zero3 and not self.model_wrapped.zero_gather_16bit_weights_on_model_save():
                warnings.warn('Saving zero checkpoint, use zero_to_fp32.py to recover weights')
                self.model_wrapped.save_checkpoint(output_dir)
                return

            if deepspeed_zero3:
                state_dict = self.model_wrapped._zero3_consolidated_16bit_state_dict()
            else:
                state_dict = self.model.state_dict()

            if self.args.should_save:
                state_dict = {k[17:] if k.startswith('base_model.model.') else k: v for k, v in state_dict.items()}
                self._save(output_dir, state_dict=state_dict)
        else:
            if self.args.lora_enable:
                state_dict = gather_lora_params(self.model, self.args.lora_bias)
                if self.args.should_save:
                    self.model.save_pretrained(output_dir, state_dict=state_dict)

            if self.args.should_save:
                self.model.config.save_pretrained(output_dir)
                self.model.generation_config.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)

            state_dict = gather_non_lora_params(self.model)
            if self.args.should_save and state_dict:
                save_file(state_dict, nncore.join(output_dir, 'pytorch_model.safetensors'))

    def _save_checkpoint(self, model, trial, metrics=None):
        super(ETChatTrainer, self)._save_checkpoint(model, trial, metrics=metrics)

        if self.args.lora_enable:
            output_dir = self._get_output_dir(trial)
            output_dir = nncore.join(output_dir, f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}')

            state_dict = gather_non_lora_params(self.model)
            if self.args.should_save:
                self.model.config.save_pretrained(output_dir)
                save_file(state_dict, nncore.join(output_dir, 'pytorch_model.safetensors'))
