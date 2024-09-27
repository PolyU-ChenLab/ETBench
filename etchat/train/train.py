# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.

import copy
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import nncore
import torch
import torch.nn as nn
import transformers
from peft import LoraConfig, get_peft_model
from tabulate import tabulate
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from etchat.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_MATCH_TOKEN, IGNORE_INDEX
from etchat.conversation import get_conv
from etchat.model import ETCHAT_MODELS
from etchat.train.etchat_trainer import ETChatTrainer
from etchat.utils.io import load_image, load_video
from etchat.utils.tokenization import tokenize
from etchat.utils.transforms import get_transform


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    language_model: Optional[str] = field(default=None)
    conv_type: Optional[str] = field(default=None)
    vision_tower: Optional[str] = field(default='eva_vit')
    vision_processor: Optional[str] = field(default='clip_center_224')
    vision_output_layer: Optional[int] = field(default=-2)
    vision_output_token: Optional[str] = field(default='patch')
    mm_projector: Optional[str] = field(default='qformer')
    pretrain_vision_tower: Optional[str] = field(default=None)
    pretrain_qformer: Optional[str] = field(default=None)
    pretrain_projector: Optional[str] = field(default=None)
    use_matching: Optional[bool] = field(default=False)
    use_time_tag: Optional[bool] = field(default=False)
    bi_attention: Optional[bool] = field(default=False)
    alpha: Optional[float] = field(default=2.0)


@dataclass
class DataArguments:
    anno_path: Optional[str] = field(default=None)
    image_path: Optional[str] = field(default=None)
    video_path: Optional[str] = field(default=None)
    video_decode_threads: Optional[int] = field(default=0)
    image_pad_to_square: Optional[bool] = field(default=False)
    gather_conv_query: Optional[bool] = field(default=False)
    fps: Optional[float] = field(default=1)
    min_video_len: Optional[int] = field(default=-1)
    max_video_len: Optional[int] = field(default=-1)
    min_num_words: Optional[int] = field(default=-1)
    max_num_words: Optional[int] = field(default=-1)
    max_retries: Optional[int] = field(default=10)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: Optional[str] = field(default='adamw_torch')
    model_max_length: Optional[int] = field(default=2048)
    legacy_tokenizer: Optional[bool] = field(default=False)
    fast_tokenizer: Optional[bool] = field(default=False)
    add_prefix_space: Optional[bool] = field(default=False)
    group_by_modality: Optional[bool] = field(default=False)
    lora_enable: Optional[bool] = field(default=False)
    lora_type: Optional[str] = field(default='attention')
    lora_r: Optional[int] = field(default=128)
    lora_alpha: Optional[int] = field(default=256)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_bias: Optional[str] = field(default='none')
    lora_lr: Optional[float] = field(default=None)
    tuning_mode: Optional[str] = field(default=None)
    save_full_model: Optional[bool] = field(default=False)
    remove_unused_columns: Optional[bool] = field(default=False)


def insert_image_token(msg):
    if random.randint(0, 1) == 0:
        return DEFAULT_IMAGE_TOKEN + '\n' + msg
    else:
        return msg + '\n' + DEFAULT_IMAGE_TOKEN


def preprocess_plain(conversations, tokenizer):
    assert len(conversations) == 2 and len(conversations[1]['value']) > 0

    query = [conversations[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()]
    prompt = DEFAULT_IMAGE_TOKEN + conversations[1]['value'] + '\n'

    input_ids = tokenize(prompt, tokenizer)

    labels = input_ids.clone()
    ins_len = tokenize(DEFAULT_IMAGE_TOKEN, tokenizer).size(0)
    labels[:ins_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels, query=query)


def preprocess_single_row(conversations, tokenizer, model_args, data_args, is_multimodal):
    conv = get_conv(model_args.conv_type)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    query = []
    for i, message in enumerate(conversations):
        role = roles[message['from']]
        assert role == conv.roles[i % 2], conversations

        msg = message['value'].strip()
        assert len(msg) > 0, conversations

        img_token_cnt = msg.count(DEFAULT_IMAGE_TOKEN)
        assert img_token_cnt in (0, 1), conversations

        if data_args.gather_conv_query:
            if is_multimodal and i == 0 and img_token_cnt == 0:
                msg = insert_image_token(msg)
            elif not is_multimodal or i > 0:
                assert img_token_cnt == 0
        else:
            if is_multimodal and role == conv.roles[0] and img_token_cnt == 0:
                msg = insert_image_token(msg)
            elif not is_multimodal or role == conv.roles[1]:
                assert img_token_cnt == 0

        if role == conv.roles[0]:
            query.append(msg.replace(DEFAULT_IMAGE_TOKEN, '').strip())

        conv.append_message(role, msg)

    if data_args.gather_conv_query:
        query = [' '.join(query)]

    prompt = conv.get_prompt()

    input_ids = tokenize(prompt, tokenizer, is_multimodal=is_multimodal)

    sep = conv.seps[0] + conv.roles[1] + ': '
    rounds = prompt.split(conv.seps[1])

    labels = input_ids.clone()
    labels[0] = IGNORE_INDEX

    cur_len = 1
    for i, rou in enumerate(rounds):
        if len(rou) == 0:
            break

        ins = sep.join(rou.split(sep)[:-1]) + sep

        rou_len = tokenize(rou, tokenizer, is_multimodal=is_multimodal).size(0)
        ins_len = tokenize(ins, tokenizer, is_multimodal=is_multimodal).size(0) - 2

        labels[cur_len:cur_len + ins_len] = IGNORE_INDEX
        cur_len += rou_len

    if labels.size(0) != cur_len:
        warnings.warn(f'Tokenization mismatch: {labels.size(0)} and {cur_len}')

    return dict(input_ids=input_ids, labels=labels, query=query)


def preprocess_multi_row(conversations, tokenizer, model_args, data_args, is_multimodal):
    conv = get_conv(model_args.conv_type)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    query = []
    for i, message in enumerate(conversations):
        role = roles[message['from']]
        assert role == conv.roles[i % 2], conversations

        msg = message['value'].strip()
        assert len(msg) > 0, conversations

        img_token_cnt = msg.count(DEFAULT_IMAGE_TOKEN)
        assert img_token_cnt in (0, 1), conversations

        if data_args.gather_conv_query:
            if is_multimodal and i == 0 and img_token_cnt == 0:
                msg = insert_image_token(msg)
            elif not is_multimodal or i > 0:
                assert img_token_cnt == 0
        else:
            if is_multimodal and role == conv.roles[0] and img_token_cnt == 0:
                msg = insert_image_token(msg)
            elif not is_multimodal or role == conv.roles[1]:
                assert img_token_cnt == 0

        if role == conv.roles[0]:
            query.append(msg.replace(DEFAULT_IMAGE_TOKEN, '').strip())

        conv.append_message(role, msg)

    if data_args.gather_conv_query:
        query = [' '.join(query)]

    prompt = conv.get_prompt()

    input_ids = tokenize(prompt, tokenizer, is_multimodal=is_multimodal)

    rounds = [m + conv.seps[0] for m in prompt.split(conv.seps[0])]
    assert (len(rounds) % 2 == 0) == (conv.system is not None)
    assert rounds[-1] == conv.seps[0]
    rounds = rounds[:-1]

    if conv.system is None:
        rounds = [''.join(rounds[i:i + 2]) for i in range(0, len(rounds), 2)]
    else:
        rounds = [''.join(rounds[:3])] + [''.join(rounds[i:i + 2]) for i in range(3, len(rounds), 2)]

    labels = input_ids.clone()

    sep = conv.seps[0] + conv.roles[1]
    cur_len = 0

    for i, rou in enumerate(rounds):
        if len(rou) == 0:
            break

        ins = sep.join(rou.split(sep)[:-1]) + sep

        rou_len = tokenize(rou, tokenizer, is_multimodal=is_multimodal).size(0)
        ins_len = tokenize(ins, tokenizer, is_multimodal=is_multimodal).size(0)

        if i > 0 and model_args.conv_type != 'qwen2':
            rou_len -= 1
            ins_len -= 1

        labels[cur_len:cur_len + ins_len] = IGNORE_INDEX
        cur_len += rou_len

    if labels.size(0) != cur_len:
        warnings.warn(f'Tokenization mismatch: {labels.size(0)} and {cur_len}')

    return dict(input_ids=input_ids, labels=labels, query=query)


def preprocess(conversations, tokenizer, model_args, data_args, is_multimodal):
    if model_args.conv_type == 'plain':
        return preprocess_plain(conversations, tokenizer)
    elif model_args.conv_type == 'vicuna_v1':
        return preprocess_single_row(conversations, tokenizer, model_args, data_args, is_multimodal)
    elif model_args.conv_type in ('phi3', 'llama3', 'gemma2', 'qwen2'):
        return preprocess_multi_row(conversations, tokenizer, model_args, data_args, is_multimodal)
    else:
        raise ValueError(f'unknown conversation type: {model_args.conv_type}')


def get_target_modules(model, lora_type, language_model):
    assert lora_type in ('linear', 'attention')

    if lora_type == 'linear':
        excludes = ['vision_tower', 'mm_projector', 'qformer', 'lm_head', 'frm_head', 'vid_head']
        target_modules = set()
        for n, m in model.named_modules():
            if any(k in n for k in excludes):
                continue
            if not isinstance(m, nn.Linear):
                continue
            target_modules.add(n.split('.')[-1])
    else:
        if language_model in ('llama', 'gemma2', 'qwen2'):
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        elif language_model == 'phi3':
            target_modules = ['qkv_proj', 'o_proj']
        else:
            raise ValueError(f'unknown language model: {language_model}')

    return list(target_modules)


class MultimodalDataset(Dataset):

    def __init__(self, tokenizer, model_args, data_args, training_args):
        super(MultimodalDataset, self).__init__()

        raw_annos = nncore.load(data_args.anno_path)

        annos = []
        for anno in raw_annos:
            num_words = len(anno['conversations'][1]['value'].split(' '))
            if data_args.min_num_words >= 0 and num_words < data_args.min_num_words:
                continue
            if data_args.max_num_words >= 0 and num_words > data_args.max_num_words:
                continue
            if data_args.min_video_len >= 0 and 'duration' in anno and anno['duration'] < data_args.min_video_len:
                continue
            if data_args.max_video_len >= 0 and 'duration' in anno and anno['duration'] > data_args.max_video_len:
                continue
            annos.append(anno)

        if training_args.local_rank in (0, -1):
            ratio = round(len(annos) / len(raw_annos) * 100, 2)
            print(f'Number of samples: {len(raw_annos)} (original) -> {len(annos)} (filtered) {ratio}%')

            tab = defaultdict(int)
            for anno in annos:
                mm_type = 'image' if 'image' in anno else 'video' if 'video' in anno else 'text'
                tab[anno.get('source', mm_type)] += 1

            tab = [[k, v, round(v / len(annos), 3)] for k, v in tab.items()]
            print(tabulate(tab, headers=['Source', '#Samples', 'Ratio'], tablefmt='pretty', stralign='left'))

        self.transform = get_transform(model_args.vision_processor)

        self.annos = annos
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.data_args = data_args

    def __len__(self):
        return len(self.annos)

    @property
    def lengths(self):
        lengths = []
        for sample in self.annos:
            cur_len = sum(c['value'].count(' ') + 1 for c in sample['conversations'])
            if 'image' not in sample and 'video' not in sample:
                cur_len = -cur_len
            lengths.append(cur_len)
        return lengths

    def __getitem__(self, idx):
        retry = 0
        while True:
            try:
                anno = copy.deepcopy(self.annos[idx])

                if self.model_args.use_time_tag and anno.get('src') is not None:
                    assert len(anno['conversations']) == 2
                    msg = anno['conversations'][0]['value']
                    msg = msg.replace(DEFAULT_MATCH_TOKEN, '{}s').format(*[round(s, 1) for s in anno.pop('src')])
                    anno['conversations'][0]['value'] = msg

                assert not ('image' in anno and 'video' in anno), anno
                mm_type = 'image' if 'image' in anno else 'video' if 'video' in anno else None

                if mm_type == 'image':
                    image_path = nncore.join(self.data_args.image_path, self.annos[idx]['image'])
                    image = load_image(image_path, self.data_args.image_pad_to_square)
                elif mm_type == 'video':
                    video_path = nncore.join(self.data_args.video_path, self.annos[idx]['video'])
                    image, tag = load_video(
                        video_path,
                        fps=self.data_args.fps,
                        min_len=self.data_args.min_video_len,
                        max_len=self.data_args.max_video_len,
                        num_threads=self.data_args.video_decode_threads)

                data = preprocess(anno['conversations'], self.tokenizer, self.model_args, self.data_args, mm_type)

                break
            except Exception as e:
                print(f'Error in loading {idx}: {e}')
                retry += 1
                if retry >= self.data_args.max_retries:
                    raise RuntimeError(f'Data loading failed after {retry} retries')
                idx = random.randint(0, len(self.annos) - 1)

        data['image'] = self.transform(image) if mm_type is not None else None
        data['tag'] = tag if mm_type == 'video' else None
        data['src'] = anno.get('src')
        data['tgt'] = anno.get('tgt')

        return data


class MultimodalDataCollator(object):

    def __init__(self, tokenizer, model_args):
        self.tokenizer = tokenizer
        self.size = int(model_args.vision_processor.split('_')[-1])

    def __call__(self, batch):
        input_ids = [d['input_ids'] for d in batch]
        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]

        labels = [d['labels'] for d in batch]
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        labels = labels[:, :self.tokenizer.model_max_length]

        data = dict(input_ids=input_ids, labels=labels, attention_mask=(input_ids != self.tokenizer.pad_token_id))

        data['image'] = [torch.zeros(1, 3, self.size, self.size) if d['image'] is None else d['image'] for d in batch]

        for key in ('query', 'tag', 'src', 'tgt'):
            data[key] = [d[key] for d in batch]

        return data


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config_cls, model_cls = ETCHAT_MODELS[model_args.language_model]

    dtype = torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    config = config_cls.from_pretrained(model_args.model_name_or_path, torch_dtype=dtype)
    config.update(model_args.__dict__)
    config.load_base_models = any((config.pretrain_vision_tower, config.pretrain_qformer, config.pretrain_projector))

    # https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/commit/b043e05
    if config.language_model == 'phi3' and config.sliding_window == 2047:
        warnings.warn('Fixing sliding window size from 2047 to 2048')
        config.sliding_window = 2048

    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        attn_implementation='eager' if config.bi_attention or config.language_model == 'gemma2' else None)

    if config.load_base_models:
        model.load_pretrained_weights()

    model.requires_grad_(False)
    model.generation_config.cache_implementation = None
    model.generation_config.repetition_penalty = None
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    if training_args.lora_enable:
        assert training_args.tuning_mode != 'llm'
        target_modules = get_target_modules(model, training_args.lora_type, config.language_model)
        print(f'LoRA target modules: {target_modules}')
        lora_config = LoraConfig(
            task_type='CAUSAL_LM',
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            target_modules=target_modules)
        model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        add_prefix_space=training_args.add_prefix_space,
        legacy=training_args.legacy_tokenizer,
        use_fast=training_args.fast_tokenizer)

    if tokenizer.pad_token is None:
        print(f'PAD token not found, using EOS token instead: {tokenizer.eos_token} ({tokenizer.eos_token_id})')
        tokenizer.pad_token = tokenizer.eos_token

    if config.use_matching:
        num_tokens = tokenizer.add_special_tokens(dict(additional_special_tokens=[DEFAULT_MATCH_TOKEN]))
        model.config.match_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_MATCH_TOKEN)
        print(f'Added {num_tokens} new token(s)')

        if num_tokens > 0 and len(tokenizer) > config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
            i_emb = model.get_input_embeddings().weight.data
            o_emb = model.get_output_embeddings().weight.data
            i_emb[-num_tokens:] = i_emb[:-num_tokens].mean(0, keepdim=True)
            o_emb[-num_tokens:] = o_emb[:-num_tokens].mean(0, keepdim=True)

    for n, p in model.named_parameters():
        if config.use_matching and any(k in n for k in ('embed_tokens', 'lm_head', 'frm_head', 'vid_head')):
            p.requires_grad = True

        if training_args.tuning_mode == 'projector':
            if 'qformer' in n and 'proj' in n or 'mm_projector' in n:
                p.requires_grad = True
        elif training_args.tuning_mode == 'attention':
            if 'qformer' in n and ('attention' in n or 'qformer_bert' not in n):
                p.requires_grad = True
        elif training_args.tuning_mode == 'qformer':
            if 'qformer' in n:
                p.requires_grad = True
        elif training_args.tuning_mode == 'llm':
            if 'vision_tower' not in n:
                p.requires_grad = True
        else:
            raise ValueError(f'unknown mode: {training_args.tuning_mode}')

    if training_args.local_rank in (0, -1):
        for n, p in model.named_parameters():
            print(p.requires_grad, p.shape, n)

        total_params = sum(p.numel() for p in model.parameters())
        learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ratio = round(learnable_params / total_params * 100, 2) if total_params > 0 else 0
        print(f'Total params: {total_params} Learnable params: {learnable_params} ({ratio}%)')

        name = tokenizer.__class__.__name__
        info = [getattr(tokenizer, k, None) for k in ('legacy', 'use_fast', 'add_prefix_space', 'add_bos_token')]
        print('Tokenizer: {}(legacy: {} fast: {} add_prefix_space: {} add_bos_token: {})'.format(name, *info))

        i_size = model.get_input_embeddings().num_embeddings
        o_size = model.get_output_embeddings().out_features
        assert i_size == o_size, (i_size, o_size)
        print(f'Tokenizer size: {len(tokenizer)} Vocab size: {config.vocab_size} Embedding size: {i_size}')

    trainer = ETChatTrainer(
        model=model,
        args=training_args,
        data_collator=MultimodalDataCollator(tokenizer, model_args),
        train_dataset=MultimodalDataset(tokenizer, model_args, data_args, training_args),
        tokenizer=tokenizer)

    has_ckpt = bool(nncore.find(training_args.output_dir, 'checkpoint-*'))
    trainer.train(resume_from_checkpoint=has_ckpt)

    trainer.save_state()
    trainer.gather_and_save_model()


if __name__ == '__main__':
    train()
