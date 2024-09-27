# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.

import re
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import BertTokenizer, DynamicCache

from etchat.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from .qformer import BertConfig, BertLMHeadModel
from .vision_encoder.builder import build_vision_tower


def _get_w(weight, key):
    return {k.split(key + '.')[1]: v for k, v in weight.items() if key in k}


def _cache_state(module, args):
    assert isinstance(args, tuple) and len(args) == 1, args
    module.cache_state = args[0]


class ETChatMetaModel:

    def __init__(self, config):
        super(ETChatMetaModel, self).__init__(config)

        self.vision_tower = build_vision_tower(config)
        self.build_mm_projector(config)

        self.norm.register_forward_pre_hook(_cache_state)

    def build_mm_projector(self, config):
        if config.mm_projector == 'qformer':
            assert config.vision_output_token == 'patch'
            self.qformer_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation_side='left')
            self.qformer_tokenizer.add_special_tokens(dict(bos_token='[DEC]'))

            bert_config = BertConfig.from_pretrained('bert-base-uncased')
            bert_config.encoder_width = vision_size = self.vision_tower.hidden_size
            bert_config.add_cross_attention = True
            bert_config.cross_attention_freq = 2

            self.qformer_bert = BertLMHeadModel(bert_config)
            self.qformer_bert.resize_token_embeddings(len(self.qformer_tokenizer))
            self.qformer_query = nn.Parameter(torch.zeros(1, 32, bert_config.hidden_size))
            self.qformer_query.data.normal_(std=bert_config.initializer_range)
            self.qformer_ln = nn.LayerNorm(vision_size)

            self.qformer_proj = nn.Linear(self.qformer_bert.config.hidden_size, vision_size)
            self.qformer_q_proj = nn.Sequential(nn.LayerNorm(vision_size), nn.Linear(vision_size, vision_size))
            self.qformer_k_proj = nn.Sequential(nn.LayerNorm(vision_size), nn.Linear(vision_size, vision_size))
            self.qformer_v_proj = nn.Sequential(nn.LayerNorm(vision_size), nn.Linear(vision_size, config.hidden_size))
        else:
            depth = int(re.match(r'^mlp(\d+)x_gelu$', config.mm_projector).group(1))
            modules = [nn.Linear(vision_size, config.hidden_size)]
            for _ in range(1, depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            self.mm_projector = nn.Sequential(*modules)

    def load_weights(self):
        if self.config.pretrain_vision_tower is not None:
            print(f'Loading vision tower weights from {self.config.pretrain_vision_tower}...')
            self.vision_tower.load_weights(self.config.pretrain_vision_tower)

        if self.config.mm_projector == 'qformer' and self.config.pretrain_qformer is not None:
            print(f'Loading qformer weights from {self.config.pretrain_qformer}...')
            weight = torch.load(self.config.pretrain_qformer, map_location='cpu')['model']
            self.qformer_bert.load_state_dict(_get_w(weight, 'Qformer'))
            self.qformer_ln.load_state_dict(_get_w(weight, 'ln_vision'))
            self.qformer_query.data = weight['query_tokens']

        if self.config.pretrain_projector is not None:
            print(f'Loading projector weights from {self.config.pretrain_projector}...')
            weight = load_file(self.config.pretrain_projector, device='cpu')
            for key in ('qformer_proj', 'qformer_q_proj', 'qformer_k_proj', 'qformer_v_proj', 'mm_projector'):
                if hasattr(self, key):
                    getattr(self, key).load_state_dict(_get_w(weight, key))

    def _update_causal_mask(self, *args, **kwargs):
        mask = super(ETChatMetaModel, self)._update_causal_mask(*args, **kwargs)
        if self.config.bi_attention:
            for idx, seps in enumerate(self.img_token_pos):
                for sep in seps:
                    mask[idx, :, sep[0]:sep[1], sep[0]:sep[1]] = 0
        return mask


class ETChatMetaForCausalLM:

    def post_init(self):
        if self.config.use_matching:
            hidden_size = self.config.hidden_size
            self.frm_head = nn.Sequential(
                nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size // 2), nn.GELU(),
                nn.Linear(hidden_size // 2, hidden_size))
            self.vid_head = nn.Sequential(
                nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size // 2), nn.GELU(),
                nn.Linear(hidden_size // 2, hidden_size))
        super(ETChatMetaForCausalLM, self).post_init()

    def load_pretrained_weights(self):
        self.model.load_weights()

    def encode_image(self, image, **kwargs):
        img_embed = self.model.vision_tower(image)

        if self.config.vision_output_token == 'patch':
            img_embed = img_embed[:, 1:]
        elif self.config.vision_output_token == 'cls':
            img_embed = img_embed[:, :1]

        if self.config.mm_projector == 'qformer':
            img_embed = self.compress_image_qformer(img_embed, **kwargs)
        else:
            img_embed = self.compress_image_pooling(img_embed, **kwargs)

        return img_embed

    def compress_image_pooling(self, img_embed, img_cnt=None, **kwargs):
        img_embed = img_embed.mean(1).unsqueeze(0)
        img_embed = self.model.mm_projector(img_embed)

        img_embeds, cur_cnt = [], 0
        for cnt in img_cnt:
            img_embeds.append(img_embed[:, cur_cnt:cur_cnt + cnt])
            cur_cnt += cnt

        return img_embeds

    def compress_image_qformer(self, img_embed, img_cnt=None, query=None, tag=None):
        img_masks = img_embed.new_ones(img_embed.size()[:-1], dtype=torch.long)

        img_embeds, cur_cnt = [], 0
        for cnt, query_text, tag_text in zip(img_cnt, query, tag):
            assert isinstance(query_text, list)

            if self.config.use_time_tag and tag_text is not None:
                assert len(query_text) == 1, query
                query_text = [f'Current time: {t}s. {query_text[0]}' for t in tag_text]

            inputs = self.model.qformer_tokenizer(
                query_text, padding='longest', truncation=True, max_length=256,
                return_tensors='pt').to(img_embed.device)

            input_ids, query_att = inputs.input_ids, inputs.attention_mask

            img_enc_feat = img_embed[cur_cnt:cur_cnt + cnt]
            img_enc_feat = img_enc_feat[None].expand(len(query_text), -1, -1, -1).flatten(0, 1)
            img_att_mask = img_masks[cur_cnt:cur_cnt + cnt]
            img_att_mask = img_att_mask[None].expand(len(query_text), -1, -1).flatten(0, 1)
            if not self.config.use_time_tag or tag_text is None:
                input_ids = input_ids[:, None].expand(-1, cnt, -1).flatten(0, 1)
                query_att = query_att[:, None].expand(-1, cnt, -1).flatten(0, 1)
            cur_cnt += cnt

            img_hidden_state = img_enc_feat.clone()

            query_embeds = self.model.qformer_query.expand(img_hidden_state.shape[0], -1, -1)
            query_att_mask = torch.cat((query_att.new_ones(query_embeds.shape[:-1]), query_att), dim=1)
            img_hidden_state = self.model.qformer_ln(img_hidden_state)

            outputs = self.model.qformer_bert.bert(
                input_ids,
                query_embeds=query_embeds,
                attention_mask=query_att_mask,
                encoder_hidden_states=img_hidden_state,
                encoder_attention_mask=img_att_mask)

            query_embeds = self.model.qformer_proj(outputs[0][:, :query_embeds.shape[1]])

            q_embed = self.model.qformer_q_proj(query_embeds)
            k_embed = self.model.qformer_k_proj(img_enc_feat)

            att = torch.matmul(q_embed, k_embed.transpose(-1, -2)) / (img_enc_feat.shape[-1]**0.5)
            att = att.softmax(-1)
            emb = torch.matmul(att, img_enc_feat)
            emb = (emb + query_embeds).mean(1, keepdim=True)
            emb = self.model.qformer_v_proj(emb)

            emb = emb.reshape(len(query_text), cnt, *emb.shape[-2:])
            emb = emb.flatten(1, 2)

            img_embeds.append(emb)

        return img_embeds

    def prepare_multimodal_data(self, input_ids, attention_mask, past_key_values, labels, image, query, src, tag, mode):
        if image is None or mode != 'generating':
            self.model.img_token_pos = [[] for _ in range(input_ids.size(0))]

        if image is None or mode == 'generating':
            if image is not None and mode == 'generating':
                if attention_mask.dim() == 4:
                    assert attention_mask.size(0) == attention_mask.size(1) == attention_mask.size(2) == 1
                    sep = (attention_mask == 0).sum().item() + self.cache_img_state.size(0)
                    attention_mask[0, 0, 0, :sep] = 0
                else:
                    attention_mask = attention_mask.new_ones(
                        (attention_mask.size(0), past_key_values[-1][-1].size(-2) + 1))
            return input_ids, attention_mask, None, labels

        assert isinstance(image, list)
        img_cnt = [img.size(0) for img in image]
        image = torch.cat(image)
        img_embed = self.encode_image(image, img_cnt=img_cnt, query=query, tag=tag)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                dummy_img_state = img_embed[batch_idx][0][:0]
                cur_input_embeds = self.model.embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat((cur_input_embeds, dummy_img_state))
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                continue

            img_token_inds = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if img_token_inds.numel() != img_embed[batch_idx].shape[0]:
                warnings.warn(f'Image token mismatch: {img_token_inds.numel()} and {img_embed[batch_idx].shape[0]}')
                assert mode == 'training', cur_input_ids

            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            cur_new_input_embeds, img_idx = [], 0
            while img_token_inds.numel() > 0:
                img_pos = img_token_inds[0]
                cur_img_state = img_embed[batch_idx][img_idx]
                self.cache_img_state = cur_img_state
                self.model.img_token_pos[batch_idx].append([img_pos.item(), img_pos.item() + cur_img_state.shape[0]])
                non_img_tokens = self.model.embed_tokens(cur_input_ids[:img_pos])

                if self.config.use_matching:
                    if src is not None and src[batch_idx] is not None:
                        inds = torch.where(cur_input_ids[:img_pos] == self.config.match_token_id)[0]
                        if inds.shape[0] > 0:
                            assert len(src[batch_idx]) == inds.shape[0], (src, inds)
                            src_ind = cur_img_state.new_tensor(src[batch_idx]).round().long().detach()
                            max_idx = cur_img_state.shape[0] - 1
                            s = src_ind.clamp(min=0, max=max_idx)
                            non_img_tokens[inds] = non_img_tokens[inds] + cur_img_state[s]
                    else:
                        assert not (cur_input_ids[:img_pos] == self.config.match_token_id).any().item()

                cur_new_input_embeds.append(non_img_tokens)
                cur_new_input_embeds.append(cur_img_state)
                cur_input_ids = cur_input_ids[img_pos + 1:]
                if labels is not None:
                    cur_new_labels.append(cur_labels[:img_pos])
                    cur_new_labels.append(labels.new_full((cur_img_state.shape[0], ), IGNORE_INDEX))
                    cur_labels = cur_labels[img_pos + 1:]

                img_token_inds = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
                img_idx += 1

            if cur_input_ids.numel() > 0:
                non_img_tokens = self.model.embed_tokens(cur_input_ids)

                if self.config.use_matching:
                    if src is not None and src[batch_idx] is not None:
                        inds = torch.where(cur_input_ids == self.config.match_token_id)[0]
                        if inds.shape[0] > 0:
                            assert len(src[batch_idx]) == inds.shape[0], (src, inds)
                            src_ind = cur_img_state.new_tensor(src[batch_idx]).round().long().detach()
                            max_idx = cur_img_state.shape[0] - 1
                            s = src_ind.clamp(min=0, max=max_idx)
                            non_img_tokens[inds] = non_img_tokens[inds] + cur_img_state[s]
                    else:
                        assert not (cur_input_ids == self.config.match_token_id).any().item()

                cur_new_input_embeds.append(non_img_tokens)
                if labels is not None:
                    cur_new_labels.append(cur_labels)

            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)

            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (cur_new_embed, cur_new_embed.new_zeros(
                        (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]))))
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (cur_new_label, cur_new_label.new_full((max_len - cur_new_label.shape[0], ), IGNORE_INDEX)))
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align)

            if attention_mask is not None:
                new_att_mask = []
                for cur_att_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_s = attention_mask.new_full((cur_new_labels.shape[0] - labels.shape[1], ), True)
                    new_attn_mask_pad_e = attention_mask.new_full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0], ), False)
                    cur_new_att_mask = torch.cat((new_attn_mask_pad_s, cur_att_mask, new_attn_mask_pad_e))
                    new_att_mask.append(cur_new_att_mask)
                attention_mask = torch.stack(new_att_mask)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds)
            if labels is not None:
                new_labels = torch.stack(new_labels)

            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    min_dtype = torch.finfo(attention_mask.dtype).min
                    attention_mask = attention_mask.new_full(
                        (*attention_mask.shape[:2], new_input_embeds.shape[1], attention_mask.shape[3]), min_dtype)
                    attention_mask = torch.triu(attention_mask, diagonal=1)
                else:
                    new_attn_mask_pad = attention_mask.new_full(
                        (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True)
                    attention_mask = torch.cat((new_attn_mask_pad, attention_mask), dim=1)
                    assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, new_input_embeds, new_labels

    def forward(self,
                input_ids=None,
                attention_mask=None,
                past_key_values=None,
                cache_position=None,
                labels=None,
                image=None,
                query=None,
                src=None,
                tgt=None,
                tag=None,
                **kwargs):
        assert self.training == (past_key_values is None)

        if self.training:
            mode = 'training'
            assert src is not None and tgt is not None
            _src = []
            for s, t in zip(src, tgt):
                _s = []
                _s += s if s is not None else []
                _s += t if t is not None else []
                _s = None if len(_s) == 0 else _s
                _src.append(_s)
            src = _src
        else:
            assert input_ids.size(0) == 1 and tgt is None
            assert isinstance(past_key_values, DynamicCache)
            mode = 'caching' if len(past_key_values) == 0 else 'generating'
            if mode == 'generating':
                assert input_ids.size(1) == 1
                src = None

        input_ids, attention_mask, inputs_embeds, labels = self.prepare_multimodal_data(
            input_ids, attention_mask, past_key_values, labels, image, query, src, tag, mode)

        if self.config.use_matching and image is not None and mode == 'generating':
            if input_ids.item() == self.config.match_token_id:
                inputs_embeds = self.model.embed_tokens(input_ids)
                inputs_embeds = inputs_embeds + self.cache_img_state[self.tgt[-1], None, None]
                input_ids = None

        if mode == 'caching':
            cache_position = torch.arange(0, inputs_embeds.size(1), device=inputs_embeds.device)
        elif mode == 'generating':
            cache_position = cache_position + self.cache_img_state.size(0) - 1

        assert kwargs.pop('inputs_embeds', None) is None
        kwargs['use_cache'] = mode != 'training'
        kwargs['position_ids'] = None
        kwargs['output_hidden_states'] = True
        kwargs['return_dict'] = True

        outputs = super(ETChatMetaForCausalLM, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs)

        if self.config.use_matching and image is not None:
            # decoder block -> -2 -> decoder block -> cache state -> norm -> -1
            frm_tokens_all = self.frm_head(self.model.norm.cache_state)
            vid_tokens_all = self.vid_head(outputs.hidden_states[-2])

            if mode != 'training':
                if mode == 'caching':
                    self.cache_frm_tokens_all = frm_tokens_all
                    self.sim, self.tgt = [], []
                img_token_pos = self.model.img_token_pos[0][-1]
                assert img_token_pos[1] - img_token_pos[0] == self.cache_img_state.size(0)
                frm_tokens = self.cache_frm_tokens_all[0, img_token_pos[0]:img_token_pos[1]]
                vid_tokens = vid_tokens_all[0][-1, None]
                sim = torch.matmul(vid_tokens, frm_tokens.t())[0].softmax(0).cpu()
                tgt = sim.argmax().item()
                self.sim.append(sim)
                self.tgt.append(tgt)

            if labels is not None and tgt is not None:
                loss_match, avg_factor = 0, 0
                shift_labels = labels[..., 1:].contiguous()
                assert len(self.model.img_token_pos) == len(tgt)
                for i, (seps, ts) in enumerate(zip(self.model.img_token_pos, tgt)):
                    if ts is None or len(ts) == 0:
                        continue

                    sep = seps[-1]
                    tgt_idx = frm_tokens_all.new_tensor(ts)
                    max_idx = sep[1] - sep[0] - 1
                    tgt_idx[tgt_idx < 0] = 0
                    tgt_idx[tgt_idx > max_idx] = max_idx

                    frm_tokens = frm_tokens_all[i, sep[0]:sep[1]]
                    idx_vec = torch.arange(frm_tokens.size(0), device=frm_tokens.device)[None].repeat(len(ts), 1)
                    sim_tgt = torch.pow(self.config.alpha, -(idx_vec - tgt_idx[:, None]).abs())

                    inds = torch.where(shift_labels[i] == self.config.match_token_id)[0][-tgt_idx.size(0):]
                    vid_tokens = vid_tokens_all[i][inds]
                    sim = torch.matmul(vid_tokens, frm_tokens.t()) / vid_tokens.size(1)**0.5

                    loss_match = loss_match + F.cross_entropy(sim, sim_tgt)
                    avg_factor += 1

                if avg_factor > 0:
                    outputs.loss = outputs.loss + loss_match / avg_factor
                else:
                    outputs.loss = outputs.loss + (frm_tokens_all + vid_tokens_all).mean() * 0

        return outputs

    def prepare_inputs_for_generation(self, *args, image=None, query=None, src=None, tag=None, **kwargs):
        model_inputs = super(ETChatMetaForCausalLM, self).prepare_inputs_for_generation(*args, **kwargs)
        model_inputs.update({'image': image, 'query': query, 'src': src, 'tag': tag})
        return model_inputs
