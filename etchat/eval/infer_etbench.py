# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.

import argparse
import copy

import nncore
import torch

from etchat.constants import DEFAULT_IMAGE_TOKEN
from etchat.conversation import get_conv
from etchat.model.builder import build_model
from etchat.utils.inference import KeywordsStoppingCriteria, RepetitionPenaltyLogitsProcessor
from etchat.utils.io import load_video
from etchat.utils.tokenization import detokenize, tokenize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_path')
    parser.add_argument('--data_path')
    parser.add_argument('--pred_path')
    parser.add_argument('--model_path')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.chunk > 1:
        pred_path = nncore.join(args.pred_path, f'etbench_{args.index}.json')
    else:
        pred_path = nncore.join(args.pred_path, 'etbench.json')

    print(f'Chunk: {args.chunk} Index: {args.index} Output Path: {pred_path}')

    if args.anno_path.endswith('json'):
        anno = nncore.load(args.anno_path)
    else:
        path = nncore.ls(args.anno_path, ext='json', join_path=True, sort=True)
        anno = nncore.flatten([nncore.load(p) for p in path])

    anno = [anno[i::args.chunk] for i in range(args.chunk)][args.index]

    model, tokenizer, transform = build_model(args.model_path, device=args.device)

    for i in nncore.ProgressBar(range(len(anno))):
        sample = copy.deepcopy(anno[i])

        video = nncore.join(args.data_path, sample['video'])
        video, tag = load_video(video, num_threads=1)
        video = transform(video).half().to(args.device)

        query, src = sample['q'], sample.get('src')

        conv = get_conv(model.config.conv_type)
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if sample['task'] in ['rar', 'eca', 'rvq']:
            logits_processor = [RepetitionPenaltyLogitsProcessor(1.2, model.config.match_token_id)]
            prompt += 'Best Option: (' if prompt[-1] == '\n' else ' Best Option: ('
        else:
            logits_processor = None

        input_ids = tokenize(prompt, tokenizer).unsqueeze(0).to(args.device)

        stop_str = conv.seps[-1]
        stopping_criteria = [KeywordsStoppingCriteria(tokenizer, stop_str)]

        with torch.inference_mode():
            out = model.generate(
                input_ids,
                image=[video],
                do_sample=True,
                temperature=0.2,
                max_new_tokens=2048,
                cache_implementation=None,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                query=[[query]],
                src=[src],
                tag=[tag])

        tokens = out[0, input_ids.size(1):]
        response = detokenize(tokens, model, tokenizer)

        if response.endswith(stop_str):
            response = response[:-len(stop_str)].strip()

        anno[i]['a'] = response

        if args.verbose:
            print()
            print(prompt)
            print(response)
            print(src)
            print(f"src: {anno[i].get('src')}")
            print(f"tgt: {anno[i].get('tgt')}")
            print(f"p: {anno[i].get('p')}")
            print(f"g: {anno[i].get('g')}")

    nncore.dump(anno, pred_path)
