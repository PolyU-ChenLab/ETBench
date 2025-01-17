# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.

import argparse

import nncore
import torch

from etchat.constants import DEFAULT_IMAGE_TOKEN
from etchat.conversation import get_conv
from etchat.model.builder import build_model
from etchat.utils.inference import KeywordsStoppingCriteria
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
        pred_path = nncore.join(args.pred_path, f'charades_sta_{args.index}.json')
    else:
        pred_path = nncore.join(args.pred_path, 'charades_sta.json')

    print(f'Chunk: {args.chunk} Index: {args.index} Output Path: {pred_path}')

    annos = nncore.load(args.anno_path)
    annos = [annos[i::args.chunk] for i in range(args.chunk)][args.index]

    model, tokenizer, transform = build_model(args.model_path, device=args.device)

    collect = []
    for i, anno in enumerate(nncore.ProgressBar(annos)):
        chunks, query = anno.split('##')
        vid, s, e = chunks.split(' ')
        s, e = [float(s), float(e)]
        span = [min(s, e), max(s, e)]

        data_path = nncore.join(args.data_path, vid + '.mp4')
        video, tag = load_video(data_path)
        video = transform(video).half().to(args.device)

        query = query.strip('.').strip()
        query = f"Localize the video moment according to the query '{query}'."

        conv = get_conv(model.config.conv_type)
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

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
                stopping_criteria=stopping_criteria,
                query=[[query]],
                tag=[tag])

        tokens = out[0, input_ids.size(1):]
        response = detokenize(tokens, model, tokenizer)

        if response.endswith(stop_str):
            response = response[:-len(stop_str)].strip()

        if hasattr(model, 'tgt') and model.tgt is not None:
            pred = [model.tgt[i] for i in model.match_inds]
        else:
            pred = None

        res = {'vid': vid, 'query': query, 'response': response, 'span': span, 'pred': pred}
        collect.append(res)

        if args.verbose:
            print()
            print(prompt)
            print(response)
            print(f'pred: {pred}')
            print(f'span: {span}')

    nncore.dump(collect, pred_path)
