# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.

import argparse
import copy

import nncore
import torch

from etchat.constants import DEFAULT_IMAGE_TOKEN
from etchat.conversation import get_conv
from etchat.model.builder import build_model
from etchat.utils.inference import KeywordsStoppingCriteria
from etchat.utils.io import load_image, load_video
from etchat.utils.tokenization import detokenize, tokenize

# yapf:disable
META_DATA = [
    ('Episodic Reasoning', 'episodic_reasoning.json', 'tvqa/frames_fps3_hq', 'frame'),
    ('Action Sequence', 'action_sequence.json', 'star/Charades_v1_480', 'video'),
    ('Action Prediction', 'action_prediction.json', 'star/Charades_v1_480', 'video'),
    ('Action Antonym', 'action_antonym.json', 'ssv2_video', 'video'),
    ('Fine-grained Action', 'fine_grained_action.json', 'Moments_in_Time_Raw/videos', 'video'),
    ('Unexpected Action', 'unexpected_action.json', 'FunQA_test/test', 'video'),
    ('Object Existence', 'object_existence.json', 'clevrer/video_validation', 'video'),
    ('Object Interaction', 'object_interaction.json', 'star/Charades_v1_480', 'video'),
    ('Object Shuffle', 'object_shuffle.json', 'perception/videos', 'video'),
    ('Moving Direction', 'moving_direction.json', 'clevrer/video_validation', 'video'),
    ('Action Localization', 'action_localization.json', 'sta/sta_video', 'video'),
    ('Scene Transition', 'scene_transition.json', 'scene_qa/video', 'video'),
    ('Action Count', 'action_count.json', 'perception/videos', 'video'),
    ('Moving Count', 'moving_count.json', 'clevrer/video_validation', 'video'),
    ('Moving Attribute', 'moving_attribute.json', 'clevrer/video_validation', 'video'),
    ('State Change', 'state_change.json', 'perception/videos', 'video'),
    ('Fine-grained Pose', 'fine_grained_pose.json', 'nturgbd', 'video'),
    ('Character Order', 'character_order.json', 'perception/videos', 'video'),
    ('Egocentric Navigation', 'egocentric_navigation.json', 'vlnqa', 'video'),
    ('Counterfactual Inference', 'counterfactual_inference.json', 'clevrer/video_validation', 'video')
]
# yapf:enable


def apply_template(data):
    query = f"Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question. Question: {data['question']}\nOptions:\n"
    answer = data['answer']

    for idx, c in enumerate(data['candidates']):
        query += f"({chr(ord('A') + idx)}) {c}\n"
        if c == answer:
            ans_idx = idx

    query += 'Only give the best option.'
    answer = f"({chr(ord('A') + ans_idx)}) {answer}"

    return query, answer


def parse_args():
    parser = argparse.ArgumentParser()
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

    print(f'Chunk: {args.chunk} Index: {args.index} Output Path: {args.pred_path}')

    meta_data = [META_DATA[i::args.chunk] for i in range(args.chunk)][args.index]

    model, tokenizer, transform = build_model(args.model_path, device=args.device)

    for meta in meta_data:
        anno = nncore.load(nncore.join(args.data_path, 'json', meta[1]))

        collect, hit = [], 0
        for i in nncore.ProgressBar(range(len(anno))):
            sample = copy.deepcopy(anno[i])

            video_path = nncore.join(args.data_path, 'video', meta[2], sample['video'])
            if meta[3] == 'video':
                video, tag = load_video(video_path, num_threads=1)
            else:
                paths = nncore.ls(video_path, join_path=True, ext='jpg')
                inds = list(range(1, len(paths) + 1, 3))
                frms = []
                for idx in inds:
                    frm = load_image(nncore.join(video_path, f'{idx:0>5}.jpg'))
                    frms.append(frm)
                video = torch.cat(frms)
                tag = list(range(video.size(0)))

            video = transform(video).half().to(args.device)

            query, answer = apply_template(sample)

            conv = get_conv(model.config.conv_type)
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + query)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            prompt += 'Best Option: (' if prompt[-1] == '\n' else ' Best Option: ('
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

            if response[0].lower() == answer[1].lower():
                hit += 1

            res = {'pred': response, 'task_type': meta[0], 'A': answer}
            collect.append(res)

            if args.verbose:
                print()
                print(prompt)
                print(response)
                print(answer)
                print(round(hit / (i + 1), 2))

        nncore.dump(collect, nncore.join(args.pred_path, meta[1]))
