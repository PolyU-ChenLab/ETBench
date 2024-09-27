# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.

from dataclasses import dataclass
from typing import List


@dataclass
class Conversation:
    system: str
    version: str
    roles: List[str]
    seps: List[str]
    messages: List[str]

    def append_message(self, role, msg):
        self.messages.append([role, msg])

    def clear(self):
        self.messages = []

    def get_prompt(self):
        assert self.version in ('vicuna_v1', 'phi3', 'llama3', 'gemma2', 'qwen2')

        prompt = self.system + self.seps[0] if self.system is not None else ''

        for i, (role, msg) in enumerate(self.messages):
            sep = self.seps[i % 2]

            if self.version == 'vicuna_v1':
                prompt += role
                if msg is not None:
                    prompt += ': ' + msg
                    if not prompt.endswith(sep):
                        prompt += sep
                else:
                    prompt += ':'
            else:
                prompt += role
                if msg is not None:
                    prompt += msg
                    if not prompt.endswith(sep):
                        prompt += sep

        prompt = prompt.lstrip('\n')
        return prompt


def get_conv(conv_type):
    if conv_type == 'vicuna_v1':
        # yapf:disable
        conv = Conversation(
            system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
            version='vicuna_v1',
            roles=('USER', 'ASSISTANT'),
            seps=(' ', '</s>'),
            messages=[])
        # yapf:enable
    elif conv_type == 'phi3':
        conv = Conversation(
            system='<|system|>\nYou are a helpful AI assistant.',
            version='phi3',
            roles=('\n<|user|>\n', '\n<|assistant|>\n'),
            seps=('<|end|>', '<|end|>'),
            messages=[])
    elif conv_type == 'llama3':
        conv = Conversation(
            system='<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant',
            version='llama3',
            roles=('\n<|start_header_id|>user<|end_header_id|>\n', '\n<|start_header_id|>assistant<|end_header_id|>\n'),
            seps=('<|eot_id|>', '<|eot_id|>'),
            messages=[])
    elif conv_type == 'gemma2':
        conv = Conversation(
            system=None,
            version='gemma2',
            roles=('\n<start_of_turn>user\n', '\n<start_of_turn>model\n'),
            seps=('<end_of_turn>', '<end_of_turn>'),
            messages=[])
    elif conv_type == 'qwen2':
        conv = Conversation(
            system='<|im_start|>system\nYou are a helpful assistant.',
            version='qwen2',
            roles=('\n<|im_start|>user\n', '\n<|im_start|>assistant\n'),
            seps=('<|im_end|>', '<|im_end|>'),
            messages=[])
    else:
        raise ValueError(f'unknown conversation type: {conv_type}')
    return conv
