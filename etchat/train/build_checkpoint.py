# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.

import argparse

from etchat.model.builder import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    model, tokenizer, _ = build_model(args.model_path)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
