# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.

import argparse

import nncore
import torch
from nncore.ops import temporal_iou


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    log_file = nncore.join(args.pred_path, 'metrics.log')
    nncore.set_default_logger(logger='eval', fmt=None, log_file=log_file)

    pred_paths = nncore.ls(args.pred_path, ext='json', join_path=True)
    nncore.log(f'Total number of files: {len(pred_paths)}')

    total, failed, iou_sum, hit_3, hit_5, hit_7 = 0, 0, 0, 0, 0, 0
    for path in pred_paths:
        data = nncore.load(path)

        for sample in data:
            total += 1

            if sample.get('pred') is not None and len(sample['pred']) == 2:
                iou = temporal_iou(torch.Tensor([sample['span']]), torch.Tensor([sample['pred']]))
                iou = iou.item() if iou.isfinite() else 0
            else:
                failed += 1
                continue

            iou_sum += iou

            if iou >= 0.3:
                hit_3 += 1

            if iou >= 0.5:
                hit_5 += 1

            if iou >= 0.7:
                hit_7 += 1

    nncore.log(f'R@0.3: {round(hit_3 / total, 3)} R@0.5: {round(hit_5 / total, 3)} R@0.7: {round(hit_7 / total, 3)}')
    nncore.log(f'mIoU: {round(iou_sum / total, 3)}')
    nncore.log(f'Failed: {failed} samples ({failed / total * 100:.1f}%)')
