# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.

import decord
import numpy as np
import torch
from decord import DECORDError, VideoReader
from PIL import Image

from .transforms import MEAN

PAD_VALUE = tuple(int(x * 255) for x in MEAN)


def load_image(image_path, pad_to_square=False):
    img = Image.open(image_path).convert('RGB')

    if pad_to_square:
        w, h = img.size

        if w > h:
            out = Image.new(img.mode, (w, w), PAD_VALUE)
            out.paste(img, (0, (w - h) // 2))
            img = out
        elif w < h:
            out = Image.new(img.mode, (h, h), PAD_VALUE)
            out.paste(img, ((h - w) // 2, 0))
            img = out

    img = torch.from_numpy(np.array(img)).unsqueeze(0)
    return img


def _load_video(video_path, fps=1, min_len=-1, max_len=-1, **kwargs):
    decord.bridge.set_bridge('torch')

    vr = VideoReader(video_path, **kwargs)

    avg_fps = vr.get_avg_fps()
    duration = round(len(vr) / avg_fps, 3)

    if min_len >= 0 and duration < min_len:
        raise RuntimeError(f'Video {video_path} too short: {duration} < {min_len}')
    if max_len >= 0 and duration > max_len:
        raise RuntimeError(f'Video {video_path} too long: {duration} > {max_len}')

    frame_idx = np.arange(0, len(vr), avg_fps / fps).round().astype('int').clip(0, len(vr) - 1).tolist()
    video = vr.get_batch(frame_idx)

    tag = [round(i / avg_fps, 1) for i in frame_idx]

    return video, tag


def load_video(video_path, num_threads=0, **kwargs):
    if num_threads == 1:
        return _load_video(video_path, num_threads=num_threads, **kwargs)

    try:
        return _load_video(video_path, num_threads=num_threads, **kwargs)
    except DECORDError as e:
        print(f'Decord error: {e} Trying single thread...')
        return _load_video(video_path, num_threads=1, **kwargs)
