# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.

import torch.nn.functional as F
import torchvision.transforms as T

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]


class RandomResizedCrop(T.RandomResizedCrop):

    def __init__(self, size, scale=(0.8, 1.0), ratio=(1.0, 1.0)):
        self.size = (size, size) if isinstance(size, int) else size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, video):
        ratio = (self.size[0] * self.size[1]) / (video.size(-1) * video.size(-2))
        scale = (self.scale[0] * ratio, self.scale[1] * ratio)
        x, y, h, w = self.get_params(video, scale, self.ratio)
        video = video[..., x:x + h, y:y + w]
        video = F.interpolate(video, size=self.size, mode='bilinear', antialias=True)
        return video


class CenterResizedCrop:

    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, video):
        h, w = video.size(-2), video.size(-1)
        s = min(self.size[0], h, w)
        x, y = int((h - s) / 2), int((w - s) / 2)
        video = video[..., x:x + s, y:y + s]
        video = F.interpolate(video, size=self.size, mode='bilinear', antialias=True)
        return video


class Normalize:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, video):
        mean, std = video.new_tensor(self.mean), video.new_tensor(self.std)
        mean, std = mean[None, :, None, None], std[None, :, None, None]
        return (video - mean) / std


class Resize(T.Resize):

    def __init__(self, size):
        super(Resize, self).__init__(size, antialias=True)


class ToTensor:

    def __call__(self, video):
        return video.float().permute(0, 3, 1, 2) / 255


def get_transform(name, inference=False):
    model_type, crop_type, size = name.split('_')
    size = int(size)

    assert model_type == 'clip'
    assert crop_type in ('center', 'random')
    assert isinstance(size, int)

    crop_cls = CenterResizedCrop if inference or crop_type == 'center' else RandomResizedCrop
    transform = T.Compose([ToTensor(), Resize(size), crop_cls(size), Normalize(MEAN, STD)])

    return transform
