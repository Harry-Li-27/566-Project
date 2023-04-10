import torch
from torch import Tensor

import torchvision.transforms as T
from PIL import Image, ImageOps, ImageFilter

from numpy import random

def Normalize():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])


def Flip_colorjitter():
    return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
        ])

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def dino_g1():
    return T.Compose([
            T.RandomResizedCrop(224, scale=(0.4, 1), interpolation=Image.BICUBIC),
            Flip_colorjitter(),
            GaussianBlur(1.0),
            Normalize(),
        ])

def dino_g2():
    return T.Compose([
            T.RandomResizedCrop(224, scale=(0.4, 1), interpolation=Image.BICUBIC),
            Flip_colorjitter(),
            GaussianBlur(0.1),
            Solarization(0.2),
            Normalize(),
        ])

def dino_l():
    return T.Compose([
            T.RandomResizedCrop(96, scale=(0.05, 0.4), interpolation=Image.BICUBIC),
            Flip_colorjitter(),
            GaussianBlur(0.5),
            Normalize(),
        ])

def moco_1():
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.08, 1.)),
        Flip_colorjitter(),
        GaussianBlur(1.0),
        Normalize()
    ])

def moco_2():
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.08, 1.)),
        Flip_colorjitter(),
        GaussianBlur(0.1),
        Solarization(0.2),
        Normalize()
    ])

def skew():
    return T.Compose([
            T.RandomPerspective(0.6, 1),
            T.CenterCrop(150),
            T.Resize(224, interpolation=Image.BICUBIC),
            Normalize()
    ])

def grid(): 
    return T.Compose([
        T.GridMask(),
        Normalize()
    ])

def randmosaic():
    return T.Compose([
        T.RandomMosaic((3,3)),
        Normalize()
    ])

def localize(Transform):
    return T.Compose([
        Transform, 
        T.RandomResizedCrop(96, scale=(0.05, 0.4), interpolation=Image.BICUBIC)
    ])

def randcrop():
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.08, 1.)),
        Normalize()
    ])

def colorjit():
    return T.Compose([
        T.RandomApply(
            [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            p=0.8
        ),
        T.RandomGrayscale(p=0.2),
        Normalize()
    ])

transdict = {
    "":None,
    "skew": skew(),
    "grid": grid(),
    "randmosaic": randmosaic(),
    "randcrop":randcrop(),
    "colorjit":colorjit(),
    "dino_g1":dino_g1(),
    "dino_g2":dino_g2(),
    "dino_l":dino_l(),
    "moco_1":moco_1(),
    "moco_2":moco_2()
}

class trans_566(object):
    def __init__(self, t1="", t2="", moco=False):
        self.t1 = transdict[t1]
        self.t2 =  transdict[t1] if t2 == "" else transdict[t2]
        if moco:
            self.t3 = localize(self.t1)
        else:
            self.t3 = None
        print(f"Debug: transform1 is {t1}, transform2 is {t2}. Training on moco: {moco}.\n")

# t = trans_566("randcrop","randcrop")
# t.t1, t.t2, t.t3

# trans_566("randcrop","colorjit")
# trans_566("randcrop","skew")
# trans_566("randcrop","grid")
# trans_566("randcrop","randmosaic")

# trans_566("colorjit","colorjit")
# trans_566("colorjit","skew")
# trans_566("colorjit","grid")
# trans_566("colorjit","randmosaic")

# trans_566("skew","skew")
# trans_566("skew","grid")
# trans_566("skew","randmosaic")

# trans_566("grid","grid")
# trans_566("grid","randmosaic")

# trans_566("randmosaic","randmosaic")

# 