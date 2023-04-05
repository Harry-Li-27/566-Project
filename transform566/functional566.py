import torch
from PIL import Image, ImageDraw
import numpy as np


from . import functional as F

@torch.jit.unused
def gridmask_pil(img, grid_size, interval_size, fill):
    if not F._is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    
    grid_w, grid_h = grid_size
    inter_w, inter_h = interval_size
    stride_w = inter_w + grid_w
    stride_h = inter_h + grid_h

    mask = Image.new("L", img.size, 255)
    draw = ImageDraw.Draw(mask)

    for x in range(0, img.width, stride_w):
        draw.line([(x, 0), (x, img.height)], fill=0, width=grid_w)
    for y in range(0, img.height, stride_h):
        draw.line([(0, y), (img.width, y)], fill=255, width=inter_h)

    return Image.composite(img, Image.new("RGB", img.size, fill), mask)

@torch.jit.unused
def mosaic_pil(img, grid_size, transforms):
    if not F._is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    
    cols, rows = grid_size
    patches = _split_image(img, cols, rows, transforms)
    return _merge_image(img, patches, cols, rows)

def gridmask_t(img, grid_size, interval_size, fill):
    pass

def mosaic_t(img):
    pass

def _split_image(img, cols, rows, transforms):
    # Get the dimensions of the image
    width, height = img.size
    n = len(transforms)

    # Calculate the number of patches in the x and y directions
    w = width // cols
    h = height // rows

    # Split the image into patches
    patches = [[0 for i in range(cols)] for j in range(rows)]
    for y in range(rows):
        upper = y * h
        lower = upper + h
        if y == rows - 1:
            lower = max(lower, height)
        for x in range(cols):
            left = x * w
            right = left + w
            if x == cols - 1:
                right =  max(right, width)
            
            img_t = img.crop((left, upper, right, lower))
            transform = transforms[np.random.randint(n)]
            if transform is not None:
                img_t = transform(img_t)
            patches[y][x] = (img_t)
            
    return patches

def _merge_image(img, patches, cols, rows):
    # Create a new blank image
    image = Image.new('RGB', img.size)
    
    coln = np.arange(cols)
    rown = np.arange(rows)
    np.random.shuffle(coln)
    np.random.shuffle(rown)
    
    upper = 0
    # Merge the patches back into the image
    for y in range(rows):
        left = 0
        lower = upper + patches[rown[y]][0].size[1]
        for x in range(cols):
            patch = patches[rown[y]][coln[x]]
            right = left + patch.size[0]
            image.paste(patch, (left, upper, right, lower))
            left = right
        upper = lower

    return image