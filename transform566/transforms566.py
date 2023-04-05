import numbers
from collections.abc import Sequence

import torch
from torch import Tensor


from . import functional566 as F566
from . import functional as F

__all__ = [
    "GridMask",
    "RandomMosaic"
]

class GridMask(torch.nn.Module):
    """Generate grids that cover the image

    Args:
        grid size: [width, height]
        interval size between each grid: [width, height]
        fill: color to fill in, can be a number or a rgb tuple (0, 0, 0), range from 0 to 255
    """
    def __init__(self, grid_size=[5,5], interval_size=[5,5], fill=0):
        super().__init__()

        if not isinstance(grid_size, (Sequence)):
            raise TypeError("Grid_size should be a sequence.")
        if not isinstance(interval_size, (Sequence)):
            raise TypeError("Interval_size should be a sequence.")

        if fill is None:
            fill = (0, 0, 0)
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a number sequence or a number.")
        elif isinstance(fill, (numbers.Number)):
            if fill > 255 or fill < 0:
                raise TypeError("Fill should be in range 0 to 255")
            else:
                fill = tuple([fill for _ in range(3)])
        
        self.grid_size = grid_size
        self.interval_size = interval_size
        self.fill = fill

    def forward(self, img):
        if isinstance(img, Tensor):
            raise TypeError("Image is tensor type. Tensor method is not implemented.")
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]
            return F566.gridmask_t(img, self.grid_size, self.interval_size, self.fill)
        
        return F566.gridmask_pil(img, self.grid_size, self.interval_size, self.fill)

    def __repr__(self):
        return ""

class RandomMosaic(torch.nn.Module):
    """Create mosaic based on original image. Can do random image transform on different patches.

    Args:
        grid size: [number of col, number of row]
        transforms: a sequence of torchvision transformers. Only one transformer will be randomly applied on each patch.
    """
    def __init__(self, grid_size, transforms=[None]):
        super().__init__()
        if not isinstance(grid_size, (Sequence)):
            raise TypeError("Grid_size should be a sequence.")
        if not isinstance(transforms, (Sequence)):
            raise TypeError("Transforms should be a sequence.")
        
        self.grid_size = grid_size
        self.transform = transforms

    def forward(self, img):
        if isinstance(img, Tensor):
            raise TypeError("Image is tensor type. Tensor method is not implemented.")

        return F566.mosaic_pil(img, self.grid_size, self.transform)
    
    def __repr__(self):
        return ""