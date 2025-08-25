# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .rasterizer import CurveFragments


# A compositor should take as input 3D lines and some corresponding information.
# Given this information, the compositor can:
#     - blend colors across the top K lines at a pixel

kEpsilon = 1e-8

class SilhouetteCompositor(nn.Module):
    """
    Accumulate lines to generate silhouettes similar to SoftRasterizer.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, fragments: CurveFragments, blur_radius, **kwargs) -> torch.Tensor:
        blur_radius += kEpsilon
        weights = (1 - fragments.dists / blur_radius)**16
        weights[weights > 1] = 0
        # images are of shape (C, H, W)
        image = 1 - torch.prod(1 - weights, dim=-1, keepdim=True)
        image = image.permute(2, 0, 1).clip(0, 1)

        return image