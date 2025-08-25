# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional, Tuple, Union
from torch import Tensor

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d import _C

from ..utils import parse_image_size


# TODO make the epsilon user configurable
kEpsilon = 1e-8

# Maximum number of lines per bins for
# coarse-to-fine rasterization
kMaxLinesPerBin = 22


def rasterize_curves(
    curves,
    image_size: Union[int, List[int], Tuple[int, int]] = 256,
    blur_radius: float = 0.0,
    lines_per_pixel: int = 8,
    bin_size: Optional[int] = None,
    max_lines_per_bin: Optional[int] = None,
    perspective_correct: bool = False,
    clip_barycentric_coords: bool = False,
):
    """
    Rasterize a batch of Curves given the shape of the desired output image.
    Each curve is rasterized onto a separate image of shape
    (H, W) if `image_size` is a tuple or (image_size, image_size) if it
    is an int.

    If the desired image size is non square (i.e. a tuple of (H, W) where H != W)
    the aspect ratio needs special consideration. There are two aspect ratios
    to be aware of:
        - the aspect ratio of each pixel
        - the aspect ratio of the output image
    The camera can be used to set the pixel aspect ratio. In the rasterizer,
    we assume square pixels, but variable image aspect ratio (i.e rectangle images).

    In most cases you will want to set the camera aspect ratio to
    1.0 (i.e. square pixels) and only vary the
    `image_size` (i.e. the output image dimensions in pixels).

    Args:
        curves: A curves object representing a batch of curves, batch size N.
        image_size: Size in pixels of the output image to be rasterized.
            Can optionally be a tuple of (H, W) in the case of non square images.
        blur_radius: Float distance in the range [0, 2] used to expand the line
            bounding boxes for rasterization. Setting blur radius
            results in blurred edges around the shape instead of a
            hard boundary. Set to 0 for no blur.
        lines_per_pixel (Optional): Number of lines to save per pixel, returning
            the nearest lines_per_pixel points along the z-axis.
        bin_size: Size of bins to use for coarse-to-fine rasterization. Setting
            bin_size=0 uses naive rasterization; setting bin_size=None attempts to
            set it heuristically based on the shape of the input. This should not
            affect the output, but can affect the speed of the forward pass.
        max_lines_per_bin: Only applicable when using coarse-to-fine rasterization
            (bin_size > 0); this is the maximum number of lines allowed within each
            bin. This should not affect the output values, but can affect
            the memory usage in the forward pass.
        perspective_correct: Bool, Whether to apply perspective correction when computing
            barycentric coordinates for pixels. This should be set to True if a perspective
            camera is used.
        clip_barycentric_coords: Whether, after any perspective correction is applied
            but before the depth is calculated (e.g. for z clipping),
            to "correct" a location outside the line (i.e. with a negative
            barycentric coordinate) to a position on the edge of the line.

    Returns:
        4-element tuple containing

        - **pix_to_line**: LongTensor of shape (image_size, image_size, lines_per_pixel)
          giving the indices of the nearest lines at each pixel, sorted in ascending z-order.
          Concretely ``pix_to_line[y, x, k] = l`` means that ``lines_verts[l]`` is the kth
          closest line (in the z-direction) to pixel (y, x). Pixels that are hit by fewer than
          lines_per_pixel are padded with -1.
        - **zbuf**: FloatTensor of shape (image_size, image_size, lines_per_pixel)
          giving the NDC z-coordinates of the nearest lines at each pixel,
          sorted in ascending z-order. Concretely, if ``pix_to_line[y, x, k] = l`` then
          ``zbuf[y, x, k] = line_verts[l, 2]``. Pixels hit by fewer than lines_per_pixel
          are padded with -1.
        - **barycentric**: FloatTensor of shape (image_size, image_size, lines_per_pixel, 3)
          giving the barycentric coordinates in NDC units of the nearest lines at each pixel,
          sorted in ascending z-order. Concretely, if ``pix_to_line[y, x, k] = l`` then
          ``t = barycentric[y, x, k]`` gives the barycentric coords for pixel
          (y, x) relative to the line defined by ``line_verts[l]``. Pixels hit by fewer than
          lines_per_pixel are padded with -1.
        - **pix_dists**: FloatTensor of shape (image_size, image_size, lines_per_pixel)
          giving the signed Euclidean distance (in NDC units) in the x/y plane of each point
          closest to the pixel. Concretely if ``pix_to_line[y, x, k] = l`` then
          ``pix_dists[y, x, k]`` is the squared distance between the pixel (y, x) and
          the line given by vertices ``line_verts[l]``. Pixels hit with fewer than
          ``lines_per_pixel`` are padded with -1.

        In the case that image_size is a tuple of (H, W) then the outputs
        will be of shape `(H, W, ...)`.
    """
    points_packed = curves.points_packed()
    lines_packed = curves.lines_packed()
    line_verts = points_packed[lines_packed]

    # In the case that H != W use the max image size to set the bin_size
    # to accommodate the num bins constraint in the coarse rasterizer.
    # If the ratio of H:W is large this might cause issues as the smaller
    # dimension will have fewer bins.
    # TODO: consider a better way of setting the bin size.
    im_size = parse_image_size(image_size)
    max_image_size = max(*im_size)
    
    # TODO: Choose naive vs coarse-to-fine based on mesh size and image size.
    if bin_size is None:
        if not points_packed.is_cuda:
            # Binned CPU rasterization is not supported.
            bin_size = 0
        else:
            # TODO better heuristics for bin size.
            if max_image_size <= 64:
                bin_size = 8
            else:
                # Heuristic based formula maps max_image_size -> bin_size as follows:
                # max_image_size < 64 -> 8
                # 16 < max_image_size < 256 -> 16
                # 256 < max_image_size < 512 -> 32
                # 512 < max_image_size < 1024 -> 64
                # 1024 < max_image_size < 2048 -> 128
                bin_size = int(2 ** max(np.ceil(np.log2(max_image_size)) - 4, 4))

    if bin_size != 0:
        # There is a limit on the number of lines per bin in the cuda kernel.
        lines_per_bin = 1 + (max_image_size - 1) // bin_size
        if lines_per_bin >= kMaxLinesPerBin:
            raise ValueError(
                "bin_size too small, number of lines per bin must be less than %d; got %d"
                % (kMaxLinesPerBin, lines_per_bin)
            )

    if max_lines_per_bin is None:
        max_lines_per_bin = int(max(10000, lines_packed.shape[0] / 5))

    pix_to_line, zbuf, barycentric_coords, dists = _RasterizeLineVerts.apply(
        line_verts,
        im_size,
        blur_radius,
        lines_per_pixel,
        bin_size,
        max_lines_per_bin,
        perspective_correct,
        clip_barycentric_coords,
    )

    return pix_to_line, zbuf, barycentric_coords, dists


class _RasterizeLineVerts(torch.autograd.Function):
    """
    Torch autograd wrapper for forward and backward pass of rasterize_meshes
    implemented in C++/CUDA.

    Args:
        line_verts: Tensor of shape (L, 2, 3) giving (packed) vertex positions
            for lines in all the meshes in the batch. Concretely,
            line_verts[l, i] = [x, y, z] gives the coordinates for the
            ith vertex of the fth line. These vertices are expected to
            be in NDC coordinates in the range [-1, 1].
        image_size, blur_radius, lines_per_pixel: same as rasterize_meshes.
        perspective_correct: same as rasterize_meshes.

    Returns:
        same as rasterize_meshes function.
    """

    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        ctx,
        line_verts: torch.Tensor,
        image_size: Union[List[int], Tuple[int, int]] = (256, 256),
        blur_radius: float = 0.01,
        lines_per_pixel: int = 0,
        bin_size: int = 0,
        max_lines_per_bin: int = 0,
        perspective_correct: bool = False,
        clip_barycentric_coords: bool = False,
    ):
        # pyre-fixme[16]: Module `pytorch3d` has no attribute `_C`.
        pix_to_line, zbuf, barycentric_coords, dists = _C.rasterize_curves(
            line_verts,
            image_size,
            blur_radius,
            lines_per_pixel,
            bin_size,
            max_lines_per_bin,
            perspective_correct,
            clip_barycentric_coords,
        )

        ctx.save_for_backward(line_verts, pix_to_line)
        ctx.mark_non_differentiable(pix_to_line)
        ctx.perspective_correct = perspective_correct
        ctx.clip_barycentric_coords = clip_barycentric_coords
        return pix_to_line, zbuf, barycentric_coords, dists

    @staticmethod
    def backward(ctx, grad_pix_to_line, grad_zbuf, grad_barycentric_coords, grad_dists):
        grad_line_verts = None
        grad_image_size = None
        grad_radius = None
        grad_lines_per_pixel = None
        grad_bin_size = None
        grad_max_lines_per_bin = None
        grad_perspective_correct = None
        grad_clip_barycentric_coords = None
        line_verts, pix_to_line = ctx.saved_tensors
        grad_line_verts = _C.rasterize_curves_backward(
            line_verts,
            pix_to_line,
            grad_zbuf,
            grad_barycentric_coords,
            grad_dists,
            ctx.perspective_correct,
            ctx.clip_barycentric_coords,
        )
        grads = (
            grad_line_verts,
            grad_image_size,
            grad_radius,
            grad_lines_per_pixel,
            grad_bin_size,
            grad_max_lines_per_bin,
            grad_perspective_correct,
            grad_clip_barycentric_coords,
        )
        return grads


def non_square_ndc_range(S1, S2):
    """
    In the case of non square images, we scale the NDC range
    to maintain the aspect ratio. The smaller dimension has NDC
    range of 2.0.

    Args:
        S1: dimension along with the NDC range is needed
        S2: the other image dimension

    Returns:
        ndc_range: NDC range for dimension S1
    """
    ndc_range = 2.0
    if S1 > S2:
        ndc_range = (S1 / S2) * ndc_range
    return ndc_range


def pix_to_non_square_ndc(i, S1, S2):
    """
    The default value of the NDC range is [-1, 1].
    However in the case of non square images, we scale the NDC range
    to maintain the aspect ratio. The smaller dimension has NDC
    range from [-1, 1] and the other dimension is scaled by
    the ratio of H:W.
    e.g. for image size (H, W) = (64, 128)
       Height NDC range: [-1, 1]
       Width NDC range: [-2, 2]

    Args:
        i: pixel position on axes S1
        S1: dimension along with i is given
        S2: the other image dimension

    Returns:
        pixel: NDC coordinate of point i for dimension S1
    """
    # NDC: x-offset + (i * pixel_width + half_pixel_width)
    ndc_range = non_square_ndc_range(S1, S2)
    offset = ndc_range / 2.0
    return -offset + (ndc_range * i + offset) / S1


def rasterize_curves_python_coarse(
    line_verts,
    image_size: Union[int, Tuple[int, int]] = 256,
    blur_radius: float = 0.0,
    lines_per_pixel: int = 8,
    bin_size: Optional[int] = None,
    max_lines_per_bin: Optional[int] = None,
):
    H, W = image_size if isinstance(image_size, tuple) else (image_size, image_size)
    K, M = lines_per_pixel, max_lines_per_bin
    device = line_verts.device
    
    BH = 1 + (H - 1) // bin_size
    BW = 1 + (W - 1) // bin_size

    # Initialize output tensors.
    bin_lines = torch.full(
        (BH, BW, M), fill_value=-1, dtype=torch.int32, device=device
    )
    
    # Calculate all line bounding boxes.
    line_x_mins = torch.min(line_verts[:, :, 0], dim=1).values
    line_x_maxs = torch.max(line_verts[:, :, 0], dim=1).values
    line_y_mins = torch.min(line_verts[:, :, 1], dim=1).values
    line_y_maxs = torch.max(line_verts[:, :, 1], dim=1).values
    line_z_mins = torch.min(line_verts[:, :, 2], dim=1).values
    
    line_x_mins = line_x_mins - np.sqrt(blur_radius) - kEpsilon
    line_x_maxs = line_x_maxs + np.sqrt(blur_radius) + kEpsilon
    line_y_mins = line_y_mins - np.sqrt(blur_radius) - kEpsilon
    line_y_maxs = line_y_maxs + np.sqrt(blur_radius) + kEpsilon
    
    # Calculate the bin ranges.
    pix_width_y = non_square_ndc_range(H, W) / H
    bin_width_y = pix_width_y * bin_size
    pix_width_x = non_square_ndc_range(W, H) / W
    bin_width_x = pix_width_x * bin_size
    
    byi, bxi = torch.meshgrid(torch.arange(BH, device=device),
                            torch.arange(BW, device=device), indexing='ij')
    byi, bxi = byi.unsqueeze(dim=-1), bxi.unsqueeze(dim=-1)
    bin_y_mins = pix_to_non_square_ndc(byi * bin_size, H, W) - pix_width_y/2
    bin_y_maxs = pix_to_non_square_ndc((byi + 1) * bin_size - 1, H, W) + pix_width_y/2
    bin_x_mins = pix_to_non_square_ndc(bxi * bin_size, W, H) - pix_width_x/2
    bin_x_maxs = pix_to_non_square_ndc((bxi + 1) * bin_size - 1, W, H) + pix_width_x/2
    
    valid = (
        (line_x_mins <= bin_x_maxs) +
        (line_x_maxs > bin_x_mins) + 
        (line_y_mins <= bin_y_maxs) +
        (line_y_maxs > bin_y_mins) +
        (line_z_mins > kEpsilon)
    )
    
    torch.sort()
    
    all_lines = torch.arange(line_verts.shape[0], device=device)
    all_lines = all_lines.unsqueeze(dim=0).repeat(BH, BW, 1)
    

def rasterize_curves_python(  # noqa: C901
    curves,
    image_size: Union[int, Tuple[int, int]] = 256,
    blur_radius: float = 0.0,
    lines_per_pixel: int = 8,
    perspective_correct: bool = False,
    clip_barycentric_coords: bool = False,
):
    """
    Naive PyTorch implementation of curve rasterization with the same inputs and
    outputs as the rasterize_curves function.

    This function is used as a comparison for the C++/CUDA implementations.
    """
    N = len(curves)
    H, W = image_size if isinstance(image_size, tuple) else (image_size, image_size)

    K = lines_per_pixel
    device = curves.device

    points_packed = curves.points_packed()
    lines_packed = curves.lines_packed()
    
    # Initialize output tensors.
    pix_to_line = torch.full(
        (H, W, K), fill_value=-1, dtype=torch.int64, device=device
    )
    zbuf = torch.full((H, W, K), fill_value=-1, dtype=torch.float32, device=device)
    bary_coords = torch.full(
        (H, W, K), fill_value=-1, dtype=torch.float32, device=device
    )
    pix_dists = torch.full(
        (H, W, K), fill_value=-1, dtype=torch.float32, device=device
    )

    yi, xi = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device), indexing='ij')
    # +Y is pointing up in the image, +X is pointing to the left in the image.
    yf = pix_to_non_square_ndc(H - 1 - yi, H, W)
    xf = pix_to_non_square_ndc(W - 1 - xi, W, H)
    
    # Calculate all line bounding boxes.
    line_verts = points_packed[lines_packed]
    line_idxs = torch.arange(lines_packed.shape[0], device=device)
    
    v0 = line_verts[:, 0, :]
    v1 = line_verts[:, 1, :]
    
    # Compute barycentric coordinates and distance.
    pxy = torch.stack([xf, yf], dim=-1).unsqueeze(dim=2)
    t, dist = point_line_barycentric_distance(pxy, v0[:, :2], v1[:, :2])
    
    # use correctted and clipped barycentric coords to calculate the z value
    if perspective_correct:
        z0, z1 = v0[:, 2], v1[:, 2]
        denom = z1 * t + z0 * (1 - t) + kEpsilon
        t = z1 * t / denom
    if clip_barycentric_coords:
        t = t.clamp(0, 1)
    pz = t * v0[:, 2] + (1 - t) * v1[:, 2]
    
    # exclude invalid lines
    pz_max_clip = 1 / kEpsilon
    pz[dist > blur_radius] = pz_max_clip
    
    # sort the lines by pz to the pixel
    pz_sorted, idxs_sorted = torch.sort(pz, dim=-1)
    pz_sorted_k, idxs_sorted_k = pz_sorted[..., :K], idxs_sorted[..., :K]
    
    mask = pz_sorted_k < pz_max_clip
    # Save to output tensors.
    zbuf[mask] = pz_sorted_k[mask]
    pix_to_line[mask] = line_idxs[idxs_sorted_k][mask]
    bary_coords[mask] = t.gather(dim=-1, index=idxs_sorted_k)[mask]
    pix_dists[mask] = dist.gather(dim=-1, index=idxs_sorted_k)[mask]

    return pix_to_line, zbuf, bary_coords, pix_dists


def rasterize_curves_python_simple(  # noqa: C901
    curves,
    image_size: Union[int, Tuple[int, int]] = 256,
    blur_radius: float = 0.0,
    lines_per_pixel: int = 8,
    perspective_correct: bool = False,
    clip_barycentric_coords: bool = False,
):
    """
    Naive PyTorch implementation of curve rasterization with the same inputs and
    outputs as the rasterize_curves function.

    This function is not optimized and is implemented as a comparison for the
    C++/CUDA implementations.
    """
    N = len(curves)
    H, W = image_size if isinstance(image_size, tuple) else (image_size, image_size)

    K = lines_per_pixel
    device = curves.device

    points_packed = curves.points_packed()
    lines_packed = curves.lines_packed()
    
    # Initialize output tensors.
    pix_to_line = torch.full(
        (H, W, K), fill_value=-1, dtype=torch.int64, device=device
    )
    zbuf = torch.full((H, W, K), fill_value=-1, dtype=torch.float32, device=device)
    bary_coords = torch.full(
        (H, W, K), fill_value=-1, dtype=torch.float32, device=device
    )
    pix_dists = torch.full(
        (H, W, K), fill_value=-1, dtype=torch.float32, device=device
    )

    # Calculate all line bounding boxes.
    line_verts = points_packed[lines_packed]
    line_idxs = torch.arange(lines_packed.shape[0], device=device)
    
    x_mins = torch.min(line_verts[:, :, 0], dim=1).values
    x_maxs = torch.max(line_verts[:, :, 0], dim=1).values
    y_mins = torch.min(line_verts[:, :, 1], dim=1).values
    y_maxs = torch.max(line_verts[:, :, 1], dim=1).values
    z_mins = torch.min(line_verts[:, :, 2], dim=1).values
    
    x_mins = x_mins - np.sqrt(blur_radius) - kEpsilon
    x_maxs = x_maxs + np.sqrt(blur_radius) + kEpsilon
    y_mins = y_mins - np.sqrt(blur_radius) - kEpsilon
    y_maxs = y_maxs + np.sqrt(blur_radius) + kEpsilon
    
    # Iterate through the horizontal lines of the image from top to bottom.
    for yi in range(H):
        print("yi:", yi)
        # Y coordinate of one end of the image. Reverse the ordering
        # of yi so that +Y is pointing up in the image.
        yfix = H - 1 - yi
        yf = pix_to_non_square_ndc(yfix, H, W)

        # Iterate through pixels on this horizontal line, left to right.
        for xi in range(W):
            # X coordinate of one end of the image. Reverse the ordering
            # of xi so that +X is pointing to the left in the image.
            xfix = W - 1 - xi
            xf = pix_to_non_square_ndc(xfix, W, H)

            invalid = (
                # Check if pixel is outside of line bbox.
                (xf < x_mins) + 
                (xf > x_maxs) + 
                (yf < y_mins) + 
                (yf > y_maxs) + 
                # Lines with at least one vertex behind the camera won't
                # render correctly and should be removed or clipped before
                # calling the rasterizer
                (z_mins < kEpsilon)
            )
            
            line_idxs_cull = line_idxs[~invalid]
            line_verts_cull = line_verts[~invalid]
            v0 = line_verts_cull[:, 0, :]
            v1 = line_verts_cull[:, 1, :]
            
            # Compute barycentric coordinates and pixel z distance.
            pxy = torch.tensor([xf, yf], dtype=torch.float32, device=device)
            t = barycentric_parameter(pxy, v0[:, :2], v1[:, :2])
            if perspective_correct:
                z0, z1 = v0[:, 2], v1[:, 2]
                t = z0 * t / (z0 * t + z1 * (1 - t))
            # Barycentric clipping
            if clip_barycentric_coords:
                t = torch.clamp(t, min=0.0)
            
            # use clipped barycentric coords to calculate the z value
            pz = t * v0[:, 2] + (1 - t) * v1[:, 2]
            
            # Calculate 2D distance from point to line.
            dist = point_line_distance(pxy, v0[:, :2], v1[:, :2])
            dist_valid_idxs = torch.arange(dist.shape[0], device=device)
            dist_valid_idxs = dist_valid_idxs[dist <= blur_radius]
            
            # sort the lines by distance to the pixel
            sort_idxs = torch.argsort(dist[dist_valid_idxs])[:K]
            sort_idxs = dist_valid_idxs[sort_idxs]

            # Save to output tensors.
            n = sort_idxs.shape[0]
            zbuf[yi, xi, :n] = pz[sort_idxs]
            pix_to_line[yi, xi, :n] = line_idxs_cull[sort_idxs]
            bary_coords[yi, xi, :n] = t[sort_idxs]
            pix_dists[yi, xi, :n] = dist[sort_idxs]

    return pix_to_line, zbuf, bary_coords, pix_dists


def barycentric_parameter(p: Tensor, v0: Tensor, v1: Tensor):

    """
    Compute the barycentric parameter of a point relative to a line.

    Args:
        p: Coordinates of a point.
        v0, v1: Coordinates of the line vertices.

    Returns
        bary: barycentric parameter t in the range [0, 1].
    """
    v1v0 = v0 - v1
    v1p = p - v1
    l2 = torch.sum(v1v0**2, dim=-1) + kEpsilon
    t = (v1p * v1v0).sum(dim=-1) / l2
    return t


def point_line_distance(p: Tensor, v0: Tensor, v1: Tensor):
    """
    Return minimum distance between line segment (v1 - v0) and point p.

    Args:
        p: Coordinates of a point.
        v0, v1: Coordinates of the end points of the line segment.

    Returns:
        non-square distance to the boundary of the triangle.

    Consider the line extending the segment - this can be parameterized as
    ``v0 + t (v1 - v0)``.

    First find the projection of point p onto the line. It falls where
    ``t = [(p - v0) . (v1 - v0)] / |v1 - v0|^2``
    where . is the dot product.

    The parameter t is clamped from [0, 1] to handle points outside the
    segment (v1 - v0).

    Once the projection of the point on the segment is known, the distance from
    p to the projection gives the minimum distance to the segment.
    """
    if p.shape[-1] != v0.shape[-1] != v1.shape[-1]:
        raise ValueError("All points must have the same number of coordinates")

    v0v1 = v1 - v0
    v0p = p - v0
    l2 = torch.sum(v0v1**2, dim=-1, keepdim=True) + kEpsilon  # |v1 - v0|^2

    t = (v0v1 * v0p).sum(dim=-1, keepdim=True) / l2
    t = torch.clamp(t, min=0.0, max=1.0)
    p_proj = v0 + t * v0v1
    delta_p = p_proj - p
    
    return torch.sum(delta_p**2, dim=-1)


def point_line_barycentric_distance(p: Tensor, v0: Tensor, v1: Tensor):
    """
    Compute the barycentric parameter of a point relative to a line, and
    return minimum distance between line segment (v1 - v0) and point p.
    """
    if p.shape[-1] != v0.shape[-1] != v1.shape[-1]:
        raise ValueError("All points must have the same number of coordinates")

    v0v1 = v1 - v0
    v0p = p - v0
    l2 = torch.sum(v0v1**2, dim=-1, keepdim=True) + kEpsilon
    t = (v0v1 * v0p).sum(dim=-1, keepdim=True) / l2

    t_clamp = torch.clamp(t, min=0.0, max=1.0)
    p_proj = v0 + t_clamp * v0v1
    dist = torch.sum((p_proj - p)**2, dim=-1)
    
    return 1 - t[..., 0], dist