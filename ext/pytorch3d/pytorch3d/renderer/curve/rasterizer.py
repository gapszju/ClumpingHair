# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from pytorch3d.renderer.cameras import try_get_projection_transform
from pytorch3d.structures import Curves

from .rasterize_curves import rasterize_curves, rasterize_curves_python


@dataclass(frozen=True)
class CurveFragments:
    """
    A dataclass representing the outputs of a rasterizer. Can be detached from the
    computational graph in order to stop the gradients from flowing through the
    rasterizer.

    Members:
        pix_to_line:
            LongTensor of shape (image_size, image_size, lines_per_pixel) giving
            the indices of the nearest lines at each pixel, sorted in ascending
            z-order. Concretely ``pix_to_line[y, x, k] = l`` means that
            ``lines_verts[l]`` is the kth closest line (in the z-direction) to pixel
            (y, x). Pixels that are hit by fewer than lines_per_pixel are padded with
            -1.

        zbuf:
            FloatTensor of shape (image_size, image_size, lines_per_pixel) giving
            the NDC z-coordinates of the nearest lines at each pixel, sorted in
            ascending z-order. Concretely, if ``pix_to_line[y, x, k] = l`` then
            ``zbuf[y, x, k] = line_verts[l, 2]``. Pixels hit by fewer than
            lines_per_pixel are padded with -1.

        bary_coords:
            FloatTensor of shape (image_size, image_size, lines_per_pixel)
            giving the barycentric coordinates in NDC units of the nearest lines at
            each pixel, sorted in ascending z-order. Concretely, if ``pix_to_line[
            y, x, k] = l`` then ``t = barycentric[y, x, k]`` gives the
            barycentric coords for pixel (y, x) relative to the line defined by
            ``line_verts[l]``. Pixels hit by fewer than lines_per_pixel are padded
            with -1.

        dists:
            FloatTensor of shape (image_size, image_size, lines_per_pixel) giving
            the signed Euclidean distance (in NDC units) in the x/y plane of each
            point closest to the pixel. Concretely if ``pix_to_line[y, x, k] = f``
            then ``pix_dists[y, x, k]`` is the squared distance between the pixel
            (y, x) and the line given by vertices ``line_verts[l]``. Pixels hit with
            fewer than ``lines_per_pixel`` are padded with -1.
    """

    pix_to_line: torch.Tensor
    zbuf: torch.Tensor
    bary_coords: torch.Tensor
    dists: Optional[torch.Tensor]

    def detach(self) -> "Fragments":
        return CurveFragments(
            pix_to_line=self.pix_to_line,
            zbuf=self.zbuf.detach(),
            bary_coords=self.bary_coords.detach(),
            dists=self.dists.detach() if self.dists is not None else self.dists,
        )


@dataclass
class CurveRasterizationSettings:
    """
    Class to store the curve rasterization params with defaults

    Members:
        image_size: Either common height and width or (height, width), in pixels.
        blur_radius: Float distance in the range [0, 2] used to expand the line
            bounding boxes for rasterization. Setting blur radius
            results in blurred edges around the shape instead of a
            hard boundary. Set to 0 for no blur.
        lines_per_pixel: (int) Number of lines to keep track of per pixel.
            We return the nearest lines_per_pixel lines along the z-axis.
        bin_size: Size of bins to use for coarse-to-fine rasterization. Setting
            bin_size=0 uses naive rasterization; setting bin_size=None attempts
            to set it heuristically based on the shape of the input. This should
            not affect the output, but can affect the speed of the forward pass.
        max_lines_per_bin: Only applicable when using coarse-to-fine
            rasterization (bin_size != 0); this is the maximum number of lines
            allowed within each bin. This should not affect the output values,
            but can affect the memory usage in the forward pass.
            Setting max_lines_per_bin=None attempts to set with a heuristic.
        perspective_correct: Whether to apply perspective correction when
            computing barycentric coordinates for pixels.
            None (default) means make correction if the camera uses perspective.
        clip_barycentric_coords: Whether, after any perspective correction
            is applied but before the depth is calculated (e.g. for
            z clipping), to "correct" a location outside the line (i.e. with
            a negative barycentric coordinate) to a position on the line. 
            None (default) means clip if blur_radius > 0, which is a condition
            under which such outside-line-points are likely.
    """

    image_size: Union[int, Tuple[int, int]] = 256
    blur_radius: float = 0.0
    lines_per_pixel: int = 1
    bin_size: Optional[int] = None
    max_lines_per_bin: Optional[int] = None
    perspective_correct: Optional[bool] = None
    clip_barycentric_coords: Optional[bool] = None


class CurveRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogeneous
    Curves.
    """

    def __init__(self, cameras=None, raster_settings=None) -> None:
        """
        Args:
            cameras: A cameras object which has a  `transform_points` method
                which returns the transformed points after applying the
                world-to-view and view-to-ndc transformations.
            raster_settings: the parameters for rasterization. This should be a
                named tuple.

        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        if raster_settings is None:
            raster_settings = CurveRasterizationSettings()

        self.cameras = cameras
        self.raster_settings = raster_settings

    def to(self, device):
        # Manually move to device cameras as it is not a subclass of nn.Module
        if self.cameras is not None:
            self.cameras = self.cameras.to(device)
        return self

    def transform(self, curves_world, **kwargs) -> Curves:
        """
        Args:
            curves_world: a Curves object representing a batch of curves with
                vertex coordinates in world space.

        Returns:
            curves_proj: a curves object with the vertex positions projected
            in NDC space

        NOTE: keeping this as a separate function for readability but it could
        be moved into forward.
        """
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of CurveRasterizer"
            raise ValueError(msg)

        n_cameras = len(cameras)
        if n_cameras != 1:
            raise ValueError("Only one camera is supported for now.")

        points_world = curves_world.points_packed()

        # NOTE: Retaining view space z coordinate for now.
        # TODO: Revisit whether or not to transform z coordinate to [-1, 1] or
        # [0, 1] range.
        eps = kwargs.get("eps", None)
        points_view = cameras.get_world_to_view_transform(**kwargs).transform_points(
            points_world, eps=eps
        )
        to_ndc_transform = cameras.get_ndc_camera_transform(**kwargs)
        projection_transform = try_get_projection_transform(cameras, kwargs)
        if projection_transform is not None:
            projection_transform = projection_transform.compose(to_ndc_transform)
            points_ndc = projection_transform.transform_points(points_view, eps=eps)
        else:
            # Call transform_points instead of explicitly composing transforms to handle
            # the case, where camera class does not have a projection matrix form.
            points_proj = cameras.transform_points(points_world, eps=eps)
            points_ndc = to_ndc_transform.transform_points(points_proj, eps=eps)

        points_ndc[..., 2] = points_view[..., 2]
        curves_ndc = curves_world.copy().update_packed(new_points_packed=points_ndc)
        return curves_ndc

    def forward(self, curves_world, **kwargs) -> CurveFragments:
        """
        Args:
            curves_world: a Curves object representing a batch of curves with
                          coordinates in world space.
        Returns:
            Fragments: Rasterization outputs as a named tuple.
        """
        curves_proj = self.transform(curves_world, **kwargs)
        raster_settings = kwargs.get("raster_settings", self.raster_settings)

        # By default, turn on clip_barycentric_coords if blur_radius > 0.
        # When blur_radius > 0, a line can be matched to a pixel that is outside the
        # line, resulting in negative barycentric coordinates.
        clip_barycentric_coords = raster_settings.clip_barycentric_coords
        if clip_barycentric_coords is None:
            clip_barycentric_coords = raster_settings.blur_radius > 0.0

        # If not specified, infer perspective_correct from the camera
        cameras = kwargs.get("cameras", self.cameras)
        if raster_settings.perspective_correct is not None:
            perspective_correct = raster_settings.perspective_correct
        else:
            perspective_correct = cameras.is_perspective()

        pix_to_line, zbuf, bary_coords, dists = rasterize_curves(
            curves_proj,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            lines_per_pixel=raster_settings.lines_per_pixel,
            bin_size=raster_settings.bin_size,
            max_lines_per_bin=raster_settings.max_lines_per_bin,
            clip_barycentric_coords=clip_barycentric_coords,
            perspective_correct=perspective_correct,
        )

        return CurveFragments(
            pix_to_line=pix_to_line,
            zbuf=zbuf,
            bary_coords=bary_coords,
            dists=dists,
        )
