/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include "utils/pytorch3d_cutils.h"

// ****************************************************************************
// *                            FORWARD PASS                                 *
// ****************************************************************************

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeCurvesNaiveCpu(
    const torch::Tensor& line_verts,
    const std::tuple<int, int> image_size,
    const float blur_radius,
    const int lines_per_pixel,
    const bool perspective_correct,
    const bool clip_barycentric_coords);

#ifdef WITH_CUDA
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
RasterizeCurvesNaiveCuda(
    const at::Tensor& line_verts,
    const std::tuple<int, int> image_size,
    const float blur_radius,
    const int num_closest,
    const bool perspective_correct,
    const bool clip_barycentric_coords);
#endif
// Forward pass for rasterizing a batch of Curves.
//
// Args:
//    line_verts: Tensor of shape (L, 2, 3) giving (packed) vertex positions for
//                lines in all the Curves in the batch. Concretely,
//                line_verts[l, i] = [x, y, z] gives the coordinates for the
//                ith vertex of the fth line. These vertices are expected to be
//                in NDC coordinates in the range [-1, 1].
//    image_size: Tuple (H, W) giving the size in pixels of the output
//                image to be rasterized.
//    blur_radius: float distance in NDC coordinates uses to expand the line
//                 bounding boxes for the rasterization. Set to 0.0 if no blur
//                 is required.
//    lines_per_pixel: the number of closeset lines to rasterize per pixel.
//    perspective_correct: Whether to apply perspective correction when
//                         computing barycentric coordinates. If this is True,
//                         then this function returns world-space barycentric
//                         coordinates for each pixel; if this is False then
//                         this function instead returns screen-space
//                         barycentric coordinates for each pixel.
//    clip_barycentric_coords: Whether, after any perspective correction
//          is applied but before the depth is calculated (e.g. for
//          z clipping), to "correct" a location outside the line (i.e. with
//          a negative barycentric coordinate) to a position on the edge of the
//          line.
//
// Returns:
//    A 4 element tuple of:
//    pix_to_line: int64 tensor of shape (H, W, K) giving the line index of
//                 each of the closest lines to the pixel in the rasterized
//                 image, or -1 for pixels that are not covered by any line.
//    zbuf: float32 Tensor of shape (H, W, K) giving the depth of each of
//          the closest lines for each pixel.
//    barycentric_coords: float tensor of shape (H, W, K) giving
//                        barycentric coordinates of the pixel with respect to
//                        each of the closest lines along the z axis, padded
//                        with -1 for pixels hit by fewer than
//                        lines_per_pixel lines.
//    dists: float tensor of shape (H, W, K) giving the euclidean distance
//           in the (NDC) x/y plane between each pixel and its K closest
//           lines along the z axis padded  with -1 for pixels hit by fewer than
//           lines_per_pixel lines.
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeCurvesNaive(
    const torch::Tensor& line_verts,
    const std::tuple<int, int> image_size,
    const float blur_radius,
    const int lines_per_pixel,
    const bool perspective_correct,
    const bool clip_barycentric_coords) {
  if (line_verts.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(line_verts);
    return RasterizeCurvesNaiveCuda(
        line_verts,
        image_size,
        blur_radius,
        lines_per_pixel,
        perspective_correct,
        clip_barycentric_coords);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    return RasterizeCurvesNaiveCpu(
        line_verts,
        image_size,
        blur_radius,
        lines_per_pixel,
        perspective_correct,
        clip_barycentric_coords);
  }
}

// ****************************************************************************
// *                            BACKWARD PASS                                 *
// ****************************************************************************

torch::Tensor RasterizeCurvesBackwardCpu(
    const torch::Tensor& line_verts,
    const torch::Tensor& pix_to_line,
    const torch::Tensor& grad_zbuf,
    const torch::Tensor& grad_bary,
    const torch::Tensor& grad_dists,
    const bool perspective_correct,
    const bool clip_barycentric_coords);

#ifdef WITH_CUDA
torch::Tensor RasterizeCurvesBackwardCuda(
    const torch::Tensor& line_verts,
    const torch::Tensor& pix_to_line,
    const torch::Tensor& grad_zbuf,
    const torch::Tensor& grad_bary,
    const torch::Tensor& grad_dists,
    const bool perspective_correct,
    const bool clip_barycentric_coords);
#endif

// Args:
//    line_verts: float32 Tensor of shape (L, 2, 3) (from forward pass) giving
//                (packed) vertex positions for lines in all the Curves in
//                 the batch.
//    pix_to_line: int64 tensor of shape (H, W, K) giving the line index of
//                 each of the closest lines to the pixel in the rasterized
//                 image, or -1 for pixels that are not covered by any line.
//    grad_zbuf: Tensor of shape (H, W, K) giving upstream gradients
//               d(loss)/d(zbuf) of the zbuf tensor from the forward pass.
//    grad_bary: Tensor of shape (H, W, K) giving upstream gradients
//               d(loss)/d(bary) of the barycentric_coords tensor returned by
//               the forward pass.
//    grad_dists: Tensor of shape (H, W, K) giving upstream gradients
//                d(loss)/d(dists) of the dists tensor from the forward pass.
//    perspective_correct: Whether to apply perspective correction when
//                         computing barycentric coordinates. If this is True,
//                         then this function returns world-space barycentric
//                         coordinates for each pixel; if this is False then
//                         this function instead returns screen-space
//                         barycentric coordinates for each pixel.
//    clip_barycentric_coords: Whether, after any perspective correction
//          is applied but before the depth is calculated (e.g. for
//          z clipping), to "correct" a location outside the line (i.e. with
//          a negative barycentric coordinate) to a position on the edge of the
//          line.
//
// Returns:
//    grad_line_verts: float32 Tensor of shape (L, 2, 3) giving downstream
//                     gradients for the line vertices.
torch::Tensor RasterizeCurvesBackward(
    const torch::Tensor& line_verts,
    const torch::Tensor& pix_to_line,
    const torch::Tensor& grad_zbuf,
    const torch::Tensor& grad_bary,
    const torch::Tensor& grad_dists,
    const bool perspective_correct,
    const bool clip_barycentric_coords) {
  if (line_verts.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(line_verts);
    CHECK_CUDA(pix_to_line);
    CHECK_CUDA(grad_zbuf);
    CHECK_CUDA(grad_bary);
    CHECK_CUDA(grad_dists);
    return RasterizeCurvesBackwardCuda(
        line_verts,
        pix_to_line,
        grad_zbuf,
        grad_bary,
        grad_dists,
        perspective_correct,
        clip_barycentric_coords);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    return RasterizeCurvesBackwardCpu(
        line_verts,
        pix_to_line,
        grad_zbuf,
        grad_bary,
        grad_dists,
        perspective_correct,
        clip_barycentric_coords);
  }
}

// ****************************************************************************
// *                          COARSE RASTERIZATION                            *
// ****************************************************************************

// torch::Tensor RasterizeCurvesCoarseCpu(
//     const torch::Tensor& line_verts,
//     const std::tuple<int, int> image_size,
//     const float blur_radius,
//     const int bin_size,
//     const int max_lines_per_bin);

torch::Tensor RasterizeCurvesCoarseCuda(
    const torch::Tensor& line_verts,
    const std::tuple<int, int> image_size,
    const float blur_radius,
    const int bin_size,
    const int max_lines_per_bin);

// Args:
//    line_verts: Tensor of shape (L, 2, 3) giving (packed) vertex positions for
//                lines in all the Curves in the batch. Concretely,
//                line_verts[l, i] = [x, y, z] gives the coordinates for the
//                ith vertex of the fth line. These vertices are expected to be
//                in NDC coordinates in the range [-1, 1].
//    image_size: Tuple (H, W) giving the size in pixels of the output
//                image to be rasterized.
//    blur_radius: float distance in NDC coordinates uses to expand the line
//                 bounding boxes for the rasterization. Set to 0.0 if no blur
//                 is required.
//    bin_size: Size of each bin within the image (in pixels)
//    max_lines_per_bin: Maximum number of lines to count in each bin.
//
// Returns:
//   bin_line_idxs: Tensor of shape (num_bins, num_bins, K) giving the
//                  indices of lines that fall into each bin.

torch::Tensor RasterizeCurvesCoarse(
    const torch::Tensor& line_verts,
    const std::tuple<int, int> image_size,
    const float blur_radius,
    const int bin_size,
    const int max_lines_per_bin) {
  if (line_verts.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(line_verts);
    return RasterizeCurvesCoarseCuda(
        line_verts,
        image_size,
        blur_radius,
        bin_size,
        max_lines_per_bin);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    AT_ERROR("NOT IMPLEMENTED");
  }
}

// ****************************************************************************
// *                            FINE RASTERIZATION                            *
// ****************************************************************************

#ifdef WITH_CUDA
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeCurvesFineCuda(
    const torch::Tensor& line_verts,
    const torch::Tensor& bin_lines,
    const std::tuple<int, int> image_size,
    const float blur_radius,
    const int bin_size,
    const int lines_per_pixel,
    const bool perspective_correct,
    const bool clip_barycentric_coords);
#endif
// Args:
//    line_verts: Tensor of shape (L, 2, 3) giving (packed) vertex positions for
//                lines in all the Curves in the batch. Concretely,
//                line_verts[l, i] = [x, y, z] gives the coordinates for the
//                ith vertex of the fth line. These vertices are expected to be
//                in NDC coordinates in the range [-1, 1].
//    bin_lines: int32 Tensor of shape (B, B, M) giving the indices of lines
//               that fall into each bin (output from coarse rasterization).
//    image_size: Tuple (H, W) giving the size in pixels of the output
//                image to be rasterized.
//    blur_radius: float distance in NDC coordinates uses to expand the line
//                 bounding boxes for the rasterization. Set to 0.0 if no blur
//                 is required.
//    bin_size: Size of each bin within the image (in pixels)
//    lines_per_pixel: the number of closeset lines to rasterize per pixel.
//    perspective_correct: Whether to apply perspective correction when
//                         computing barycentric coordinates. If this is True,
//                         then this function returns world-space barycentric
//                         coordinates for each pixel; if this is False then
//                         this function instead returns screen-space
//                         barycentric coordinates for each pixel.
//    clip_barycentric_coords: Whether, after any perspective correction
//          is applied but before the depth is calculated (e.g. for
//          z clipping), to "correct" a location outside the line (i.e. with
//          a negative barycentric coordinate) to a position on the edge of the
//          line.
//
// Returns (same as rasterize_Curves):
//    A 4 element tuple of:
//    pix_to_line: int64 tensor of shape (H, W, K) giving the line index of
//                 each of the closest lines to the pixel in the rasterized
//                 image, or -1 for pixels that are not covered by any line.
//    zbuf: float32 Tensor of shape (H, W, K) giving the depth of each of
//          the closest lines for each pixel.
//    barycentric_coords: float tensor of shape (H, W, K) giving
//                        barycentric coordinates of the pixel with respect to
//                        each of the closest lines along the z axis, padded
//                        with -1 for pixels hit by fewer than
//                        lines_per_pixel lines.
//    dists: float tensor of shape (H, W, K) giving the euclidean distance
//           in the (NDC) x/y plane between each pixel and its K closest
//           lines along the z axis padded  with -1 for pixels hit by fewer than
//           lines_per_pixel lines.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeCurvesFine(
    const torch::Tensor& line_verts,
    const torch::Tensor& bin_lines,
    const std::tuple<int, int> image_size,
    const float blur_radius,
    const int bin_size,
    const int lines_per_pixel,
    const bool perspective_correct,
    const bool clip_barycentric_coords) {
  if (line_verts.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(line_verts);
    CHECK_CUDA(bin_lines);
    return RasterizeCurvesFineCuda(
        line_verts,
        bin_lines,
        image_size,
        blur_radius,
        bin_size,
        lines_per_pixel,
        perspective_correct,
        clip_barycentric_coords);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    AT_ERROR("NOT IMPLEMENTED");
  }
}

// ****************************************************************************
// *                         MAIN ENTRY POINT                                 *
// ****************************************************************************

// This is the main entry point for the forward pass of the curve rasterizer;
// it uses either naive or coarse-to-fine rasterization based on bin_size.
//
// Args:
//    line_verts: Tensor of shape (L, 2, 3) giving (packed) vertex positions for
//                lines in all the Curves in the batch. Concretely,
//                line_verts[l, i] = [x, y, z] gives the coordinates for the
//                ith vertex of the fth line. These vertices are expected to be
//                in NDC coordinates in the range [-1, 1].
//    image_size: Tuple (H, W) giving the size in pixels of the output
//                image to be rasterized.
//    blur_radius: float distance in NDC coordinates uses to expand the line
//                 bounding boxes for the rasterization. Set to 0.0 if no blur
//                 is required.
//    lines_per_pixel: the number of closeset lines to rasterize per pixel.
//    bin_size: Bin size (in pixels) for coarse-to-fine rasterization. Setting
//              bin_size=0 uses naive rasterization instead.
//    max_lines_per_bin: The maximum number of lines allowed to fall into each
//                      bin when using coarse-to-fine rasterization.
//    perspective_correct: Whether to apply perspective correction when
//                         computing barycentric coordinates. If this is True,
//                         then this function returns world-space barycentric
//                         coordinates for each pixel; if this is False then
//                         this function instead returns screen-space
//                         barycentric coordinates for each pixel.
//    clip_barycentric_coords: Whether, after any perspective correction
//          is applied but before the depth is calculated (e.g. for
//          z clipping), to "correct" a location outside the line (i.e. with
//          a negative barycentric coordinate) to a position on the edge of the
//          line.
//
// Returns:
//    A 4 element tuple of:
//    pix_to_line: int64 tensor of shape (H, W, K) giving the line index of
//                 each of the closest lines to the pixel in the rasterized
//                 image, or -1 for pixels that are not covered by any line.
//    zbuf: float32 Tensor of shape (H, W, K) giving the depth of each of
//          the closest lines for each pixel.
//    barycentric_coords: float tensor of shape (H, W, K) giving
//                        barycentric coordinates of the pixel with respect to
//                        each of the closest lines along the z axis, padded
//                        with -1 for pixels hit by fewer than
//                        lines_per_pixel lines.
//    dists: float tensor of shape (H, W, K) giving the euclidean distance
//           in the (NDC) x/y plane between each pixel and its K closest
//           lines along the z axis padded  with -1 for pixels hit by fewer than
//           lines_per_pixel lines.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeCurves(
    const torch::Tensor& line_verts,
    const std::tuple<int, int> image_size,
    const float blur_radius,
    const int lines_per_pixel,
    const int bin_size,
    const int max_lines_per_bin,
    const bool perspective_correct,
    const bool clip_barycentric_coords) {
  if (bin_size > 0 && max_lines_per_bin > 0) {
    // Use coarse-to-fine rasterization
    at::Tensor bin_lines = RasterizeCurvesCoarse(
        line_verts,
        image_size,
        blur_radius,
        bin_size,
        max_lines_per_bin);
    return RasterizeCurvesFine(
        line_verts,
        bin_lines,
        image_size,
        blur_radius,
        bin_size,
        lines_per_pixel,
        perspective_correct,
        clip_barycentric_coords);
  } else {
    // Use the naive per-pixel implementation
    return RasterizeCurvesNaive(
        line_verts,
        image_size,
        blur_radius,
        lines_per_pixel,
        perspective_correct,
        clip_barycentric_coords);
  }
}