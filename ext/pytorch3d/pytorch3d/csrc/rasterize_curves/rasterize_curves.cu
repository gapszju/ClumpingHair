/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <float.h>
#include <math.h>
#include <thrust/tuple.h>
#include <cstdio>
#include <tuple>
#include "rasterize_coarse/bitmask.cuh"
#include "rasterize_points/rasterization_utils.cuh"
#include "utils/float_math.cuh"
#include "utils/geometry_utils.cuh"

namespace {
// A structure for holding details about a pixel.
struct Pixel {
  float z;
  int64_t idx; // idx of line
  float dist; // abs distance of pixel to line
  float bary;
};

__device__ bool operator<(const Pixel& a, const Pixel& b) {
  return a.z < b.z || (a.z == b.z && a.idx < b.idx);
}

// Get the xyz coordinates of the two vertices for the line given by the
// index line_idx into line_verts.
__device__ thrust::tuple<float3, float3> GetSingleLineVerts(
    const float* line_verts,
    int64_t line_idx) {
  const float x0 = line_verts[line_idx * 6 + 0];
  const float y0 = line_verts[line_idx * 6 + 1];
  const float z0 = line_verts[line_idx * 6 + 2];
  const float x1 = line_verts[line_idx * 6 + 3];
  const float y1 = line_verts[line_idx * 6 + 4];
  const float z1 = line_verts[line_idx * 6 + 5];

  const float3 v0xyz = make_float3(x0, y0, z0);
  const float3 v1xyz = make_float3(x1, y1, z1);

  return thrust::make_tuple(v0xyz, v1xyz);
}

// Get the min/max x/y/z values for the line given by vertices v0, v1, v2.
__device__ thrust::tuple<float2, float2, float2>
GetLineBoundingBox(float3 v0, float3 v1) {
  const float xmin = fminf(v0.x, v1.x);
  const float ymin = fminf(v0.y, v1.y);
  const float zmin = fminf(v0.z, v1.z);
  const float xmax = fmaxf(v0.x, v1.x);
  const float ymax = fmaxf(v0.y, v1.y);
  const float zmax = fmaxf(v0.z, v1.z);

  return thrust::make_tuple(
      make_float2(xmin, xmax),
      make_float2(ymin, ymax),
      make_float2(zmin, zmax));
}

// Check if the point (px, py) lies outside the line bounding box line_bbox.
// Return true if the point is outside.
__device__ bool CheckPointOutsideBoundingBox(
    float3 v0,
    float3 v1,
    float blur_radius,
    float2 pxy) {
  const auto bbox = GetLineBoundingBox(v0, v1);
  const float2 xlims = thrust::get<0>(bbox);
  const float2 ylims = thrust::get<1>(bbox);
  const float2 zlims = thrust::get<2>(bbox);

  const float x_min = xlims.x - blur_radius;
  const float y_min = ylims.x - blur_radius;
  const float x_max = xlims.y + blur_radius;
  const float y_max = ylims.y + blur_radius;

  // Lines with at least one vertex behind the camera won't render correctly
  // and should be removed or clipped before calling the rasterizer
  const bool z_invalid = zlims.x < kEpsilon;

  // Check if the current point is oustside the triangle bounding box.
  return (
      pxy.x > x_max || pxy.x < x_min || pxy.y > y_max || pxy.y < y_min ||
      z_invalid);
}

// Calculate the barycentric parameter of a point pxy with respect to the
// line with vertices v0, v1.
__device__ float LineBarycentricParameterForward(
    const float2& p,
    const float2& v0,
    const float2& v1) {
  const float2 v1v0 = v0 - v1;
  const float l2 = dot(v1v0, v1v0) + kEpsilon;
  return dot(v1v0, p - v1) / l2;
}

// Forward pass for applying perspective correction to barycentric coordinates.
//
// Args:
//     bary: Screen-space barycentric coordinates for a point
//     z0, z1: Camera-space z-coordinates of the line vertices
//
// Returns
//     World-space barycentric parameter
//
__device__ inline float LineBaryPerspectiveCorrectionForward(
    const float bary,
    const float z0,
    const float z1) {
  const float w0_top = bary * z1;
  const float w1_top = (1 - bary) * z0;
  const float denom = fmaxf(w0_top + w1_top, kEpsilon);

  return w0_top / denom;
}

__device__ inline float ClampBackward(float x, float min_val, float max_val) {
    if (x < min_val || x > max_val) {
        return 0.0f;
    } else {
        return 1.0f;
    }
}

__device__ inline auto LineBaryPerspectiveCorrectionBackward(
    const float bary,
    const float z0,
    const float z1,
    const float grad_out) {
  // Recompute forward pass
  const float w0_top = bary * z1;
  const float w1_top = (1 - bary) * z0;
  const float denom = fmaxf(w0_top + w1_top, kEpsilon);

  // Now do backward pass
  const float grad_denom = -grad_out * w0_top / (denom * denom);
  const float grad_w0_top = grad_denom + grad_out / denom;
  const float grad_w1_top = grad_denom;
  const float grad_z0 = grad_w1_top * (1 - bary);
  const float grad_z1 = grad_w0_top * bary;
  const float grad_bary = grad_w0_top * z1 - grad_w1_top * z0;

  return thrust::make_tuple(grad_bary, grad_z0, grad_z1);
}

__device__ inline auto LineBarycentricParameterBackward(
    const float2& p,
    const float2& v0,
    const float2& v1,
    const float grad_bary) {
  const float2 v1v0 = v0 - v1;
  const float2 v1p = p - v1;
  const float t_bot = dot(v1v0, v1v0) + kEpsilon;
  const float t_top = dot(v1v0, v1p);

  // bary = t_top / t_bot
  const float grad_t_bot = -grad_bary * t_top / (t_bot * t_bot);
  const float grad_t_top = grad_bary / t_bot;
  const float2 grad_v1v0 = grad_t_bot * 2.f * v1v0 + grad_t_top * v1p;
  const float2 grad_v1p = grad_t_top * v1v0;
  const float2 grad_v1 = -1.f * grad_v1v0 - grad_v1p;
  const float2 grad_v0 = grad_v1v0;
  const float2 grad_p = grad_v1p;

  return thrust::make_tuple(grad_p, grad_v0, grad_v1);
}

// This function checks if a pixel given by xy location pxy lies within the
// line with index line_idx in line_verts. One of the inputs is a list (q)
// which contains Pixel structs with the indices of the lines which intersect
// with this pixel sorted by closest z distance. If the point pxy lies in the
// line, the list (q) is updated and re-orderered in place. In addition
// the auxiliary variables q_size, q_max_z and q_max_idx are also modified.
// This code is shared between RasterizeCurvesNaiveCudaKernel and
// RasterizeCurvesFineCudaKernel.
template <typename LineQ>
__device__ void UpdateKClosestQueue(
    const float* line_verts, // (L, 2, 3)
    const int64_t line_idx,
    int& q_size,
    float& q_max_z,
    int& q_max_idx,
    LineQ& q,
    const float blur_radius,
    const float2 pxy, // Coordinates of the pixel
    const int K,
    const bool perspective_correct,
    const bool clip_barycentric_coords) {
  const auto v01 = GetSingleLineVerts(line_verts, line_idx);
  const float3 v0 = thrust::get<0>(v01);
  const float3 v1 = thrust::get<1>(v01);

  // Only need xy for barycentric coordinates and distance calculations.
  const float2 v0xy = make_float2(v0.x, v0.y);
  const float2 v1xy = make_float2(v1.x, v1.y);

  // Perform checks and skip if:
  // 1. the line is behind the camera
  // 2. the pixel is outside the line bbox
  const float zmax = fmaxf(v0.z, v1.z);
  const bool outside_bbox = CheckPointOutsideBoundingBox(
      v0, v1, sqrt(blur_radius), pxy); // use sqrt of blur for bbox
  if (zmax < 0 || outside_bbox) {
    return;
  }

  // Calculate barycentric coords and euclidean dist to triangle.
  const float bary0 = LineBarycentricParameterForward(pxy, v0xy, v1xy);
  const float bary = !perspective_correct
      ? bary0
      : LineBaryPerspectiveCorrectionForward(bary0, v0.z, v1.z);
  const float bary_clip =
      !clip_barycentric_coords ? bary : __saturatef(bary);

  // Use barycentric coordinates to get the depth of the current pixel
  const float pz = bary_clip * v0.z + (1 - bary_clip) * v1.z;

  if (pz < 0) {
    return; // Line is behind the image plane.
  }

  // Get abs squared distance
  const float dist = PointLineDistanceForward(pxy, v0xy, v1xy);

  // Check if pixel is outside blur region
  if (dist >= blur_radius) {
    return;
  }

  // Compare the distance of the pixel to t1 with the distance to t2.
  // If dist_t1 < dist_t2, overwrite the values for t2 in the top K lines.
  if (q_size < K) {
    // Just insert it.
    q[q_size] = {pz, line_idx, dist, bary_clip};
    if (pz > q_max_z) {
      q_max_z = pz;
      q_max_idx = q_size;
    }
    q_size++;
  } else if (pz < q_max_z) {
    // Overwrite the old max, and find the new max.
    q[q_max_idx] = {pz, line_idx, dist, bary_clip};
    q_max_z = pz;
    for (int i = 0; i < K; i++) {
      if (q[i].z > q_max_z) {
        q_max_z = q[i].z;
        q_max_idx = i;
      }
    }
  }

}

} // namespace

// ****************************************************************************
// *                          NAIVE RASTERIZATION                             *
// ****************************************************************************
__global__ void RasterizeCurvesNaiveCudaKernel(
    const float* line_verts,
    const int64_t num_lines,
    const float blur_radius,
    const bool perspective_correct,
    const bool clip_barycentric_coords,
    const int H,
    const int W,
    const int K,
    at::PackedTensorAccessor32<int64_t, 3> pix_to_line,
    at::PackedTensorAccessor32<float, 3> zbuf,
    at::PackedTensorAccessor32<float, 3> pix_dists,
    at::PackedTensorAccessor32<float, 3> bary) {
  // Simple version: One thread per output pixel
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= H || col >= W) {
    return;
  }

  // Reverse ordering of X and Y axes
  const int yi = H - 1 - row;
  const int xi = W - 1 - col;

  // screen coordinates to ndc coordinates of pixel.
  const float xf = PixToNonSquareNdc(xi, W, H);
  const float yf = PixToNonSquareNdc(yi, H, W);
  const float2 pxy = make_float2(xf, yf);

  // For keeping track of the K closest points we want a data structure
  // that (1) gives O(1) access to the closest point for easy comparisons,
  // and (2) allows insertion of new elements. In the CPU version we use
  // std::priority_queue; then (2) is O(log K). We can't use STL
  // containers in CUDA; we could roll our own max heap in an array, but
  // that would likely have a lot of warp divergence so we do something
  // simpler instead: keep the elements in an unsorted array, but keep
  // track of the max value and the index of the max value. Then (1) is
  // still O(1) time, while (2) is O(K) with a clean loop. Since K <= 8
  // this should be fast enough for our purposes.
  Pixel q[kMaxPointsPerPixel];
  int q_size = 0;
  float q_max_z = -1000;
  int q_max_idx = -1;
  
  // Loop through the lines in the curves.
  for (size_t l = 0; l < num_lines; ++l) {
    // Check if the pixel pxy is inside the line bounding box and if it is,
    // update q, q_size, q_max_z and q_max_idx in place.
    UpdateKClosestQueue(
        line_verts,
        l,
        q_size,
        q_max_z,
        q_max_idx,
        q,
        blur_radius,
        pxy,
        K,
        perspective_correct,
        clip_barycentric_coords);
  }

  BubbleSort(q, q_size);
  
  for (int k = 0; k < q_size; ++k) {
    pix_to_line[row][col][k]  = q[k].idx;
    zbuf[row][col][k]         = q[k].z;
    pix_dists[row][col][k]    = q[k].dist;
    bary[row][col][k]         = q[k].bary;
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
RasterizeCurvesNaiveCuda(
    const at::Tensor& line_verts,
    const std::tuple<int, int> image_size,
    const float blur_radius,
    const int lines_per_pixel,
    const bool perspective_correct,
    const bool clip_barycentric_coords) {
  TORCH_CHECK(
      line_verts.ndimension() == 3 && line_verts.size(1) == 2 &&
          line_verts.size(2) == 3,
      "line_verts must have dimensions (num_lines, 2, 3)");
  if (lines_per_pixel > kMaxPointsPerPixel) {
    std::stringstream ss;
    ss << "Must have points_per_pixel <= " << kMaxPointsPerPixel;
    AT_ERROR(ss.str());
  }

  // Check inputs are on the same device
  at::TensorArg line_verts_t{line_verts, "line_verts", 1};
  at::CheckedFrom c = "RasterizeCurvesNaiveCuda";
  at::checkAllSameGPU(c, {line_verts_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(line_verts.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int H = std::get<0>(image_size);
  const int W = std::get<1>(image_size);
  const int K = lines_per_pixel;

  auto long_opts = line_verts.options().dtype(at::kLong);
  auto float_opts = line_verts.options().dtype(at::kFloat);

  // Initialize output tensors.
  at::Tensor pix_to_line = at::full({H, W, K}, -1, long_opts);
  at::Tensor zbuf = at::full({H, W, K}, -1, float_opts);
  at::Tensor pix_dists = at::full({H, W, K}, -1, float_opts);
  at::Tensor bary = at::full({H, W, K}, -1, float_opts);

  if (pix_to_line.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(pix_to_line, zbuf, bary, pix_dists);
  }

  const dim3 threads(8, 8);
  const dim3 blocks(1 + (W - 1) / threads.x, 1 + (H - 1) / threads.y);

  RasterizeCurvesNaiveCudaKernel<<<blocks, threads, 0, stream>>>(
      line_verts.contiguous().data_ptr<float>(),
      line_verts.size(0),
      blur_radius,
      perspective_correct,
      clip_barycentric_coords,
      H,
      W,
      K,
      pix_to_line.packed_accessor32<int64_t, 3>(),
      zbuf.packed_accessor32<float, 3>(),
      pix_dists.packed_accessor32<float, 3>(),
      bary.packed_accessor32<float, 3>());

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(pix_to_line, zbuf, bary, pix_dists);
}

// ****************************************************************************
// *                            BACKWARD PASS                                 *
// ****************************************************************************
__global__ void RasterizeCurvesBackwardCudaKernel(
    const float* line_verts, // (L, 2, 3)
    const int64_t* pix_to_line, // (H, W, K)
    const bool perspective_correct,
    const bool clip_barycentric_coords,
    const int H,
    const int W,
    const int K,
    const float* grad_zbuf, // (H, W, K)
    const float* grad_bary, // (H, W, K, 3)
    const float* grad_dists, // (H, W, K)
    float* grad_line_verts) { // (L, 2, 3)

  // Parallelize over each pixel in images of
  // size H * W, for each image in the batch of size N.
  const int num_threads = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int pix_idx = tid; pix_idx < H * W; pix_idx += num_threads) {
    // Reverse ordering of X and Y axes.
    const int yi = H - 1 - pix_idx / W;
    const int xi = W - 1 - pix_idx % W;

    const float xf = PixToNonSquareNdc(xi, W, H);
    const float yf = PixToNonSquareNdc(yi, H, W);
    const float2 pxy = make_float2(xf, yf);

    // Loop over all the lines for this pixel.
    for (int k = 0; k < K; k++) {
      // Index into (H, W, K) grad tensors
      // pixel index + top k index
      int i = pix_idx * K + k;

      const int l = pix_to_line[i];
      if (l < 0) {
        continue; // padded line.
      }
      // Get xyz coordinates of the three line vertices.
      const auto v01 = GetSingleLineVerts(line_verts, l);
      const float3 v0 = thrust::get<0>(v01);
      const float3 v1 = thrust::get<1>(v01);

      // Only neex xy for barycentric coordinate and distance calculations.
      const float2 v0xy = make_float2(v0.x, v0.y);
      const float2 v1xy = make_float2(v1.x, v1.y);

      // Get upstream gradients for the line.
      const float grad_dist_upstream = grad_dists[i];
      const float grad_zbuf_upstream = grad_zbuf[i];
      const float grad_bary_upstream = grad_bary[i];
      const float bary0 = LineBarycentricParameterForward(pxy, v0xy, v1xy);
      const float bary = !perspective_correct
          ? bary0
          : LineBaryPerspectiveCorrectionForward(bary0, v0.z, v1.z);

      const float bary_clip =
          !clip_barycentric_coords ? bary : __saturatef(bary);

      auto grad_dist_f = PointLineDistanceBackward(
          pxy, v0xy, v1xy, grad_dist_upstream);
      const float2 ddist_d_v0 = thrust::get<1>(grad_dist_f);
      const float2 ddist_d_v1 = thrust::get<2>(grad_dist_f);

      // Upstream gradient for barycentric coords from zbuf calculation:
      // zbuf = bary * z0 + (1 - bary) * z1
      // Therefore
      // d_zbuf/d_bary_w0 = z0 - z1
      const float d_zbuf_d_baryclip = v0.z - v1.z;

      // Total upstream barycentric gradients are the sum of
      // external upstream gradients and contribution from zbuf.
      float grad_bary0 =
            (grad_bary_upstream + grad_zbuf_upstream * d_zbuf_d_baryclip);

      if (clip_barycentric_coords) {
        grad_bary0 = ClampBackward(bary, 0.f, 1.f) * grad_bary0;
      }

      float dz0_persp = 0.f, dz1_persp = 0.f;
      if (perspective_correct) {
        auto perspective_grads = LineBaryPerspectiveCorrectionBackward(
            bary0, v0.z, v1.z, grad_bary0);
        grad_bary0 = thrust::get<0>(perspective_grads);
        dz0_persp = thrust::get<1>(perspective_grads);
        dz1_persp = thrust::get<2>(perspective_grads);
      }

      auto grad_bary_l =
          LineBarycentricParameterBackward(pxy, v0xy, v1xy, grad_bary0);
      const float2 dbary_d_v0 = thrust::get<1>(grad_bary_l);
      const float2 dbary_d_v1 = thrust::get<2>(grad_bary_l);

      atomicAdd(grad_line_verts+l*6 + 0, dbary_d_v0.x + ddist_d_v0.x);
      atomicAdd(grad_line_verts+l*6 + 1, dbary_d_v0.y + ddist_d_v0.y);
      atomicAdd(grad_line_verts+l*6 + 2, grad_zbuf_upstream * bary_clip + dz0_persp);
      atomicAdd(grad_line_verts+l*6 + 3, dbary_d_v1.x + ddist_d_v1.x);
      atomicAdd(grad_line_verts+l*6 + 4, dbary_d_v1.y + ddist_d_v1.y);
      atomicAdd(grad_line_verts+l*6 + 5, grad_zbuf_upstream * (1 - bary_clip) + dz1_persp);
    }
  }
}

at::Tensor RasterizeCurvesBackwardCuda(
    const at::Tensor& line_verts, // (L, 2, 3)
    const at::Tensor& pix_to_line, // (H, W, K)
    const at::Tensor& grad_zbuf, // (H, W, K)
    const at::Tensor& grad_bary, // (H, W, K)
    const at::Tensor& grad_dists, // (H, W, K)
    const bool perspective_correct,
    const bool clip_barycentric_coords) {
  // Check inputs are on the same device
  at::TensorArg line_verts_t{line_verts, "line_verts", 1},
      pix_to_line_t{pix_to_line, "pix_to_line", 2},
      grad_zbuf_t{grad_zbuf, "grad_zbuf", 3},
      grad_bary_t{grad_bary, "grad_bary", 4},
      grad_dists_t{grad_dists, "grad_dists", 5};
  at::CheckedFrom c = "RasterizeCurvesBackwardCuda";
  at::checkAllSameGPU(
      c, {line_verts_t, pix_to_line_t, grad_zbuf_t, grad_bary_t, grad_dists_t});
  at::checkAllSameType(
      c, {line_verts_t, grad_zbuf_t, grad_bary_t, grad_dists_t});

  // This is nondeterministic because atomicAdd
  at::globalContext().alertNotDeterministic("RasterizeCurvesBackwardCuda");

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(line_verts.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int L = line_verts.size(0);
  const int H = pix_to_line.size(0);
  const int W = pix_to_line.size(1);
  const int K = pix_to_line.size(2);

  at::Tensor grad_line_verts = at::zeros({L, 2, 3}, line_verts.options());

  if (grad_line_verts.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_line_verts;
  }

  const size_t blocks = 1024;
  const size_t threads = 64;

  RasterizeCurvesBackwardCudaKernel<<<blocks, threads, 0, stream>>>(
      line_verts.contiguous().data_ptr<float>(),
      pix_to_line.contiguous().data_ptr<int64_t>(),
      perspective_correct,
      clip_barycentric_coords,
      H,
      W,
      K,
      grad_zbuf.contiguous().data_ptr<float>(),
      grad_bary.contiguous().data_ptr<float>(),
      grad_dists.contiguous().data_ptr<float>(),
      grad_line_verts.data_ptr<float>());

  AT_CUDA_CHECK(cudaGetLastError());
  return grad_line_verts;
}

// ****************************************************************************
// *                            FINE RASTERIZATION                            *
// ****************************************************************************
__global__ void RasterizeCurvesFineCudaKernel(
    const float* line_verts, // (L, 2, 3)
    const at::PackedTensorAccessor32<int64_t, 3> bin_lines, // (BH, BW, M)
    const float blur_radius,
    const int bin_size,
    const bool perspective_correct,
    const bool clip_barycentric_coords,
    const int M,
    const int H,
    const int W,
    const int K,
    at::PackedTensorAccessor32<int64_t, 3> pix_to_line, // (H, W, K)
    at::PackedTensorAccessor32<float, 3> zbuf, // (H, W, K)
    at::PackedTensorAccessor32<float, 3> pix_dists, // (H, W, K)
    at::PackedTensorAccessor32<float, 3> bary // (H, W, K)
) {
  // get each pixel coordinate
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= H || col >= W) {
    return;
  }

  // bin coordinate
  const int by = row / bin_size;
  const int bx = col / bin_size;

  // Reverse ordering of X and Y axes
  const int yi = H - 1 - row;
  const int xi = W - 1 - col;

  // screen coordinates to ndc coordinates of pixel.
  const float xf = PixToNonSquareNdc(xi, W, H);
  const float yf = PixToNonSquareNdc(yi, H, W);
  const float2 pxy = make_float2(xf, yf);

  // This part looks like the naive rasterization kernel, except we use
  // bin_lines to only look at a subset of lines already known to fall
  // in this bin. TODO abstract out this logic into some data structure
  // that is shared by both kernels?
  Pixel q[kMaxPointsPerPixel];
  int q_size = 0;
  float q_max_z = -1000;
  int q_max_idx = -1;

  for (int m = 0; m < M; m++) {
    const int64_t l = bin_lines[by][bx][m];
    if (l < 0) {
      continue; // bin_lines uses -1 as a sentinal value.
    }
    // Check if the pixel pxy is inside the line bounding box and if it is,
    // update q, q_size, q_max_z and q_max_idx in place.
    UpdateKClosestQueue(
        line_verts,
        l,
        q_size,
        q_max_z,
        q_max_idx,
        q,
        blur_radius,
        pxy,
        K,
        perspective_correct,
        clip_barycentric_coords);
  }

  // Now we've looked at all the lines for this bin, so we can write
  // output for the current pixel.
  BubbleSort(q, q_size);
  
  for (int k = 0; k < q_size; ++k) {
    pix_to_line[row][col][k]  = q[k].idx;
    zbuf[row][col][k]         = q[k].z;
    pix_dists[row][col][k]    = q[k].dist;
    bary[row][col][k]         = q[k].bary;
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
RasterizeCurvesFineCuda(
    const at::Tensor& line_verts,
    const at::Tensor& bin_lines,
    const std::tuple<int, int> image_size,
    const float blur_radius,
    const int bin_size,
    const int lines_per_pixel,
    const bool perspective_correct,
    const bool clip_barycentric_coords) {
  TORCH_CHECK(
      line_verts.ndimension() == 3 && line_verts.size(1) == 2 &&
          line_verts.size(2) == 3,
      "line_verts must have dimensions (num_lines, 2, 3)");
  TORCH_CHECK(bin_lines.ndimension() == 3, "bin_lines must have 3 dimensions");
  if (lines_per_pixel > kMaxPointsPerPixel) {
    std::stringstream ss;
    ss << "Must have points_per_pixel <= " << kMaxPointsPerPixel;
    AT_ERROR(ss.str());
  }

  // Check inputs are on the same device
  at::TensorArg line_verts_t{line_verts, "line_verts", 1},
      bin_lines_t{bin_lines, "bin_lines", 2};
  at::CheckedFrom c = "RasterizeCurvesFineCuda";
  at::checkAllSameGPU(
      c, {line_verts_t, bin_lines_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(line_verts.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // bin_lines shape (BH, BW, M)
  const int BH = bin_lines.size(0);
  const int BW = bin_lines.size(1);
  const int M = bin_lines.size(2);
  const int K = lines_per_pixel;

  const int H = std::get<0>(image_size);
  const int W = std::get<1>(image_size);

  auto long_opts = bin_lines.options().dtype(at::kLong);
  auto float_opts = line_verts.options().dtype(at::kFloat);

  at::Tensor pix_to_line = at::full({H, W, K}, -1, long_opts);
  at::Tensor zbuf = at::full({H, W, K}, -1, float_opts);
  at::Tensor pix_dists = at::full({H, W, K}, -1, float_opts);
  at::Tensor bary = at::full({H, W, K}, -1, float_opts);

  if (pix_to_line.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(pix_to_line, zbuf, bary, pix_dists);
  }

  const dim3 threads(8, 8);
  const dim3 blocks(1 + (W - 1) / threads.x, 1 + (H - 1) / threads.y);

    RasterizeCurvesFineCudaKernel<<<blocks, threads, 0, stream>>>(
      line_verts.contiguous().data_ptr<float>(),
      bin_lines.packed_accessor32<int64_t, 3>(),
      blur_radius,
      bin_size,
      perspective_correct,
      clip_barycentric_coords,
      M,
      H,
      W,
      K,
      pix_to_line.packed_accessor32<int64_t, 3>(),
      zbuf.packed_accessor32<float, 3>(),
      pix_dists.packed_accessor32<float, 3>(),
      bary.packed_accessor32<float, 3>());

    return std::make_tuple(pix_to_line, zbuf, bary, pix_dists);
}

// ****************************************************************************
// *                            COARSE RASTERIZATION                            *
// ****************************************************************************

__global__ void RasterizeCurvesCoarseCudaKernel(
    const float* line_verts, // (L, 2, 3)
    const int64_t num_lines,
    const int H,
    const int W,
    const float blur_radius,
    const int bin_size,
    const int max_elem_per_bin,
    at::PackedTensorAccessor32<int, 2> elems_per_bin,
    at::PackedTensorAccessor32<int64_t, 3> bin_lines) {
  extern __shared__ char sbuf[];
  const int M = max_elem_per_bin;
  const int block_size = blockDim.x;
  const int64_t line_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Integer divide round up
  const int num_bins_x = 1 + (W - 1) / bin_size;
  const int num_bins_y = 1 + (H - 1) / bin_size;

  // This is a boolean array of shape (num_bins_y, num_bins_x, block_size)
  // stored in shared memory that will track whether each elem in the block
  // falls into each bin of the image.
  BitMask binmask((unsigned int*)sbuf, num_bins_y, num_bins_x, block_size);
  binmask.block_clear();

  if (line_idx < num_lines) {

    // Calculate the bounding box of the current line
    const auto v01 = GetSingleLineVerts(line_verts, line_idx);
    const float3 v0 = thrust::get<0>(v01);
    const float3 v1 = thrust::get<1>(v01);

    const auto bbox = GetLineBoundingBox(v0, v1);
    const float2 xlims = thrust::get<0>(bbox);
    const float2 ylims = thrust::get<1>(bbox);
    const float2 zlims = thrust::get<2>(bbox);

    const float x_min = xlims.x - sqrtf(blur_radius);
    const float y_min = ylims.x - sqrtf(blur_radius);
    const float x_max = xlims.y + sqrtf(blur_radius);
    const float y_max = ylims.y + sqrtf(blur_radius);

    // Lines with at least one vertex behind the camera won't render correctly
    // and should be removed or clipped before calling the rasterizer
    const bool z_valid = zlims.x > kEpsilon;

    if (z_valid) {
      // Reverse ordering of x and y axis so that +X is right and +Y is down.
      const int xi_min = W - 1 - NonSquareNdcToPix(x_max, W, H);
      const int yi_min = H - 1 - NonSquareNdcToPix(y_max, H, W);
      const int xi_max = W - 1 - NonSquareNdcToPix(x_min, W, H);
      const int yi_max = H - 1 - NonSquareNdcToPix(y_min, H, W);
      
      const int xb_min = xi_min / bin_size;
      const int yb_min = yi_min / bin_size;
      const int xb_max = xi_max / bin_size;
      const int yb_max = yi_max / bin_size;

      for (int by = max(yb_min, 0); by <= min(yb_max, num_bins_y-1); ++by) {
        for (int bx = max(xb_min, 0); bx <= min(xb_max, num_bins_x-1); ++bx) {
            binmask.set(by, bx, threadIdx.x);
        }
      }
    }
  }
  __syncthreads();

  // Now we have processed every elem in the current block. We need to
  // count the number of elems in each bin so we can write the indices
  // out to global memory. We have each thread handle a different bin.
  for (int byx = threadIdx.x; byx < num_bins_y * num_bins_x; byx += blockDim.x) {
    const int by = byx / num_bins_x;
    const int bx = byx % num_bins_x;
    const int count = binmask.count(by, bx);

    if (count == 0) {
      continue;
    }

    // This atomically increments the (global) number of elems found
    // in the current bin, and gets the previous value of the counter;
    // this effectively allocates space in the bin_faces array for the
    // elems in the current block that fall into this bin.
    const int start = atomicAdd(&elems_per_bin[by][bx], count);
    if (start + count > M) {
      // The number of elems in this bin is so big that they won't fit.
      // We print a warning using CUDA's printf. This may be invisible
      // to notebook users, but apparent to others. It would be nice to
      // also have a Python-friendly warning, but it is not obvious
      // how to do this without slowing down the normal case.
      const char* warning =
          "Bin size was too small in the coarse rasterization phase. "
          "This caused an overflow, meaning output may be incomplete. "
          "To solve, "
          "try increasing max_faces_per_bin / max_points_per_bin, "
          "decreasing bin_size, "
          "or setting bin_size to 0 to use the naive rasterization.";
      printf(warning);
      continue;
    }

    // Now loop over the binmask and write the active bits for this bin
    // out to bin_faces.
    int next_idx = start;
    for (int e = 0; e < block_size; ++e) {
      if (binmask.get(by, bx, e)) {
        bin_lines[by][bx][next_idx] = blockIdx.x * block_size + e;
        next_idx++;
      }
    }
  }
  __syncthreads();
}

at::Tensor RasterizeCurvesCoarseCuda(
    const at::Tensor& line_verts,
    const std::tuple<int, int> image_size,
    const float blur_radius,
    const int bin_size,
    const int max_lines_per_bin) {
  TORCH_CHECK(
      line_verts.ndimension() == 3 && line_verts.size(1) == 2 &&
          line_verts.size(2) == 3,
      "line_verts must have dimensions (num_lines, 2, 3)");

  const int H = std::get<0>(image_size);
  const int W = std::get<1>(image_size);
  const int M = max_lines_per_bin;

  // Check inputs are on the same device
  at::TensorArg line_verts_t{line_verts, "line_verts", 1};
  at::CheckedFrom c = "RasterizeCurvesCoarseCuda";
  at::checkAllSameGPU(c, {line_verts_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(line_verts.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Integer divide round up
  const int num_bins_y = 1 + (H - 1) / bin_size;
  const int num_bins_x = 1 + (W - 1) / bin_size;

  if (num_bins_y >= kMaxItemsPerBin || num_bins_x >= kMaxItemsPerBin) {
    std::stringstream ss;
    ss << "In RasterizeCoarseCuda got num_bins_y: " << num_bins_y
       << ", num_bins_x: " << num_bins_x << ", "
       << "; that's too many!";
    AT_ERROR(ss.str());
  }
  auto int_opts = line_verts.options().dtype(at::kInt);
  auto long_opts = line_verts.options().dtype(at::kLong);
  at::Tensor elems_per_bin = at::zeros({num_bins_y, num_bins_x}, int_opts);
  at::Tensor bin_lines = at::full({num_bins_y, num_bins_x, M}, -1, long_opts);

  if (bin_lines.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return bin_lines;
  }

  const size_t threads = 512;
  const size_t blocks = 1 + (line_verts.size(0) - 1) / threads;
  const size_t shared_size = num_bins_y * num_bins_x * threads / 8;

    RasterizeCurvesCoarseCudaKernel<<<blocks, threads, shared_size, stream>>>(
      line_verts.contiguous().data_ptr<float>(),
      line_verts.size(0),
      H,
      W,
      blur_radius,
      bin_size,
      M,
      elems_per_bin.packed_accessor32<int, 2>(),
      bin_lines.packed_accessor32<int64_t, 3>());

    AT_CUDA_CHECK(cudaGetLastError());

  return bin_lines;
}