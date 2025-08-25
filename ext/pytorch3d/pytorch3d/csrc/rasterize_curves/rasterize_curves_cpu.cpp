/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/extension.h>
#include <algorithm>
#include <list>
#include <queue>
#include <thread>
#include <tuple>
#include "ATen/core/TensorAccessor.h"
#include "rasterize_points/rasterization_utils.h"
#include "utils/geometry_utils.h"
#include "utils/vec2.h"
#include "utils/vec3.h"

// Get (x, y, z) values for vertex from (2, 3) tensor line.
template <typename Line>
inline auto ExtractVerts(const Line& line, const int vertex_index) {
  return std::make_tuple(
    line[vertex_index][0], line[vertex_index][1], line[vertex_index][2]);
}

// Compute min/max x/y for each line.
inline auto ComputeLineBoundingBoxes(const torch::Tensor& line_verts) {
  const int total_L = line_verts.size(0);
  auto float_opts = line_verts.options().dtype(torch::kFloat32);
  auto line_verts_a = line_verts.accessor<float, 3>();
  torch::Tensor line_bboxes = torch::full({total_L, 6}, -2.0, float_opts);

  // Loop through all the lines
  for (int l = 0; l < total_L; ++l) {
    const auto& line = line_verts_a[l];
    float x0, x1, y0, y1, z0, z1;
    std::tie(x0, y0, z0) = ExtractVerts(line, 0);
    std::tie(x1, y1, z1) = ExtractVerts(line, 1);

    const float x_min = std::min(x0, x1);
    const float y_min = std::min(y0, y1);
    const float x_max = std::max(x0, x1);
    const float y_max = std::max(y0, y1);
    const float z_min = std::min(z0, z1);
    const float z_max = std::max(z0, z1);

    line_bboxes[l][0] = x_min;
    line_bboxes[l][1] = y_min;
    line_bboxes[l][2] = x_max;
    line_bboxes[l][3] = y_max;
    line_bboxes[l][4] = z_min;
    line_bboxes[l][5] = z_max;
  }

  return line_bboxes;
}

// Check if the point (px, py) lies inside the line bounding box line_bbox.
// Return true if the point is outside.
template <typename Line>
inline bool CheckPointOutsideBoundingBox(
    const Line& line_bbox,
    float blur_radius,
    float px,
    float py) {
  // Read line bbox coordinates and expand by blur radius.
  float x_min = line_bbox[0] - blur_radius;
  float y_min = line_bbox[1] - blur_radius;
  float x_max = line_bbox[2] + blur_radius;
  float y_max = line_bbox[3] + blur_radius;

  // Lines with at least one vertex behind the camera won't render correctly
  // and should be removed or clipped before calling the rasterizer
  const bool z_invalid = line_bbox[4] < kEpsilon;

  // Check if the current point is within the line bounding box.
  return (px > x_max || px < x_min || py > y_max || py < y_min || z_invalid);
}

// Calculate minimum distance between a line segment (v1 - v0) and point p.
//
// Args:
//     p: Coordinates of a point.
//     v0, v1: Coordinates of the end points of the line segment.
//
// Returns:
//     squared distance of the point to the line.
//
// Consider the line extending the segment - this can be parameterized as:
// v0 + t (v1 - v0).
//
// First find the projection of point p onto the line. It falls where:
// t = [(p - v0) . (v1 - v0)] / |v1 - v0|^2
// where . is the dot product.
//
// The parameter t is clamped from [0, 1] to handle points outside the
// segment (v1 - v0).
//
// Once the projection of the point on the segment is known, the distance from
// p to the projection gives the minimum distance to the segment.
//
template <typename T>
inline auto PointLineDistanceWithBarycentricForward(
    const vec2<T>& p,
    const vec2<T>& v0,
    const vec2<T>& v1) {
  const vec2<T> v1v0 = v1 - v0;
  const T l2 = dot(v1v0, v1v0) + kEpsilon*kEpsilon;

  const T t = dot(v1v0, p - v0) / l2;
  const T tt = std::min(std::max(t, 0.00f), 1.00f);
  const vec2<T> p_proj = v0 + tt * v1v0;
  const T dist = dot(p - p_proj, p - p_proj);

  return std::make_tuple(dist, 1-t);
}

template <typename T>
inline T LineBarycentricParameterForward(
    const vec2<T>& p,
    const vec2<T>& v0,
    const vec2<T>& v1) {
  const vec2<T> v1v0 = v0 - v1;
  const T l2 = dot(v1v0, v1v0) + kEpsilon;
  return dot(v1v0, p - v1) / l2;
}

template <typename T>
inline auto LineBarycentricParameterBackward(
    const vec2<T>& p,
    const vec2<T>& v0,
    const vec2<T>& v1,
    const T grad_bary) {
  const vec2<T> v1v0 = v0 - v1;
  const vec2<T> v1p = p - v1;
  const T t_bot = dot(v1v0, v1v0) + kEpsilon;
  const T t_top = dot(v1v0, v1p);

  // bary = t_top / t_bot
  const T grad_t_bot = -grad_bary * t_top / (t_bot * t_bot);
  const T grad_t_top = grad_bary / t_bot;
  const vec2<T> grad_v1v0 = grad_t_bot * 2.f * v1v0 + grad_t_top * v1p;
  const vec2<T> grad_v1p = grad_t_top * v1v0;
  const vec2<T> grad_v1 = -1.f * grad_v1v0 - grad_v1p;
  const vec2<T> grad_v0 = grad_v1v0;
  const vec2<T> grad_p = grad_v1p;

  return std::make_tuple(grad_p, grad_v0, grad_v1);
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
template <typename T>
inline T LineBaryPerspectiveCorrectionForward(
    const T bary,
    const T z0,
    const T z1) {
  const T w0_top = bary * z1;
  const T w1_top = (1 - bary) * z0;
  const T denom = w0_top + w1_top + kEpsilon;

  return w0_top / denom;
}

template <typename T>
inline auto LineBaryPerspectiveCorrectionBackward(
    const T bary,
    const T z0,
    const T z1,
    const T grad_out) {
  // Recompute forward pass
  const T w0_top = bary * z1;
  const T w1_top = (1 - bary) * z0;
  const T denom = w0_top + w1_top + kEpsilon;

  // Now do backward pass
  const T grad_denom = -grad_out * w0_top / (denom * denom);
  const T grad_w0_top = grad_denom + grad_out / denom;
  const T grad_w1_top = grad_denom;
  const T grad_z0 = grad_w1_top * (1 - bary);
  const T grad_z1 = grad_w0_top * bary;
  const T grad_bary = grad_w0_top * z1 - grad_w1_top * z0;

  return std::make_tuple(grad_bary, grad_z0, grad_z1);
}

template <typename T>
inline T ClampBackward(T x, T min_val, T max_val) {
    if (x < min_val || x > max_val) {
        return static_cast<T>(0.0f);
    } else {
        return static_cast<T>(1.0f);
    }
}


namespace {
void RasterizeCurvesNaiveCpu_worker(
    const int start_yi,
    const int end_yi,
    const float blur_radius,
    const bool perspective_correct,
    const bool clip_barycentric_coords,
    const int H,
    const int W,
    const int K,
    at::TensorAccessor<float, 3>& line_verts_a,
    at::TensorAccessor<float, 2>& line_bboxes_a,
    at::TensorAccessor<float, 3>& zbuf_a,
    at::TensorAccessor<int64_t, 3>& line_idxs_a,
    at::TensorAccessor<float, 3>& pix_dists_a,
    at::TensorAccessor<float, 3>& barycentric_coords_a) {
    // Iterate through the horizontal lines of the image from top to bottom.
    for (int yi = start_yi; yi < end_yi; ++yi) {
      // Reverse the order of yi so that +Y is pointing upwards in the image.
      const int yidx = H - 1 - yi;

      // Y coordinate of the top of the pixel.
      const float yf = PixToNonSquareNdc(yidx, H, W);
      // Iterate through pixels on this horizontal line, left to right.
      for (int xi = 0; xi < W; ++xi) {
        // Reverse the order of xi so that +X is pointing to the left in the
        // image.
        const int xidx = W - 1 - xi;

        // X coordinate of the left of the pixel.
        const float xf = PixToNonSquareNdc(xidx, W, H);

        // Use a deque to hold values:
        // (z, idx, r, bary)
        // Sort the deque as needed to mimic a priority queue.
        std::deque<std::tuple<float, int, float, float>> q;

        // Loop through the lines in the mesh.
        for (int l = 0; l < line_verts_a.size(0); ++l) {
          // Get coordinates of two line vertices.
          const auto& line = line_verts_a[l];
          float x0, x1, y0, y1, z0, z1;
          std::tie(x0, y0, z0) = ExtractVerts(line, 0);
          std::tie(x1, y1, z1) = ExtractVerts(line, 1);

          const vec2<float> v0(x0, y0);
          const vec2<float> v1(x1, y1);

          // Skip if point is outside the line bounding box.
          const auto line_bbox = line_bboxes_a[l];
          const bool outside_bbox = CheckPointOutsideBoundingBox(
              line_bbox, std::sqrt(blur_radius), xf, yf);
          if (outside_bbox) {
            continue;
          }

          // Compute distance and barycentric coordinates
          const vec2<float> pxy(xf, yf);
          float dist, bary0;
          std::tie(dist, bary0) = PointLineDistanceWithBarycentricForward(pxy, v0, v1);

          if (dist > blur_radius) {
            continue; // Point is outside the line segment so ignore.
          }
            
          const float bary = !perspective_correct
              ? bary0
              : LineBaryPerspectiveCorrectionForward(bary0, z0, z1);

          const float bary_clip =
              !clip_barycentric_coords ? bary : std::clamp(bary, 0.00f, 1.00f);

          // Use barycentric coordinates to get the depth of the current pixel
          const float pz = bary_clip * z0 + (1 - bary_clip) * z1;

          if (pz < 0) {
            continue; // Point is behind the image plane so ignore.
          }

          q.emplace_back(pz, l, dist, bary_clip);
          // Sort the deque inplace based on the z distance
          // to mimic using a priority queue.
          std::sort(q.begin(), q.end());
          if (static_cast<int>(q.size()) > K) {
            // remove the last value
            q.pop_back();
          }
        }
        while (!q.empty()) {
          // Loop through and add values to the output tensors
          auto t = q.back();
          q.pop_back();
          const int i = q.size();
          zbuf_a[yi][xi][i] = std::get<0>(t);
          line_idxs_a[yi][xi][i] = std::get<1>(t);
          pix_dists_a[yi][xi][i] = std::get<2>(t);
          barycentric_coords_a[yi][xi][i] = std::get<3>(t);
        }
      }
    }
  }
} // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeCurvesNaiveCpu(
    const torch::Tensor& line_verts,
    const std::tuple<int, int> image_size,
    const float blur_radius,
    const int lines_per_pixel,
    const bool perspective_correct,
    const bool clip_barycentric_coords) {
  if (line_verts.ndimension() != 3 || line_verts.size(1) != 2 ||
      line_verts.size(2) != 3) {
    AT_ERROR("line_verts must have dimensions (num_lines, 2, 3)");
  }

  const int H = std::get<0>(image_size);
  const int W = std::get<1>(image_size);
  const int K = lines_per_pixel;

  auto long_opts = line_verts.options().dtype(torch::kInt64);
  auto float_opts = line_verts.options().dtype(torch::kFloat32);

  // Initialize output tensors.
  torch::Tensor line_idxs = torch::full({H, W, K}, -1, long_opts);
  torch::Tensor zbuf = torch::full({H, W, K}, -1, float_opts);
  torch::Tensor pix_dists = torch::full({H, W, K}, -1, float_opts);
  torch::Tensor barycentric_coords = torch::full({H, W, K}, -1, float_opts);

  auto line_verts_a = line_verts.accessor<float, 3>();
  auto line_idxs_a = line_idxs.accessor<int64_t, 3>();
  auto zbuf_a = zbuf.accessor<float, 3>();
  auto pix_dists_a = pix_dists.accessor<float, 3>();
  auto barycentric_coords_a = barycentric_coords.accessor<float, 3>();

  auto line_bboxes = ComputeLineBoundingBoxes(line_verts);
  auto line_bboxes_a = line_bboxes.accessor<float, 2>();

  const int64_t n_threads = at::get_num_threads();
  std::vector<std::thread> threads;
  threads.reserve(n_threads);
  const int chunk_size = 1 + (H - 1) / n_threads;
  int start_yi = 0;
  for (int iThread = 0; iThread < n_threads; ++iThread) {
    const int64_t end_yi = std::min(start_yi + chunk_size, H);
    threads.emplace_back(
        RasterizeCurvesNaiveCpu_worker,
        start_yi,
        end_yi,
        blur_radius,
        perspective_correct,
        clip_barycentric_coords,
        H,
        W,
        K,
        std::ref(line_verts_a),
        std::ref(line_bboxes_a),
        std::ref(zbuf_a),
        std::ref(line_idxs_a),
        std::ref(pix_dists_a),
        std::ref(barycentric_coords_a));
    start_yi += chunk_size;
  }

  for (auto&& thread : threads) {
    thread.join();
  }

  return std::make_tuple(line_idxs, zbuf, barycentric_coords, pix_dists);
}

torch::Tensor RasterizeCurvesBackwardCpu(
    const torch::Tensor& line_verts, // (L, 2, 3)
    const torch::Tensor& pix_to_line, // (H, W, K)
    const torch::Tensor& grad_zbuf, // (H, W, K)
    const torch::Tensor& grad_bary, // (H, W, K)
    const torch::Tensor& grad_dists, // (H, W, K)
    const bool perspective_correct,
    const bool clip_barycentric_coords) {
  const int L = line_verts.size(0);
  const int H = pix_to_line.size(0);
  const int W = pix_to_line.size(1);
  const int K = pix_to_line.size(2);

  torch::Tensor grad_line_verts = torch::zeros({L, 2, 3}, line_verts.options());
  auto line_verts_a = line_verts.accessor<float, 3>();
  auto pix_to_line_a = pix_to_line.accessor<int64_t, 3>();
  auto grad_dists_a = grad_dists.accessor<float, 3>();
  auto grad_zbuf_a = grad_zbuf.accessor<float, 3>();
  auto grad_bary_a = grad_bary.accessor<float, 3>();

  for (int y = 0; y < H; ++y) {
    // Reverse the order of yi so that +Y is pointing upwards in the image.
    const int yidx = H - 1 - y;

    // Y coordinate of the top of the pixel.
    const float yf = PixToNonSquareNdc(yidx, H, W);
    // Iterate through pixels on this horizontal line, left to right.
    for (int x = 0; x < W; ++x) {
      // Reverse the order of xi so that +X is pointing to the left in the
      // image.
      const int xidx = W - 1 - x;

      // X coordinate of the left of the pixel.
      const float xf = PixToNonSquareNdc(xidx, W, H);
      const vec2<float> pxy(xf, yf);

      // Iterate through the lines that hit this pixel.
      for (int k = 0; k < K; ++k) {
        // Get line index from forward pass output.
        const int l = pix_to_line_a[y][x][k];
        if (l < 0) {
          continue; // padded line.
        }

        // Get coordinates of two line vertices.
        const auto& line = line_verts_a[l];
        float x0, x1, y0, y1, z0, z1;
        std::tie(x0, y0, z0) = ExtractVerts(line, 0);
        std::tie(x1, y1, z1) = ExtractVerts(line, 1);

        const vec2<float> v0xy(x0, y0);
        const vec2<float> v1xy(x1, y1);

        // Get upstream gradients for the line.
        const float grad_dist_upstream = grad_dists_a[y][x][k];
        const float grad_zbuf_upstream = grad_zbuf_a[y][x][k];
        const float grad_bary_upstream = grad_bary_a[y][x][k];

        const float bary0 =
            LineBarycentricParameterForward(pxy, v0xy, v1xy);
        const float bary = !perspective_correct
            ? bary0
            : LineBaryPerspectiveCorrectionForward(bary0, z0, z1);
        const float bary_clip =
            !clip_barycentric_coords ? bary : std::clamp(bary, 0.00f, 1.00f);

        const auto grad_dist_l = PointLineDistanceBackward(
            pxy, v0xy, v1xy, grad_dist_upstream);
        const auto ddist_d_v0 = std::get<1>(grad_dist_l);
        const auto ddist_d_v1 = std::get<2>(grad_dist_l);

        // Upstream gradient for barycentric coords from zbuf calculation:
        // zbuf = bary * z0 + (1 - bary) * z1
        // Therefore
        // d_zbuf/d_bary_w0 = z0 - z1
        const float d_zbuf_d_baryclip = z0 - z1;

        // Total upstream barycentric gradients are the sum of
        // external upstream gradients and contribution from zbuf.
        float grad_bary0 =
            (grad_bary_upstream + grad_zbuf_upstream * d_zbuf_d_baryclip);

        if (clip_barycentric_coords) {
          grad_bary0 = ClampBackward(bary, 0.f, 1.f) * grad_bary0;
        }

        if (perspective_correct) {
          auto perspective_grads = LineBaryPerspectiveCorrectionBackward(
              bary0, z0, z1, grad_bary0);
          grad_bary0 = std::get<0>(perspective_grads);
          grad_line_verts[l][0][2] += std::get<1>(perspective_grads);
          grad_line_verts[l][1][2] += std::get<2>(perspective_grads);
        }

        auto grad_bary_l =
            LineBarycentricParameterBackward(pxy, v0xy, v1xy, grad_bary0);
        const vec2<float> dbary_d_v0 = std::get<1>(grad_bary_l);
        const vec2<float> dbary_d_v1 = std::get<2>(grad_bary_l);

        // Update output gradient buffer.
        grad_line_verts[l][0][0] += dbary_d_v0.x + ddist_d_v0.x;
        grad_line_verts[l][0][1] += dbary_d_v0.y + ddist_d_v0.y;
        grad_line_verts[l][0][2] += grad_zbuf_upstream * bary_clip;
        grad_line_verts[l][1][0] += dbary_d_v1.x + ddist_d_v1.x;
        grad_line_verts[l][1][1] += dbary_d_v1.y + ddist_d_v1.y;
        grad_line_verts[l][1][2] += grad_zbuf_upstream * (1 - bary_clip);
      }
    }
  }

  return grad_line_verts;
}

// torch::Tensor RasterizeCurvesCoarseCpu(
//     const torch::Tensor& face_verts,
//     const torch::Tensor& mesh_to_face_first_idx,
//     const torch::Tensor& num_faces_per_mesh,
//     const std::tuple<int, int> image_size,
//     const float blur_radius,
//     const int bin_size,
//     const int max_faces_per_bin) {
//   if (face_verts.ndimension() != 3 || face_verts.size(1) != 3 ||
//       face_verts.size(2) != 3) {
//     AT_ERROR("face_verts must have dimensions (num_faces, 3, 3)");
//   }
//   if (num_faces_per_mesh.ndimension() != 1) {
//     AT_ERROR("num_faces_per_mesh can only have one dimension");
//   }

//   const int N = num_faces_per_mesh.size(0); // batch size.
//   const int M = max_faces_per_bin;

//   const float H = std::get<0>(image_size);
//   const float W = std::get<1>(image_size);

//   // Integer division round up.
//   const int BH = 1 + (H - 1) / bin_size;
//   const int BW = 1 + (W - 1) / bin_size;

//   auto opts = num_faces_per_mesh.options().dtype(torch::kInt32);
//   torch::Tensor faces_per_bin = torch::zeros({N, BH, BW}, opts);
//   torch::Tensor bin_faces = torch::full({N, BH, BW, M}, -1, opts);
//   auto bin_faces_a = bin_faces.accessor<int32_t, 4>();

//   // Precompute all face bounding boxes.
//   auto face_bboxes = ComputeLineBoundingBoxes(face_verts);
//   auto face_bboxes_a = face_bboxes.accessor<float, 2>();

//   const float ndc_x_range = NonSquareNdcRange(W, H);
//   const float pixel_width_x = ndc_x_range / W;
//   const float bin_width_x = pixel_width_x * bin_size;

//   const float ndc_y_range = NonSquareNdcRange(H, W);
//   const float pixel_width_y = ndc_y_range / H;
//   const float bin_width_y = pixel_width_y * bin_size;

//   // Iterate through the curves in the batch.
//   for (int n = 0; n < N; ++n) {
//     const int face_start_idx = mesh_to_face_first_idx[n].item().to<int32_t>();
//     const int face_stop_idx =
//         (face_start_idx + num_faces_per_mesh[n].item().to<int32_t>());

//     float bin_y_min = -1.0f;
//     float bin_y_max = bin_y_min + bin_width_y;

//     // Iterate through the horizontal bins from top to bottom.
//     for (int by = 0; by < BH; ++by) {
//       float bin_x_min = -1.0f;
//       float bin_x_max = bin_x_min + bin_width_x;

//       // Iterate through bins on this horizontal line, left to right.
//       for (int bx = 0; bx < BW; ++bx) {
//         int32_t faces_hit = 0;

//         for (int32_t f = face_start_idx; f < face_stop_idx; ++f) {
//           // Get bounding box and expand by blur radius.
//           float face_x_min = face_bboxes_a[f][0] - std::sqrt(blur_radius);
//           float face_y_min = face_bboxes_a[f][1] - std::sqrt(blur_radius);
//           float face_x_max = face_bboxes_a[f][2] + std::sqrt(blur_radius);
//           float face_y_max = face_bboxes_a[f][3] + std::sqrt(blur_radius);
//           float face_z_min = face_bboxes_a[f][4];

//           // Lines with at least one vertex behind the camera won't render
//           // correctly and should be removed or clipped before calling the
//           // rasterizer
//           if (face_z_min < kEpsilon) {
//             continue;
//           }

//           // Use a half-open interval so that faces exactly on the
//           // boundary between bins will fall into exactly one bin.
//           bool x_overlap =
//               (face_x_min <= bin_x_max) && (bin_x_min < face_x_max);
//           bool y_overlap =
//               (face_y_min <= bin_y_max) && (bin_y_min < face_y_max);

//           if (x_overlap && y_overlap) {
//             // Got too many faces for this bin, so throw an error.
//             if (faces_hit >= max_faces_per_bin) {
//               AT_ERROR("Got too many faces per bin");
//             }
//             // The current point falls in the current bin, so
//             // record it.
//             bin_faces_a[n][by][bx][faces_hit] = f;
//             faces_hit++;
//           }
//         }

//         // Shift the bin to the right for the next loop iteration
//         bin_x_min = bin_x_max;
//         bin_x_max = bin_x_min + bin_width_x;
//       }
//       // Shift the bin down for the next loop iteration
//       bin_y_min = bin_y_max;
//       bin_y_max = bin_y_min + bin_width_y;
//     }
//   }
//   return bin_faces;
// }
