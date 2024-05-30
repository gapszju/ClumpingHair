import os
import torch
import json
import numpy as np

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
)
from pytorch3d.structures import Meshes, Curves
from pytorch3d.renderer import (
    MeshRasterizer,
    RasterizationSettings,
    FoVPerspectiveCameras,
)
from pytorch3d.renderer.curve import (
    CurveFragments,
    rasterize_curves,
)


def load_cameras(camera_path, device="cpu"):
    if camera_path.endswith(".json"):
        with open(camera_path) as f:
            cam_params = json.load(f)
        R, T = look_at_view_transform(
            eye=(cam_params["position"],),
            at=(cam_params["look_at"],),
            up=(cam_params["up"],),
            device=device,
        )
        cameras = FoVPerspectiveCameras(
            R=R,
            T=T,
            fov=cam_params["fov"],
            device=device,
        )
        return cameras
    
    if camera_path.endswith(".npy"):
        cam_params = np.load(camera_path, allow_pickle=True).item()
        scale = cam_params["scale"][0]
        ortho_ratio = cam_params["ortho_ratio"]
        center = torch.tensor(cam_params["center"])
        
        R = torch.tensor(cam_params["R"])
        T = - (R @ center).reshape(3, 1)
        
        K = torch.eye(3)
        K[2, 2] = -K[2, 2]
        K[0, 0] = -K[0, 0]
        K *= scale / ortho_ratio / 512
        
        R = K @ R
        T = K @ T
        T[2, 0] += 1
        
        cameras = FoVOrthographicCameras(
            R=R.T.unsqueeze(0),
            T=T.T,
            K=torch.eye(4).unsqueeze(0),
            device=device,
        )
        return cameras
    

def transform_points_to_ndc(cameras: FoVPerspectiveCameras, points_world):
    # NOTE: Retaining view space z coordinate for now.
    points_view = cameras.get_world_to_view_transform().transform_points(points_world)
    to_ndc_transform = cameras.get_ndc_camera_transform()
    points_proj = cameras.transform_points(points_world)
    points_ndc = to_ndc_transform.transform_points(points_proj)

    points_ndc[..., 2] = points_view[..., 2]
    return points_ndc


def transform_to_ndc(cameras: FoVPerspectiveCameras, curves_world: Curves) -> Curves:
    """
    Args:
        cameras: a batch of cameras, batch_size = 1.
        curves_world: a Curves object representing a batch of curves with
            vertex coordinates in world space.

    Returns:
        curves_proj: a curves object with the vertex positions projected
        in NDC space

    NOTE: keeping this as a separate function for readability but it could
    be moved into forward.
    """

    points_world = curves_world.points_packed()

    points_ndc = transform_points_to_ndc(cameras, points_world)
    curves_ndc = curves_world.copy().update_packed(points_ndc)
    return curves_ndc


def render_meshes_zbuf(meshes, cameras, image_size):
    rasterizer = MeshRasterizer(
        cameras, RasterizationSettings(image_size=image_size, blur_radius=0, faces_per_pixel=1)
    ).to(meshes.device)
    fragments = rasterizer(meshes)
    zbuf = fragments.zbuf.reshape(image_size[0], image_size[1])
    zbuf[zbuf == -1] = np.inf
    return zbuf


def filter_fragments_zbuf(fragments: CurveFragments, zbuf: torch.Tensor) -> CurveFragments:
    mask = (fragments.zbuf > zbuf[..., None])
    fragments.dists[mask] = -1
    fragments.zbuf[mask] = -1
    fragments.bary_coords[mask] = -1
    
    return fragments


def curve_softmax_rgb_blend(
    colors: torch.Tensor,
    fragments: CurveFragments,
    znear: float = 1.0,
    zfar: float = 100,
    sigma: float = 1e-6,
    gamma: float = 1e-6,
    background_color: torch.Tensor = 0,
) -> torch.Tensor:
    """
    RGB and alpha channel blending to return an RGBA image based on the method
    proposed in [1]
      - **RGB** - blend the colors based on the 2D distance based probability map and
        relative z distances.
      - **A** - blend based on the 2D distance based probability map.

    Args:
        colors: (H, W, K, C) RGB color for each of the top K lines per pixel.
        fragments: namedtuple with outputs of rasterization. We use properties
            - pix_to_line: LongTensor of shape (H, W, K) specifying the indices
              of the lines (in the packed representation) which
              overlap each pixel in the image.
            - dists: FloatTensor of shape (H, W, K) specifying
              the 2D euclidean distance from the center of each pixel
              to each of the top K overlapping lines.
            - zbuf: FloatTensor of shape (H, W, K) specifying
              the interpolated depth from each pixel to to each of the
              top K overlapping lines.
        znear: float, near clipping plane in the z direction
        zfar: float, far clipping plane in the z direction
        sigma: float, parameter which controls the width of the sigmoid
          function used to calculate the 2D distance based probability.
          Sigma controls the sharpness of the edges of the shape.
        gamma: float, parameter which controls the scaling of the
          exponential function used to control the opacity of the color.
        background_color: (C) element list/tuple/torch.Tensor specifying
          the RGB values for the background color.

    Returns:
        RGBA pixel_colors: (H, W, 4)

    [0] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
    Image-based 3D Reasoning'
    """
    # Weight for background color
    eps = 1e-10

    # Mask for padded pixels.
    mask = fragments.dists >= 0

    # Sigmoid probability map based on the distance of the pixel to the face.
    prob_map = 2 * torch.sigmoid(-fragments.dists / sigma) * mask
    # alpha = 1 - torch.prod((1.0 - prob_map), dim=-1)
    alpha = torch.tanh(prob_map.mean(dim=-1) * 16)

    # Weights for each face. Adjust the exponential by the max z to prevent
    # overflow. zbuf shape (N, H, W, K), find max over K.
    z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
    z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
    weights_num = prob_map * torch.exp((z_inv - z_inv_max) / gamma)

    # Also apply exp normalize trick for the background color weight.
    # Clamp to ensure delta is never 0.
    delta = torch.exp((eps - z_inv_max) / gamma).clamp(min=eps)

    # Normalize weights.
    # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
    denom = weights_num.sum(dim=-1)[..., None] + delta

    # Sum: weights * textures + background color
    weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)
    weighted_background = delta * background_color
    pixel_colors = (weighted_colors + weighted_background) / denom
    pixel_colors = torch.cat([pixel_colors, alpha[..., None]], dim=-1)

    return pixel_colors


def calc_orientation_color(curves_proj: Curves, fragments, clump_scale=None):
    strands = curves_proj.points_packed().reshape(-1, 32, 3)
    lines_packed = curves_proj.lines_packed()
    direction = strands[:, 1:, :] - strands[:, :-1, :]
    tangent = torch.cat([direction, direction[:, -1:, :]], dim=1)
    tangent[..., 0] *= -1
    tangent[..., 2] = 0
    tangent = tangent / torch.norm(tangent, dim=-1, keepdim=True).clamp(min=1e-10)
    tangent = tangent * 0.5 + 0.5
    if clump_scale is not None:
        tangent[..., 2] = clump_scale[..., 0]

    H, W, K = fragments.pix_to_line.shape
    pix_to_verts = lines_packed[fragments.pix_to_line]  # (H, W, K, 2)
    # use gather to accelerate backward
    idx = pix_to_verts.reshape(-1, 1).repeat(1, 3)
    colors = tangent.reshape(-1, 3).gather(0, idx).reshape(H, W, K, 2, 3)
    # colors = tangent.reshape(-1, 3)[pix_to_verts]
    colors = (
            fragments.bary_coords[..., None] * colors[:, :, :, 0, :]
            + (1 - fragments.bary_coords[..., None]) * colors[:, :, :, 1, :]
        )
    colors[fragments.dists < 0] = 0
    return colors


def render_feature_map(config, curves, img_size, mesh_zbuf=None, clump_scale=None):
    fragments = CurveFragments(
            *rasterize_curves(
                curves,
                image_size=img_size,
                blur_radius=(config["blur_radius"] / img_size[0]) ** 2,
                lines_per_pixel=config["lines_per_pixel"],
                bin_size=config["bin_size"],
                perspective_correct=False,
                clip_barycentric_coords=True,
            )
        )
    if mesh_zbuf is not None:
        fragments = filter_fragments_zbuf(fragments, mesh_zbuf)
        
    orien_colors = calc_orientation_color(curves, fragments, clump_scale)
    image = curve_softmax_rgb_blend(
            torch.cat([orien_colors, fragments.zbuf[..., None]], dim=-1),
            fragments,
            sigma=config["sigma"],
            gamma=config["gamma"],
        )
    image_silh = image[..., -1]
    image_feat = image[..., :-1]
    image_orien = image_feat[..., :3]
    
    image_depth = image_feat[..., 3]
    zbuf_min = fragments.zbuf[fragments.zbuf > 0].min().item()
    zbuf_range = fragments.zbuf.max().item() - zbuf_min
    image_depth = (image_depth - zbuf_min) / zbuf_range
    image_depth = 1 - image_depth
    image_depth[image_depth > 1] = 0
    
    return image_silh, image_depth, image_orien