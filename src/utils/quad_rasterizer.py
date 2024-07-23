import torch
from pytorch3d.renderer import (
    MeshRasterizer,
    RasterizationSettings,
    TexturesVertex,
    FoVPerspectiveCameras,
)
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes, join_meshes_as_scene

from typing import Union, Optional, Tuple


def smooth(x: torch.Tensor, n: int = 3) -> torch.Tensor:
    ret = torch.cumsum(torch.concat((torch.repeat_interleave(x[:1], n - 1, dim=0), x)), 0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def xy_normal(strands: torch.Tensor, normalize: bool = True, smooth_degree: Optional[int] = None):
    d = torch.empty(strands.shape[:2] + (2, ), device=strands.device)

    if smooth_degree is not None:
        strands = smooth(strands, smooth_degree)

    d[:, :-1, :] = strands[:, 1:, [0, 1]] - strands[:, : -1, [0, 1]]
    d[:, -1, :] = d[:, -2, :]

    n = torch.cat((d[:, :, [1]], -d[:, :, [0]]), dim=2)
    
    if normalize:
        n = n / torch.linalg.norm(n, dim=2, keepdims=True)
    
    return n


def build_quads(
    strands: torch.Tensor,
    w: float = 0.0001,
    return_in_strands_shape: bool = True,
    calculate_faces: bool = True):

    n_strands, n_points = strands.shape[:2]

    n_xy = xy_normal(strands)
    n_xyz = torch.cat((n_xy, torch.zeros(n_strands, n_points, 1, device=strands.device)), axis=2)

    verts = torch.empty((n_strands, 2 * n_points, 3), device=strands.device)
    verts[:, 0::2, :] = strands + w * n_xyz
    verts[:, 1::2, :] = strands - w * n_xyz
    
    indices = torch.empty((n_strands, 2 * n_points, 1), device=strands.device, dtype=verts.dtype)
    values = torch.tensor(list(range(n_strands * n_points)), dtype=indices.dtype, device=indices.device)
    values = values.view((n_strands, n_points, 1))
    indices[:, 0::2, :] = values
    indices[:, 1::2, :] = values
    indices = indices.reshape(-1, 1)
    
    if calculate_faces:
        faces = torch.empty((2 * n_points - 2, 3), dtype=torch.long, device=strands.device)

        # Second edge in each face is boundary one, but the're inversed sinced triangles must have the same orientation

        faces[0::2, :] = \
            torch.stack((
                torch.arange(0, 2 * n_points - 3, step=2),
                torch.arange(1, 2 * n_points - 1, step=2),
                torch.arange(3, 2 * n_points + 1, step=2)
            )).T

        faces[1::2, :] = \
            torch.stack((
                torch.arange(3, 2 * n_points    , step=2),
                torch.arange(2, 2 * n_points    , step=2),
                torch.arange(0, 2 * n_points - 3, step=2)
            )).T

        full_faces_array = \
            torch.arange(0, verts.shape[1] * n_strands, verts.shape[1], device=strands.device).reshape(n_strands, 1, 1) + \
            faces.unsqueeze(0)
        
        if not return_in_strands_shape:
            full_faces_array = full_faces_array.reshape(-1, 3)

    if not return_in_strands_shape:
        verts = verts.reshape(-1, 3)

    if calculate_faces:
        return verts, full_faces_array, indices
    else:
        return verts, indices


class SoftShader(ShaderBase):

    def __init__(
        self,
        image_size: int,
        feats_dim: int = 32,
        sigma: float = 1e-3,
        gamma: float = 1e-5,
        return_alpha: bool = False,
        znear: Union[float, torch.Tensor] = 1.0,
        zfar: Union[float, torch.Tensor] = 100,
        num_head_faces: Optional[int] = None,
        **kwargs
    ):
        
        super().__init__(**kwargs)
        
        self.image_size = image_size
        self.feats_dim = feats_dim
        self.sigma = sigma
        self.gamma = gamma
        self.return_alpha = return_alpha
        self.background_color = torch.zeros(feats_dim, dtype=torch.float32)
        self.znear = znear
        self.zfar = zfar
        self.num_head_faces = num_head_faces

    def get_colors(self, fragments: Fragments, meshes: Meshes):
        pix_to_face, barycentric_coords = fragments.pix_to_face, fragments.bary_coords

        vertex_attributes = meshes.textures.verts_features_packed().unsqueeze(0)
        faces = meshes.faces_packed().unsqueeze(0)

        res = vertex_attributes[range(vertex_attributes.shape[0]), faces.flatten(start_dim=1).T]
        res = torch.transpose(res, 0, 1)
        attributes = res.reshape(faces.shape[0], faces.shape[1], 3, vertex_attributes.shape[-1])
        
        if self.return_alpha:
            alpha_mask = (pix_to_face != -1).float()[:, :, :, 0].unsqueeze(1)

        # Reshaping for torch.gather
        D = attributes.shape[-1]
        attributes = attributes.clone() # Needed for backprop
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])

        N, H, W, K, _ = barycentric_coords.shape

        # Needed for correct working of torch.gather
        mask = (pix_to_face == -1)
        pix_to_face = pix_to_face.clone() # Needed for backprop
        pix_to_face[mask] = 0

        # Building a tensor of sampled values
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)

        # Barycentric interpolation
        pixel_vals = (barycentric_coords.unsqueeze(-1) * pixel_face_vals).sum(-2)
        pixel_vals[mask] = 0
        
        if self.return_alpha:
            pixel_vals = torch.cat((pixel_vals, alpha_mask), 1)

        return pixel_vals
    
    def forward(self, fragments: Fragments, meshes: Meshes):

        N, H, W, K = fragments.pix_to_face.shape
        
        # Mask for padded pixels.
        if self.num_head_faces is None:
            mask = fragments.pix_to_face >= 0
        else:
            first_head_face = meshes.faces_list()[0].shape[0] - self.num_head_faces
            head_mask = fragments.pix_to_face >= first_head_face
            mask = head_mask
            
            for i in range(1, head_mask.shape[-1]):
                mask[..., i] += mask[..., i - 1]

            mask += fragments.pix_to_face < 0
            mask = ~mask.bool()

        colors = self.get_colors(fragments, meshes)
        pixel_colors = torch.ones((N, H, W, self.feats_dim), dtype=colors.dtype, device=colors.device)
        background_color = self.background_color.to(colors.device)

        '''
        # Fragments dists recalculation
        grid = torch.stack(torch.meshgrid((torch.linspace(1, -1, 512, device=meshes.device), ) * 2, indexing='xy'))

        s = torch.ones_like(fragments.dists[mask])
        s[fragments.dists[mask] < 0] = -1

        verts = meshes.verts_packed()

        face_to_edge = fragments.pix_to_face[mask] + 1 - 2 * (fragments.pix_to_face[mask] % 2)

        p0 = verts[face_to_edge][:, :2]
        p1 = verts[face_to_edge + 2][:, :2]

        t0 = verts[fragments.pix_to_face[mask]][:, :2]
        t1 = verts[fragments.pix_to_face[mask] + 2][:, :2]

        p_edges_norm = (p1 - p0).norm(dim=1, keepdim=True)
        t_edges_norm = (t1 - t0).norm(dim=1, keepdim=True)

        i = torch.argwhere(mask)
        x = grid[:, i[:, 1], i[:, 2]].permute(1, 0)

        d_p = ((x - p0)[:, 0] * (p1 - p0)[:, 1] - (x - p0)[:, 1] * (p1 - p0)[:, 0]).squeeze(0).abs() / (p_edges_norm.squeeze(1) + 1e-8)
        d_t = ((x - t0)[:, 0] * (t1 - t0)[:, 1] - (x - t0)[:, 1] * (t1 - t0)[:, 0]).squeeze(0).abs() / (t_edges_norm.squeeze(1) + 1e-8)

        fragments.dists[mask] = s * (torch.min(d_p, d_t) ** 2)
        fragments.pix_to_face[mask][d_t < d_p] = fragments.pix_to_face[mask][d_t < d_p] + s[d_t < d_p].long()
        '''

        # Weight for background color
        eps = 1e-10

        # Sigmoid probability map based on the distance of the pixel to the face.
        prob_map = torch.sigmoid(-fragments.dists / self.sigma) * mask

        # Weights for each face. Adjust the exponential by the max z to prevent
        # overflow. zbuf shape (N, H, W, K), find max over K.
        # TODO: there may still be some instability in the exponent calculation.

        alpha = torch.prod((1.0 - prob_map), dim=-1)

        # Reshape to be compatible with (N, H, W, K) values in fragments
        if torch.is_tensor(self.zfar):
            # pyre-fixme[16]
            self.zfar = self.zfar[:, None, None, None]
        if torch.is_tensor(self.znear):
            # pyre-fixme[16]: Item `float` of `Union[float, Tensor]` has no attribute
            #  `__getitem__`.
            self.znear = self.znear[:, None, None, None]

        z_inv = (self.zfar - fragments.zbuf) / (self.zfar - self.znear) * mask

        z_inv_max = torch.max(z_inv, dim=-1).values.unsqueeze(-1).clamp(min=eps)
        weights_num = prob_map * torch.exp((z_inv - z_inv_max) / self.gamma)

        # Also apply exp normalize trick for the background color weight.
        # Clamp to ensure delta is never 0.
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
        delta = torch.exp((eps - z_inv_max) / self.gamma).clamp(min=eps)

        # Normalize weights.
        # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
        denom = weights_num.sum(dim=-1).unsqueeze(-1) + delta

        # Sum: weights * textures + background color
        weighted_colors = (weights_num.unsqueeze(-1) * colors).sum(-2)
        weighted_background = delta * background_color
        pixel_colors = (weighted_colors + weighted_background) / denom
        
        # The cumulative product ensures that alpha will be 0.0 if at least 1
        # face fully covers the pixel as for that face, prob will be 1.0.
        # This results in a multiplication by 0.0 because of the (1.0 - prob)
        # term. Therefore 1.0 - alpha will be 1.0.
        # alpha = torch.prod((1.0 - prob_map), dim=-1)
        pixel_colors = pixel_colors * (1.0 - alpha.unsqueeze(-1))

        return pixel_colors.permute(0, 3, 1, 2), mask, colors


class QuadRasterizer(torch.nn.Module):

    def __init__(
        self,
        render_size: int,
        feats_dim: int = 3,
        quad_w: float = 0.0001,
        antialiasing_factor: int = 1,
        head_mesh: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        faces_per_pixel: int = 16,
        blur_radius: float = 1,
        sigma: float = 1e-5,
        gamma: float = 1e-5,
        znear: Union[float, torch.Tensor] = 0.1,
        zfar: Union[float, torch.Tensor] = 10.0,
        use_gpu: bool = True,
    ):
        super().__init__()

        self.render_size = render_size
        self.feats_dim = feats_dim
        self.quad_w = quad_w
        self.sigma = sigma
        self.gamma = gamma
        self.blur_radius = blur_radius
        self.antialiasing_factor = antialiasing_factor
        self.use_gpu = use_gpu

        raster_settings = RasterizationSettings(
            image_size=self.render_size * antialiasing_factor,
            blur_radius=(blur_radius/render_size)**2,
            faces_per_pixel=faces_per_pixel,
            # bin_size=0,
            # cull_backfaces=False,
            # perspective_correct=False
        )
        
        self.head_mesh: Meshes = None

        if head_mesh is not None:
            self.set_head_mesh(head_mesh)
            self.num_head_faces = self.head_mesh.faces_list()[0].shape[0]
        else:
            self.num_head_faces = None

        self.rasterizer = MeshRasterizer(raster_settings=raster_settings)
        
        self.shader = SoftShader(
            image_size=render_size,
            feats_dim=feats_dim,
            sigma=sigma,
            gamma=gamma,
            zfar=zfar,
            znear=znear,
            num_head_faces=self.num_head_faces,
        )

        if self.use_gpu:
            self.rasterizer = self.rasterizer.cuda()
            self.shader = self.shader.cuda()

    def forward(self, hair: torch.Tensor, cameras: FoVPerspectiveCameras):
        # calc tangents
        hair_proj = cameras.transform_points(hair)
        directions = hair_proj[:, 1:, :] - hair_proj[:, :-1, :]
        tangents = torch.cat([directions, directions[:, -1:, :]], dim=1)
        tangents[..., 0] = -tangents[..., 0]
        tangents = tangents / torch.norm(tangents[..., :2], dim=-1, keepdim=True)
        tangents[..., 2] = 1.0

        hair_verts, hair_faces, indices = build_quads(hair, w=self.quad_w)
        texture_tensor = tangents.reshape(-1, 3)[indices[:, 0].long()]
        hair_texture = TexturesVertex([texture_tensor])
        hair_mesh = Meshes(
            verts=[hair_verts.reshape(-1, 3)],
            faces=[hair_faces.reshape(-1, 3)],
            textures=hair_texture
        )

        if self.use_gpu:
            cameras = cameras.cuda()
            hair_mesh = hair_mesh.cuda()

        self.rasterizer.cameras = cameras

        if self.head_mesh is not None:
            meshes = join_meshes_as_scene([hair_mesh, self.head_mesh])
        else:
            meshes = hair_mesh
            
        fragments = self.rasterizer(meshes)
        images, mask, colors = self.shader(fragments, meshes)
        
        # depth
        images = torch.cat((images, fragments.zbuf[:, :, :, :1].permute(0, 3, 1, 2)), dim=1)

        if self.antialiasing_factor > 1:
            images = torch.nn.functional.avg_pool2d(
                images,
                kernel_size=self.antialiasing_factor,
                stride=self.antialiasing_factor
            )
        
        return images


    def set_head_mesh(self, head_mesh: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        
        head_verts, head_faces = head_mesh
        head_texture = torch.zeros(len(head_verts), self.feats_dim)
        if self.use_gpu:
            head_texture = head_texture.cuda()
        
        self.head_mesh = Meshes(
            verts=[head_verts],
            faces=[head_faces],
            textures=TexturesVertex([head_texture])
        )
        
        if self.use_gpu:
            self.head_mesh = self.head_mesh.cuda()