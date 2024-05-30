import os
import sys
import torch
import time
import struct
import bpy
import bmesh
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from mathutils import Matrix
from mathutils.bvhtree import BVHTree
from mathutils.interpolate import poly_3d_calc

from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import TexturesUV

from pytorch3d.ops import knn_points

from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

device = torch.device("cuda")


def read_hair_cy(filepath: str, scale: float=1.0, coord_transform: bool=True) -> bpy.types.Object:
    """Read a cyHair format hair file.

    We convert the coords from OpenGL's axis convention to Blender's.
    """
    with open(filepath, "rb") as fp:
        header = fp.read(4)
        assert header == b"HAIR"
        num_strands = struct.unpack("I", fp.read(4))[0]
        num_points = struct.unpack("I", fp.read(4))[0]
        bit_flags = int.from_bytes(fp.read(4), "little")
        fp.seek(40-16, os.SEEK_CUR)  # skips unused fields
        info = fp.read(88).decode("ascii").rstrip('\x00')

        seg_counts = np.frombuffer(fp.read(2 * num_strands), dtype=np.uint16)
        positions = np.frombuffer(fp.read(12 * num_points), dtype=np.float32).reshape(-1, 3)
        if coord_transform:
            # convert opengl coordinate system to blender coordinate system
            positions = positions @ np.array(Matrix.Rotation(np.pi/2, 3, "X")).astype(np.float32).T
        thicknesses = np.frombuffer(fp.read(4 * num_points), dtype=np.float32) if bit_flags & 0b00100 else\
                      np.empty(0)
        
    indices = np.cumsum(seg_counts+1)[:-1]
    position_list = np.split(positions, indices)
    thickness_list = np.split(thicknesses, indices) if bit_flags & 0b00100 else None
    
    return position_list, thickness_list


def write_hair_cy(filepath, num_verts, positions, thicknesses=None, info=""):
    """Saves the given strands to Cem Yuksel's hair format.

    The file format specification can be found in:
    http://www.cemyuksel.com/research/hairmodels/

    Args:
        filepath: path of the saved hair file.
        num_verts: list of point counts of each spline.
        positions: list of 3D point coordinates (tuples of 3 floats).
        thicknesses: list of point thicknesses.
        info: information string to be saved in the file.
    """
    if not filepath.endswith(".hair"):
        raise NotImplementedError(f"{filepath} is not a .hair file!")
    
    if isinstance(num_verts, torch.Tensor):
        num_verts = num_verts.detach().cpu().numpy()
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()
    if isinstance(thicknesses, torch.Tensor):
        thicknesses = thicknesses.detach().cpu().numpy()
    
    num_strands = len(num_verts)
    num_points = len(positions)

    print(f"Writing {num_strands} strands, {num_points} total points.")

    with open(filepath, "wb") as fp:
        fp.write(b"HAIR")
        fp.write(struct.pack("I", num_strands))
        fp.write(struct.pack("I", num_points))
        bit_flags = "11000" if thicknesses is None else "11100"
        fp.write(struct.pack("i", int(bit_flags[::-1], 2)))
        fp.write(b"\0" * (40-16)) # skips unused fields
        
        # file information
        fp.write(bytes((info + "\0"*88)[:88], "ascii"))

        fp.write(np.array(num_verts).astype(np.uint16)-1)
        fp.write(np.array(positions).astype(np.float32))
        if not thicknesses is None:
            fp.write(np.array(thicknesses).astype(np.float32))


def read_hair_data(hair_path, coord_transform: bool=True):
    # helpful to parse binary data: https://docs.python.org/3/library/struct.html
    # this decodes the following file format:
    # NumofStrand(int)
    # NumOfVertices(int) Vertex1(3 float) Vertex2(3 float) Vertex3(3 float) ...
    # NumOfVertices(int) Vertex1(3 float) Vertex2(3 float) Vertex3(3 float) ...
    # ...
    # ...
    position_list = []
    with open(hair_path, "rb") as fp:
        num_of_strands = struct.unpack('i', fp.read(4))[0]
        for i in range(num_of_strands):
            num_of_vertices = struct.unpack('i', fp.read(4))[0]
            strand = np.frombuffer(fp.read(12 * num_of_vertices), dtype=np.float32).reshape(-1, 3)
            if coord_transform:
                # convert opengl coordinate system to blender coordinate system
                strand = strand @ np.array(Matrix.Rotation(np.pi/2, 3, "X")).astype(np.float32).T
            position_list.append(strand)
            
    return position_list


def bmesh_from_pydata(verts, faces, verts_uvs=None, faces_uvs=None):
    bm = bmesh.new()

    # add verts
    for v_co in verts:
        bm.verts.new(v_co)
    bm.verts.ensure_lookup_table()

    # add faces
    for f_idx in faces:
        bm.faces.new([bm.verts[i] for i in f_idx])
    bm.faces.ensure_lookup_table()

    if verts_uvs is not None and faces_uvs is not None:
        # add uv layer
        uv_layer = bm.loops.layers.uv.new()
        for face, uvs in zip(bm.faces, faces_uvs):
            for loop, uv in zip(face.loops, uvs):
                loop[uv_layer].uv = verts_uvs[uv]
                
    bm.normal_update()

    return bm


def bmesh_from_pytorch3d(mesh: Meshes):
    verts = mesh.verts_list()[0].detach().cpu().numpy()
    faces = mesh.faces_list()[0].detach().cpu().numpy()

    if mesh.textures is None:
        return bmesh_from_pydata(verts, faces)
    else:
        verts_uvs = mesh.textures.verts_uvs_list()[0].detach().cpu().numpy()
        faces_uvs = mesh.textures.faces_uvs_list()[0].detach().cpu().numpy()
        return bmesh_from_pydata(verts, faces, verts_uvs, faces_uvs)


def sample_nearest_surface(surface: bmesh.types.BMesh, points: list):
    """
    Samples the nearest surface to a list of points and returns the surface normals, tangents, and UV coordinates.

    Args:
        surface (bmesh.types.BMesh): The surface to sample.
        points (list): A list of points to sample the surface at.

    Returns:
        tuple: A tuple containing the surface normals, tangents, and UV coordinates.
    """

    def _calc_tangent(verts, uvs):
        edge1 = verts[1, :] - verts[0, :]
        edge2 = verts[2, :] - verts[0, :]
        delta_uv1 = uvs[1, :] - uvs[0, :]
        delta_uv2 = uvs[2, :] - uvs[0, :]

        tangent = delta_uv2[1] * edge1 - delta_uv1[1] * edge2
        tangent /= np.linalg.norm(tangent)
        return tangent

    surface_normals = []
    surface_tangents = []
    surface_uv_coords = []

    tree = BVHTree.FromBMesh(surface)
    uv_layer = surface.loops.layers.uv.active

    for idx, point in enumerate(points):
        location, normal, index, distance = tree.find_nearest(point)
        # assert distance < 0.005, f"index: {idx}, distance: {distance}"

        face = surface.faces[index]
        verts = np.array([l.vert.co for l in face.loops])
        uvs = np.array([l[uv_layer].uv for l in face.loops])

        bicentric = np.array(poly_3d_calc(verts, location))

        surface_normals += (normal,)
        surface_tangents += (_calc_tangent(verts, uvs),)
        surface_uv_coords += ((bicentric[:, None] * uvs).sum(0),)

    surface_normals = np.array(surface_normals, dtype=np.float32)
    surface_tangents = np.array(surface_tangents, dtype=np.float32)
    surface_uv_coords = np.array(surface_uv_coords, dtype=np.float32)

    return surface_normals, surface_tangents, surface_uv_coords


def sample_surface_texture(
    surface: bmesh.types.BMesh, points: list, texture: np.ndarray, mode: str = "nearest"
):
    _, _, uv_coords = sample_nearest_surface(surface, points)
    H, W = texture.shape[:2]

    xs = uv_coords[..., 0] * (W - 1)
    ys = (1 - uv_coords[..., 1]) * (H - 1)

    pix_values = texture[ys.astype(np.int32), xs.astype(np.int32)]

    return pix_values


def save_hair_strands(filepath: str, hair_strands: np.ndarray | torch.Tensor):
    write_hair_cy(filepath, [hair_strands.shape[1]]*hair_strands.shape[0], hair_strands.reshape(-1, 3))


def resample_strands_fast(strand_list: list, sample_num: int = 32, eps=1e-3):
    if isinstance(strand_list[0], np.ndarray):
        if all([strand.shape[0] == sample_num for strand in strand_list]):
            return np.stack(strand_list)
        
        new_strands = []
        
        for strand in strand_list:
            distances = np.linalg.norm(np.diff(strand, axis=0), axis=1)
            cum_distances = np.hstack([0, np.cumsum(distances)])
            if cum_distances[-1] < eps:
                continue

            t = np.linspace(0, cum_distances[-1], sample_num, endpoint=True)

            strand_new = np.zeros((t.shape[0], 3))
            strand_new[:, 0] = np.interp(t, cum_distances, strand[:, 0])
            strand_new[:, 1] = np.interp(t, cum_distances, strand[:, 1])
            strand_new[:, 2] = np.interp(t, cum_distances, strand[:, 2])
        
            new_strands.append(strand_new)

        return np.stack(new_strands)

    elif isinstance(strand_list[0], torch.Tensor):
        if all([strand.shape[0] == sample_num for strand in strand_list]):
            return torch.stack(strand_list)
        
        new_strands = []
        
        for strand in strand_list:
            distances = torch.norm(torch.diff(strand, dim=0), dim=-1)
            cum_distances = F.pad(torch.cumsum(distances, dim=0), (1, 0), value=0)
            if cum_distances[-1] < eps:
                continue

            t = torch.linspace(0, cum_distances[-1], sample_num, device=strand.device)

            indices = torch.searchsorted(cum_distances, t, side="right")[:-1] - 1
            weights = (t[:-1] - cum_distances[indices]) / (cum_distances[indices+1] - cum_distances[indices])
            weights = weights.unsqueeze(-1)
            strand_new = strand[indices] * (1 - weights) + strand[indices+1] * weights
            strand_new = torch.cat([strand_new, strand[-1:]], dim=0)
            
            new_strands.append(strand_new)

        return torch.stack(new_strands)


def resample_hair_strands(strand_list: list, sample_num: int = 64):
    """
    Resamples a list of hair strands using natural cubic spline interpolation.

    Args:
    - strand_list (list): A list of hair strands represented as torch.Tensor objects.
    - sample_num (int): The number of samples to use for the resampled hair strands.

    Returns:
    - A torch.Tensor object representing the resampled hair strands.
    """
    if all([strand.shape[0] == sample_num for strand in strand_list]):
        return torch.stack(strand_list)
    
    strand_interp_list = []
    
    for strand in strand_list:
        if isinstance(strand, np.ndarray):
            strand = torch.tensor(strand)
        t = torch.linspace(0, 1, strand.shape[0], device=strand.device)
        coeffs = natural_cubic_spline_coeffs(t, strand)
        spline = NaturalCubicSpline(coeffs)

        samples = torch.linspace(0, 1, sample_num, device=strand.device)
        strand_interp_list += [spline.evaluate(samples)]

    return torch.stack(strand_interp_list)



class HairInterpolator:
    def __init__(
        self,
        mesh: Meshes,
        guide_roots: torch.Tensor,
        mask_img: torch.Tensor,
        region_img: torch.Tensor = None,
        num_samples=1000,
        num_nn=5,
    ):
        if region_img is None:
            region_img = torch.zeros_like(mask_img, device=mask_img.device)

        # calc region values
        bl_mesh = bmesh_from_pytorch3d(mesh)
        region_img_np = region_img.detach().cpu().numpy()
        guide_regions = sample_surface_texture(
            bl_mesh, guide_roots, texture=region_img_np
        )
        guide_regions = torch.tensor(guide_regions, dtype=torch.float32, device=device)

        # create feature mesh
        assert len(mesh) == 1, "Only support single mesh"
        feature_map = torch.concatenate((region_img, mask_img[..., :1]), axis=-1)
        feature_mesh = Meshes(
            verts=mesh.verts_list(),
            faces=mesh.faces_list(),
            textures=TexturesUV(
                verts_uvs=mesh.textures.verts_uvs_list(),
                faces_uvs=mesh.textures.faces_uvs_list(),
                maps=[feature_map],
                sampling_mode="nearest",
            ),
        ).to(device)

        # sample and filter points
        sampled_points, features = sample_points_from_meshes(
            feature_mesh, num_samples, return_textures=True
        )
        prob = torch.rand(num_samples, device=sampled_points.device)
        alpha_values = features[..., -1]
        mask = prob < alpha_values[0]
        filtered_points = sampled_points[0][mask]

        # calc knn
        distances, indices, _ = knn_points(
            filtered_points[None], guide_roots[None], K=num_nn
        )

        # calc region weights
        knn_guide_regions = guide_regions[indices[0]]
        point_regions = features[0, :, :4][mask]
        is_same_region = torch.isclose(
            knn_guide_regions[:, :, :3], point_regions[:, None, :3]
        )
        is_same_region = is_same_region.all(-1).float()
        region_weights = is_same_region + (1 - is_same_region) * (
            1 - point_regions[:, -1:]
        )

        # final weights
        eps = 1e-8
        dist_weights = 1.0 / (distances + eps)
        weights = dist_weights * region_weights
        normalized_weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + eps)

        # results
        self.guide_roots = guide_roots
        self.sampled_points = filtered_points
        self.indices = indices[0]
        self.weights = normalized_weights[0]
        self.strands_interp = None

        guide_colors = plt.cm.Spectral(np.random.rand(self.guide_roots.shape[0]))
        self.guide_colors = torch.tensor(guide_colors, device=device)
        self.strand_colors = torch.sum(
            self.guide_colors[self.indices] * self.weights[:, :, None], dim=1
        )

    def eval(self, guides):
        guides_local = guides - self.guide_roots[:, None, :]
        strands_interp_local = torch.sum(
            guides_local[self.indices] * self.weights[:, :, None, None], dim=1
        )
        self.strands_interp = strands_interp_local + self.sampled_points[:, None, :]

        return self.strands_interp

    def save(self, out_path):
        if self.strands_interp is None:
            self.eval()

        strands_interp = self.strands_interp
        save_hair_strands(out_path, strands_interp)

