import os
import sys
import torch
import numpy as np

from numba import njit, prange
from opensimplex.api import _noise3, _init
from fast_pytorch_kmeans import KMeans

from pytorch3d.io import load_hair, save_hair


def noise3array(points, frequency):
    return _noise3array(points, frequency, *_init(12))

@njit(cache=True, parallel=True)
def _noise3array(points, frequency, perm, perm_grad_index3):
    values = np.empty((points.shape[0]), dtype=np.float32)
    for i in prange(points.shape[0]):
        point = points[i] * 100 * frequency
        values[i] = _noise3(point[0], point[1], point[2], perm, perm_grad_index3)
    return values


def calc_hair_noise_offsets(hair_strands: torch.Tensor, frequency: float):
    seg_lens = torch.norm(hair_strands[:, 1:] - hair_strands[:, :-1], dim=-1)
    seg_lens = torch.cat([seg_lens, seg_lens[:, -1:]], dim=-1)
    seg_lens = torch.cumsum(seg_lens, dim=-1) * 100

    hair_roots = np.random.rand(hair_strands.shape[0], 3) * 100
    return _calc_offsets(
        hair_roots, seg_lens.cpu().numpy(), frequency, *_init(12)
    )

@njit(cache=True, parallel=True)
def _calc_offsets(hair_roots, seg_lens, frequency, perm, perm_grad_index3):
    offsets = np.empty((hair_roots.shape[0], seg_lens.shape[1], 3), dtype=np.float32)

    for i_s in prange(offsets.shape[0]):
        for i_p in prange(offsets.shape[1]):
            root = hair_roots[i_s]
            dist = seg_lens[i_s, i_p] * frequency
            offsets[i_s, i_p, 0] = _noise3(root[0] + dist, root[1], root[2], perm, perm_grad_index3)
            offsets[i_s, i_p, 1] = _noise3(root[0], root[1] + dist, root[2], perm, perm_grad_index3)
            offsets[i_s, i_p, 2] = _noise3(root[0], root[1], root[2] + dist, perm, perm_grad_index3)

    return offsets


class Cut:
    def __init__(self, hair_strands: torch.Tensor, cut_ratio: float = 0.5):
        self.hair_strands = hair_strands
        self.device = hair_strands.device
        num_strands, n_sample, _ = hair_strands.shape

        cut_ratio = 1 - torch.rand((num_strands, 1), device=self.device) * cut_ratio
        new_indices = torch.arange(0, n_sample, device=self.device) * cut_ratio
        new_indices_lower = torch.floor(new_indices).long()
        new_indices_upper = torch.ceil(new_indices).long()
        weights = new_indices - new_indices_lower.float()

        self.cut_idx_lower = new_indices_lower.unsqueeze(2).expand(-1, -1, 3)
        self.cut_idx_upper = new_indices_upper.unsqueeze(2).expand(-1, -1, 3)
        self.cut_weights = weights.unsqueeze(2)

    def eval(self, hair_strands=None):
        if hair_strands is None:
            hair_strands = self.hair_strands
        
        lower_points = hair_strands.gather(1, self.cut_idx_lower)
        upper_points = hair_strands.gather(1, self.cut_idx_upper)
        hair_modified = (1 - self.cut_weights) * lower_points + self.cut_weights * upper_points

        return hair_modified


class Noise:
    def __init__(
        self, hair_strands: torch.Tensor, frequency: float = 1, seed: int = 12
    ):
        self.hair_strands = hair_strands
        self.device = hair_strands.device

        seg_lens = torch.norm(hair_strands[:, 1:] - hair_strands[:, :-1], dim=-1)
        seg_lens = torch.cat([seg_lens, seg_lens[:, -1:]], dim=-1)
        seg_lens = torch.cumsum(seg_lens, dim=-1) * 100

        hair_roots = np.random.rand(hair_strands.shape[0], 3) * 100
        offsets = _calc_offsets(
            hair_roots, seg_lens.cpu().numpy(), frequency, *_init(seed)
        )
        self.offsets = torch.tensor(offsets, device=self.device)

        self.factor = torch.linspace(0, 1, hair_strands.shape[1], device=self.device)

    def eval(self, scale: float = None, shape: float = 0.5, hair_strands=None):
        factor = self.factor[None, :, None]
        if hair_strands is None:
            hair_strands = self.hair_strands
        
        if type(scale) == torch.Tensor:
            factor = factor.expand(hair_strands.shape[0], -1, 3)
        weights = factor**shape * scale

        return hair_strands + self.offsets * weights


class Clumping:
    def __init__(
        self,
        hair_strands: torch.Tensor,
        n_clusters: int = 256,
        noise_frequency: float = 0.1,
    ):
        self.hair_strands = hair_strands
        self.n_clusters = n_clusters
        self.device = hair_strands.device

        num_strands = hair_strands.shape[0]
        kmeans = KMeans(
            n_clusters=n_clusters, mode="euclidean", max_iter=100, verbose=1
        )
        self.guide_indices = kmeans.fit_predict(hair_strands.detach().reshape(num_strands, -1))
        self.hair_guides = kmeans.centroids.reshape(n_clusters, -1, 3)

        self.noise = Noise(self.hair_guides, frequency=noise_frequency)
        self.factor = torch.linspace(0, 1, hair_strands.shape[1], device=self.device)

    def eval(
        self,
        scale: float,
        noise: float,
        shape: float = 0.5,
        shape_noise: float = 0.2,
        volumize: bool = False,
    ):
        factor = self.factor[None, :, None]
        if type(scale) == torch.Tensor:
            factor = factor.expand(self.hair_strands.shape[0], -1, 3)
        weights = factor**shape * scale
        
        hair_guides = self.noise.eval(noise, shape=shape_noise)
        hair_nn = hair_guides[self.guide_indices]
        hair_delta = hair_nn - self.hair_strands
        
        if volumize:
            # Blend between space curve version and groomed point. This
            # pushes the clump towards round as it is clumped.
            space = hair_nn.cross(hair_delta)
            space = space.cross(hair_nn)
            vlen = hair_delta.norm(dim=-1, keepdim=True)
            slen = space.norm(dim=-1, keepdim=True) + 1e-10
            space = space / slen * vlen
            
            v_weights = weights.pow(0.7).detach()
            hair_delta = v_weights * hair_delta + (1 - v_weights) * space.detach()

        return self.hair_strands + hair_delta * weights


class HairModifier:
    def __init__(self, hair_strands: torch.Tensor, n_clusters: int = 256):
        self.hair_strands = hair_strands
        self.device = hair_strands.device
        params = {
            "clumping": {
                "n_clusters": n_clusters,
                "noise_freq": 0.05,
                "scale": 0.5,
                "noise": 0.01,
                "shape": 0.5,
                "shape_noise": 0.1,
                "volumize": False,
            },
            "noise": {
                "frequency": 0.1,
                "scale": 0.002,
                "shape": 0.1
            },
            "cut": {
                "scale": 0.3,
            },
        }
        self.params = params

        # clumping
        hair_roots = hair_strands[:, 0].cpu().numpy()
        clump_rand = noise3array(hair_roots, 0.5)
        self.clump_rand = torch.tensor(clump_rand)[:, None, None].to(self.device)
        self.clumping = Clumping(
            hair_strands,
            n_clusters=params["clumping"]["n_clusters"],
            noise_frequency=params["clumping"]["noise_freq"],
        )

        # noise
        noise_rand = torch.randn((hair_strands.shape[0], 1, 1)).to(self.device)
        mask = torch.rand(noise_rand.shape).to(self.device) < 0.1
        noise_rand[mask] = noise_rand[mask] * 2 + 4
        self.noise_mask = mask
        self.noise_rand = noise_rand
        self.noise = Noise(
            hair_strands.detach(),
            frequency=params["noise"]["frequency"]
        )
        
        # cut
        self.cut = Cut(hair_strands, cut_ratio=params["cut"]["scale"])
        

    def eval(self, scale: float | torch.Tensor, add_noise: bool = True, add_cut: bool = True):
        params = self.params

        # clumping
        if isinstance(scale, float) or scale.shape[0] == 1:
            clump_scale = (scale * 10 - 5 + self.clump_rand * 2).sigmoid()
        else:
            clump_scale = scale
        self.clump_scale = clump_scale

        modified_hair = self.clumping.eval(
            clump_scale,
            params["clumping"]["noise"],
            params["clumping"]["shape"],
            params["clumping"]["shape_noise"],
            volumize=params["clumping"]["volumize"],
        )

        # noise
        if add_noise:
            noise_scale = params["noise"]["scale"] * (1 + self.noise_rand)
            noise_scale[~self.noise_mask] *= 1 - clump_scale[~self.noise_mask]

            modified_hair = self.noise.eval(
                noise_scale,
                params["noise"]["shape"],
                modified_hair,
            )
        
        # cut
        if add_cut:
            modified_hair = self.cut.eval(modified_hair)

        return modified_hair


if __name__ == "__main__":
    from .visualizaton import render_hair_shading
    
    torch.random.manual_seed(0)
    np.random.seed(0)

    for hair_name in [
        "0d285f5be7fa09c3dbbf1c9334047888",
        "29dc6387017754215b532130750e2370",
        "86a0c5ecca8721acb831a702880c519e",
        "5914e5debe3331898796dd11f5cd5a30",
        "97460f1480c7c477919a68ebb1a660ba",
        "Adrianne-Palicki-06_2880x1800",
        "Amy-Adams-10_1920x1440",
        "cc5bf02bc6205f878cde02df9d00e7b9",
        "f_11403540",
        "GUEST_fdaf1def-ce59-4cea-acda-230b4de4dbdd",
    ]:
        hair_path = f"../HairStep/results/real_imgs/hair3D/80k_resample/{hair_name}.hair"
        model_path = "../HairStep/data/head_model_uv.obj"
        conf_path = "data/modifiers/params_template.json"

        hair_strands = torch.stack(load_hair(hair_path, device="cuda"))

        modifier = HairModifier(hair_strands, n_clusters=2048)
        
        for scale in np.linspace(0, 1, 11):
            output_path = f"X:/differential_rendering/modifiers/modifier_vis/2048_test/{hair_name}/scale_{scale:.1f}.hair"

            modified_hair = modifier.eval(scale)

            render_hair_shading(model_path, hair_path, modified_hair.cpu().numpy(), os.path.realpath(output_path),
                        render_origin=False, side_view=False, render_data=False, melanin=0.2, img_size=512, save_proj=False)
