import os
import sys
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from pytorch3d.io import load_hair, save_hair
from pytorch3d.ops import knn_points
from fast_pytorch_kmeans import KMeans

from .utils import *

device = torch.device("cuda")


def calc_optimized_guides(strands: torch.Tensor, n_clusters, num_nn, epoch=5000, p=2.5, use_lstsq=False):
    eps = 1e-10
    
    strand_roots = strands[:, 0, :]
    strands_local = strands - strand_roots[:, None, :]
    num_strands = strands.shape[0]
    
    # calc init guides
    kmeans = KMeans(n_clusters=n_clusters, mode="euclidean", max_iter=100, verbose=0)
    kmeans.fit_predict(strands.reshape(num_strands, -1))
    guides = kmeans.centroids.reshape(n_clusters, -1, 3)
    valid = [guide.sum()>0 for guide in guides]
    guides = guides[valid]
    guide_roots = guides[:, 0, :]
    guides_local = guides - guide_roots[:, None, :]

    # calc init weights
    dists, indices, _ = knn_points(strand_roots.unsqueeze(0), guide_roots.unsqueeze(0), K=num_nn)
    dists, indices = dists[0], indices[0]
    
    if use_lstsq:
        # linear regression
        strand_guide_nn = guides_local[indices].reshape(*indices.shape, -1).permute(0, 2, 1)
        weights = torch.linalg.lstsq(
            strand_guide_nn, strands_local.reshape(indices.shape[0], -1, 1)
        ).solution.squeeze(-1)
    
        return guides, weights, indices
    
    else:
        weights = 1.0 / dists.clip(eps)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).clip(eps)
        
        ref_guide_diff = torch.diff(guides_local, dim=1).clone()
    
        # optimize
        weights.requires_grad = True
        guides_local.requires_grad = True
        optimizer = torch.optim.Adam([weights, guides_local], lr=1e-4)
        time_start = time.time()
        for i in range(epoch):
            optimizer.zero_grad()
            
            # calc strands
            strands_interp_local = torch.sum(
                guides_local[indices] * weights[:, :, None, None], dim=1
            )
            strands_interp = strands_interp_local + strand_roots[:, None, :]
            
            # cal tangents
            guide_diff = torch.diff(guides_local, dim=1)

            # calc loss
            loss_geo = generalized_mean((strands_interp - strands).norm(dim=-1), p)
            loss_tangent = F.l1_loss(guide_diff, ref_guide_diff)
            loss_reg = torch.mean((1 - weights.sum(dim=-1))**2)
            loss_smooth = hair_smooth_loss(guides_local)
            
            loss = loss_geo + 0.01*loss_tangent + 0.1*loss_reg + 1*loss_smooth
            
            loss.backward()
            # print(f"epoch: {i:04d}, loss_geo: {loss_geo.item():.5f}, loss_reg: {loss_reg.item():.5f}, loss_smooth: {loss_smooth.item():.5f}")

            optimizer.step()
            with torch.no_grad():
                weights.clamp_(0, 1)
                guides_local[:, 0, :] = 0

        time_elapsed = time.time() - time_start
        print(f"used {epoch} iterations ({time_elapsed:.2f}s) to optimize guides")
    
        guides = guides_local + guide_roots[:, None, :]
        return guides.detach(), weights.detach(), indices


class HairModel:
    def __init__(
        self,
        hair_path: str,
        head_path: str = None,
        n_guide=512,
        n_sample=32,
        n_cluster=256,
        device="cpu",
    ):
        self.hair_list = load_hair(hair_path, device=device)
        self.num_strands = len(self.hair_list)
        self.hair_roots = torch.stack([hair[0] for hair in self.hair_list])
        self.n_sample = n_sample
        self.n_cluster = n_cluster
        self.device = device

        # guides
        print("Reparameterizing hair strands...")
        self._calc_guides(n_guide, n_sample, device)
        print("Calculating modifiers...")
        self._calc_modifiers(n_cluster)

        # head
        if head_path is not None:
            self.head_mesh = load_obj_with_uv(head_path, device=device)
            self._calc_root_normals()

    def _calc_guides(self, n_guide, n_sample, device):
        hair_list_cpu = [curve.cpu() for curve in self.hair_list]
        hair_strands = resample_strands_fast(hair_list_cpu, n_sample).to(device)
        guides, weights, indices = calc_optimized_guides(
            hair_strands, n_guide, num_nn=32, epoch=5000, p=2.7, use_lstsq=False
        )
        self.guides : torch.Tensor = guides
        self.guide_indices : torch.Tensor = indices
        self.guide_weights : torch.Tensor = weights
        self.guide_roots = guides[:, 0, :]
        self.hair_strands = hair_strands

        # for visualization
        guide_colors = plt.cm.Spectral(np.random.rand(guides.shape[0]))
        self.guide_colors = torch.tensor(guide_colors, device=device)
        self.strand_colors = torch.sum(self.guide_colors[indices] * weights[:, :, None], dim=1)

    def _calc_root_normals(self):
        hair_roots = self.hair_strands[:, 0, :].detach().cpu().numpy()
        bl_mesh = bmesh_from_pytorch3d(self.head_mesh)
        normals, tangents, _ = sample_nearest_surface(bl_mesh, hair_roots)
        self.root_normals = torch.tensor(normals, device=self.device)

    def get_hair_interp(self):
        guides_local = self.guides - self.guides[:, :1, :]
        n_sample = self.n_sample

        idx = self.guide_indices.reshape(-1, 1, 1).expand(-1, n_sample, 3)
        hair_guides_nn = guides_local.gather(0, idx)
        hair_guides_nn = hair_guides_nn.reshape(*self.guide_indices.shape, n_sample, 3)
        hair_interp_local = torch.sum(
            hair_guides_nn * self.guide_weights[:, :, None, None], dim=1
        )
        hair_interp = hair_interp_local + self.hair_roots[:, None, :]

        return hair_interp

    def update_modifiers(self):
        self._calc_modifiers(self.n_cluster)

    def _calc_modifiers(self, n_cluster):
        params = {
            "clumping": {
                "n_clusters": n_cluster,
                "scale": 0.5,
                "shape": 0.5,
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

        # cluster centers
        hair_interp = self.get_hair_interp().detach()
        cluster_indices = KMeans(
            n_clusters=n_cluster, mode="euclidean", max_iter=100, verbose=1
        ).fit_predict(hair_interp.reshape(self.num_strands, -1))

        clusters = [[] for i in range(n_cluster)]
        for i, idx in enumerate(cluster_indices):
            clusters[idx].append(i)

        self.clusters = clusters
        self.cluster_indices = cluster_indices

        factor = torch.linspace(0, 1, self.n_sample, device=self.device)[None, :, None]

        # clumping
        clump_rand = noise3array(self.hair_roots.cpu().numpy(), 0.5)
        self.clump_rand = torch.tensor(clump_rand)[:, None, None].to(self.device)
        self.clump_scale = torch.zeros_like(self.clump_rand).to(self.device) + 0.1
        self.clump_factor = factor**params["clumping"]["shape"]

        # noise
        noise_rand = torch.randn((self.num_strands, 1, 1)).to(self.device)
        mask = torch.rand(noise_rand.shape).to(self.device) < 0.1
        noise_rand[mask] = noise_rand[mask] * 2 + 4
        noise_scale = params["noise"]["scale"] * (1 + noise_rand)

        hair_noise = calc_hair_noise_offsets(hair_interp, params["noise"]["frequency"])
        hair_noise = torch.tensor(hair_noise).to(self.device)
        self.hair_noise = hair_noise
        self.noise_scale =  noise_scale
        self.noise_factor = factor**params["noise"]["shape"]
        self.noise_mask = mask

        # cut
        cut_ratio = 1 - torch.rand((self.num_strands, 1), device=self.device) * params["cut"]["scale"]
        new_indices = torch.arange(0, self.n_sample, device=self.device) * cut_ratio
        new_indices_lower = torch.floor(new_indices).long()
        new_indices_upper = torch.ceil(new_indices).long()
        weights = new_indices - new_indices_lower.float()

        self.cut_idx_lower = new_indices_lower.unsqueeze(2).expand(-1, -1, 3)
        self.cut_idx_upper = new_indices_upper.unsqueeze(2).expand(-1, -1, 3)
        self.cut_weights = weights.unsqueeze(2)

    def get_cluster_offsets(self, hair_interp):
        # calc cluster centers
        cluster_centers = []
        for cluster in self.clusters:
            idx = torch.tensor(cluster).to(self.device)
            idx = idx.reshape(-1, 1, 1).expand(-1, self.n_sample, 3)
            cluster_hair = hair_interp.gather(0, idx)

            if len(cluster_hair) == 0:
                cluster_center = torch.zeros(self.n_sample, 3).to(self.device)
            else:
                cluster_center = cluster_hair.mean(0)
            cluster_centers.append(cluster_center)

        cluster_centers = torch.stack(cluster_centers)
        self.cluster_centers = cluster_centers

        # calc hair offsets
        idx = self.cluster_indices.reshape(-1, 1, 1).expand(-1, self.n_sample, 3)
        cluster_center_per_strand = cluster_centers.gather(0, idx)
        hair_offsets = cluster_center_per_strand - hair_interp

        return hair_offsets

    def eval(
        self,
        clump_scale: float | torch.Tensor = None,
        add_noise: bool = True,
        add_cut: bool = True,
    ):
        hair_interp = self.get_hair_interp()
        if clump_scale is None:
            return hair_interp

        # clumping
        if not isinstance(clump_scale, torch.Tensor) or clump_scale.shape[0] == 1:
            clump_scale = (clump_scale * 10 - 5 + self.clump_rand * 2).sigmoid()
        self.clump_scale = clump_scale

        clump_factor = self.clump_factor
        hair_offsets = self.get_cluster_offsets(hair_interp)
        hair_modified = hair_interp + hair_offsets * clump_factor * clump_scale

        # noise
        if add_noise:
            noise_scale = self.noise_scale.clone()
            noise_scale[~self.noise_mask] *= 1 - clump_scale[~self.noise_mask]
            hair_modified += self.hair_noise * self.noise_factor * noise_scale

        # cut
        if add_cut:
            lower_points = hair_modified.gather(1, self.cut_idx_lower)
            upper_points = hair_modified.gather(1, self.cut_idx_upper)
            hair_modified = (1 - self.cut_weights) * lower_points + self.cut_weights * upper_points

        return hair_modified


if __name__ == "__main__":
    head_path = "../HairStep/data/head_smooth.obj"
    mask_path = "data/hairstep/scalp_mask.png"
    hair_path = "X:/hairstep/USC_HairSalon/hair3D/resample_32/strands00187.hair"
    out_path = "output/hair_interp_resample.hair"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    hair_model = HairModel(hair_path, n_guide=512, device=device)
    hair = hair_model.eval()
    hair_guide = hair_model.guides
    save_hair(out_path, [hair.shape[1]] * hair.shape[0], hair.reshape(-1, 3).detach().cpu().numpy())
    save_hair(out_path.replace(".hair", "_guide.hair"), [hair_guide.shape[1]] * hair_guide.shape[0], hair_guide.reshape(-1, 3).detach().cpu().numpy())
    
    from src.utils.visualizaton import render_hair_shading
    render_hair_shading(head_path, hair_path, hair.detach().cpu().numpy(),
                os.path.realpath(out_path).replace(".hair", ".png"),
                img_size=1024, render_origin=False, side_view=True,
                device_idx=torch.cuda.current_device()-torch.cuda.device_count())
