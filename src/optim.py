import os
import sys
import torch
import yaml
import argparse
import pickle
import pyexr
import time
import numpy as np

import torch.nn.functional as F
from torchvision.transforms.v2.functional import gaussian_blur
from matplotlib import pyplot as plt

from pytorch3d.structures import Curves

from .model.network import sNet
from .utils import *
from .utils import visualizer as hair_visualizer
from .hair_model import HairModel
from .evaluation import compute_hisa, compute_hida

device = torch.device("cuda")


def get_model(config):
    ckpt = torch.load(config["ckpt_path"], map_location=device)["state_dict"]
    in_channels = [ckpt["conv1.weight"].shape[1], ckpt["conv2.weight"].shape[1]]
    model = sNet(base_model="resnet18", out_dim=128, in_channels=in_channels)
    
    model.load_state_dict(ckpt)
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()
    
    return model


def get_references(config):
    if config["ref_img_path"].endswith(".exr"):
        # test dataset
        img_data = pyexr.open(config["ref_img_path"])
        ref_img_origin = img_data.get('default')[..., :3]
        ref_img_depth = torch.tensor(img_data.get('depth')[..., 0]).to(device)
        ref_img_orien = torch.tensor(img_data.get("orientation")[..., :3]).to(device)
        ref_img_silh = (ref_img_depth != 0.0).float()

        mask = torch.ones_like(ref_img_silh).cpu().numpy()

    else:
        # hairstep
        ref_img_origin = plt.imread(config["ref_img_path"])[..., :3]
        if ref_img_origin.dtype == np.uint8:
            ref_img_origin = (ref_img_origin / 255.0).astype(np.float32)
        
        mask = plt.imread(config["ref_seg_path"])
        if len(mask.shape) == 3:
            mask = mask[..., 0]

        ref_img_orien, ref_img_silh, ref_img_depth = load_ref_imgs_hairstep(
        config["ref_seg_path"], config["ref_orien_path"], config["ref_depth_path"], device=device)

    ref_img = (
        0.299 * ref_img_origin[..., 0]
        + 0.587 * ref_img_origin[..., 1]
        + 0.114 * ref_img_origin[..., 2]
    )
    ref_img *= mask
    ref_img = torch.tensor(ref_img).unsqueeze(0).to(device)

    os.makedirs(config["output_dir"], exist_ok=True)
    plt.imsave(os.path.join(config["output_dir"], "reference.png"), ref_img_origin)
    plt.imsave(os.path.join(config["output_dir"], "input.png"),
               ref_img[0,:,:,None].expand(-1, -1, 3).cpu().numpy().clip(0, 1))

    return ref_img, ref_img_depth, ref_img_orien, ref_img_silh


def search_best_param(config, model, ref_feature, H, W, cameras, mesh_zbuf, curves, hair_model):
    loss_list = []
    best_param : float = 0.0
    for scale in np.linspace(0.0, 1.0, 101):
        hair_strands_mod = hair_model.eval(scale)
        hair_strands_proj = transform_points_to_ndc(cameras, hair_strands_mod)
        curves.update_packed(hair_strands_proj.reshape(-1, 3))
        image_silh, image_depth, image_orien = render_feature_map(
            config["rasterize_hard"], curves, (W, H), mesh_zbuf, hair_model.clump_scale)

        image_feature = torch.cat([image_depth.unsqueeze(0), image_orien.permute(2, 0, 1)], dim=0)
        render_feature = F.normalize(model(None, image_feature[None]), dim=-1)[0]
        loss = 1 - torch.sum(ref_feature * render_feature, dim=-1)

        print(f"searching | loss: {loss.item():.4f}, scale: {scale:.4f}")

        loss_list.append(loss.item())
        if loss == min(loss_list):
            best_param = scale
            best_param_tensor = hair_model.clump_scale.clone()

    # vis search result
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    text = f"best parameter: {best_param:.4f}"
    fig.text(0.1, 0.01, text, ha='left', va='bottom')
    fig.text(0.1, 0.99, f"min loss: {min(loss_list):.04f}", ha='left', va='top')
    ax.plot(loss_list)
    ax.set_title("loss")
    fig.savefig(os.path.join(config["output_dir"], "fig_search.png"))

    plt.close("all")
    return best_param, best_param_tensor


def hairstep_loss(
    ref_image_orien: torch.Tensor,
    pred_image_orien: torch.Tensor,
    ref_image_depth: torch.Tensor,
    pred_image_depth: torch.Tensor,
):
    # mask
    ref_mask = ref_image_orien[..., :2].detach().sum(-1) > 0
    pred_mask = pred_image_orien[..., :2].detach().sum(-1) > 0
    mask = ref_mask & pred_mask
    labels = torch.ones(mask.sum()).to(device)

    # orientation loss
    ref_orien = ref_image_orien[mask][:, :2] * 2 - 1
    pred_orien = pred_image_orien[mask][:, :2] * 2 - 1

    loss_orien = F.cosine_embedding_loss(pred_orien, ref_orien, labels)

    # depth loss
    sobel = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],                         # x-direction
                          [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)   # y-direction
    sobel = sobel.to(ref_image_depth.device).unsqueeze(1)
    
    combined_depth = torch.stack([ref_image_depth, pred_image_depth]).unsqueeze(1)
    combined_depth = gaussian_blur(combined_depth, kernel_size=13, sigma=2.0)
    
    depth_grad = F.conv2d(combined_depth, sobel, padding=1)
    ref_depth_grad = depth_grad[0].permute(1, 2, 0)[mask]
    pred_depth_grad = depth_grad[1].permute(1, 2, 0)[mask]

    loss_depth = F.cosine_embedding_loss(pred_depth_grad, ref_depth_grad, labels)

    return loss_orien, loss_depth


def optimize_strands(config):
    hair_name = os.path.basename(config["output_dir"])
    os.makedirs(config["output_dir"], exist_ok=True)
    print("process on", hair_name)
    start_time = time.time()

    # model
    model = get_model(config)

    # reference
    ref_img, ref_img_depth, ref_img_orien, ref_img_silh = get_references(config)
    ref_feature = F.normalize(model(ref_img[None], None), dim=-1)[0]
    ref_img_mask = ref_img_orien[..., :2].sum(-1) > 0
    C, H, W = ref_img.shape

    hida_pairs = torch.tensor(np.load(config["hida_pair_path"])).to(device)
    hida_labels = torch.tensor(np.load(config["hida_label_path"])).to(device)

    # hair generater
    hair_model = HairModel(
        config["hair_path"],
        config["head_path"],
        config["n_guide"],
        config["n_sample"],
        config["n_cluster"],
        device=device,
    )
    ref_guides_laplacian = hair_model.guides.clone().diff(dim=1).diff(dim=1)

    # load scene
    cameras = load_cameras(config["camera_path"], device=device)
    head_mesh = load_obj_with_uv(config["head_path"], device=device)
    mesh_zbuf = render_meshes_zbuf(head_mesh, cameras, (W, H))
    curves = Curves(transform_points_to_ndc(cameras, hair_model.hair_strands))

    # eval init hair
    image_silh, image_depth, image_orien = render_feature_map(
        config["rasterize_hard"], curves, (W, H), mesh_zbuf)
    hisa = compute_hisa(ref_img_orien, image_orien)
    hida = compute_hida(image_depth, hida_pairs, hida_labels)
    print(f"init hisa: {hisa:.4f}, hida: {hida:.4f}")

    # optimization
    hair_guides = hair_model.guides
    hair_guides.requires_grad = True
    optimizer_guide = torch.optim.Adam([hair_guides], lr=config["lr_guide"])

    clump_scale = None
    if config["with_modifier"]:
        if config["clump_scale"] is None:
            best_param, best_param_tensor = search_best_param(
                config, model, ref_feature, H, W, cameras, mesh_zbuf, curves, hair_model)
        else:
            hair_model.eval(config["clump_scale"])
            best_param_tensor = hair_model.clump_scale.clone()
        # clump_scale = torch.ones(1, device=device) * best_param
        clump_scale = best_param_tensor
        clump_scale.requires_grad = True
        optimizer_clump = torch.optim.Adam([clump_scale], lr=config["lr_clump"])

    # save vis constants
    output_vis = os.path.join(config["output_dir"], "vis")
    os.makedirs(output_vis, exist_ok=True)
    print("save vis constants", output_vis)
    with open(os.path.join(output_vis, "config.yml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    torch.save({
        "guides": hair_guides.detach(),
        "guide_colors": hair_model.guide_colors,
        "strands_interp": hair_model.eval().detach(),
        "strand_colors": hair_model.strand_colors,
    }, os.path.join(output_vis, "constants.pt"))

    # main loop
    loss_outline_list = []
    loss_modifier_list = []
    clump_scale_list = []
    for i in range(config["epoch"]):
        hair_strands_mod = hair_model.eval(clump_scale)
        hair_strands_proj = transform_points_to_ndc(cameras, hair_strands_mod)
        curves.update_packed(hair_strands_proj.reshape(-1, 3))

        # outline loss
        image_silh, image_depth, image_orien = render_feature_map(
            config["rasterize_medium"], curves, (W, H), mesh_zbuf, hair_model.clump_scale
        )
        strands_root_tangent = F.normalize(hair_strands_mod.diff(dim=1)[:, 0, :], dim=-1)
        guides_laplacian = hair_guides.diff(dim=1).diff(dim=1)

        loss_silh = F.mse_loss(image_silh, ref_img_silh)
        loss_geom = F.mse_loss(ref_guides_laplacian, guides_laplacian)
        loss_orien, loss_depth = hairstep_loss(
            ref_img_orien, image_orien, ref_img_depth, image_depth
        )
        loss_smooth = hair_smooth_loss(hair_guides)
        loss_root = 1 - F.cosine_similarity(strands_root_tangent, hair_model.root_normals).clip(max=0.05).mean()
        loss_outline = [loss_silh, loss_orien, loss_depth, loss_geom, loss_smooth, loss_root]

        loss_outline = sum([w*l for w, l in zip(config["loss_outline_weights"], loss_outline)])
        loss_outline_list.append(loss_outline.item())
        info_str = f"iter: {i:04d}, silh: {loss_silh.item():.6f}, orien: {loss_orien.item():.6f}, depth: {loss_depth.item():.6f}, smooth: {loss_smooth.item():.6f}"

        # save vis results
        if i % config["vis_interval"] == 0:
            torch.save({
                "image_orien": image_orien.detach(),
                "image_silh": image_silh.detach(),
                "image_depth": image_depth.detach(),
                "guides": hair_guides.detach(),
                "strands_interp": hair_strands_mod.detach(),
            }, os.path.join(output_vis, f"iter_{i:04d}.pt"))

        # update
        optimizer_guide.zero_grad()
        loss_outline.backward(retain_graph=True)
        hair_guides.grad *= torch.linspace(0, 1, hair_guides.shape[1], device=device)[:, None]**0.1
        optimizer_guide.step()

        # modifier loss
        if config["with_modifier"]:
            image_silh, image_depth, image_orien = render_feature_map(
                config["rasterize_hard"], curves, (W, H), mesh_zbuf, hair_model.clump_scale
            )
            image_feature = torch.cat([image_depth.unsqueeze(0), image_orien.permute(2, 0, 1)], dim=0)
            render_feature = F.normalize(model(None, image_feature[None]), dim=-1)[0]
            loss_modifier = 1 - torch.sum(ref_feature * render_feature, dim=-1)

            loss_modifier_list.append(loss_modifier.item())
            clump_scale_list.append(clump_scale.mean().item())
            info_str += f", modifier: {loss_modifier.item():.6f}"

            # update
            optimizer_clump.zero_grad()
            loss_modifier.backward()
            optimizer_clump.step()
            with torch.no_grad():
                clump_scale.clamp_(0.0, 1.0)

        # evaluation
        hisa = compute_hisa(ref_img_orien, image_orien)
        hida = compute_hida(image_depth, hida_pairs, hida_labels)
        info_str += f", hisa: {hisa:.4f}, hida: {hida:.4f}"
        print(info_str)

    # vis optim result
    fig, axes = plt.subplots(1, 3, figsize=(5*3, 5))
    axes[0].plot(loss_outline_list)
    axes[0].set_title("loss outline")
    axes[1].plot(loss_modifier_list)
    axes[1].set_title("loss modifier")
    axes[2].plot(clump_scale_list)
    axes[2].set_title("clump scale")
    fig.savefig(os.path.join(config["output_dir"], "fig_optimize.png"))
    plt.close("all")

    # save results
    result_dir = os.path.join(config["output_dir"], "results")
    os.makedirs(result_dir, exist_ok=True)
    save_hair_strands(os.path.join(result_dir, "wo_modifier.hair"), hair_model.eval().detach())
    save_hair_strands(os.path.join(result_dir, "only_clumping.hair"),
                      hair_model.eval(clump_scale, add_noise=False, add_cut=False).detach())
    save_hair_strands(os.path.join(result_dir, "clumping_noise.hair"),
                      hair_model.eval(clump_scale, add_noise=True, add_cut=False).detach())
    save_hair_strands(os.path.join(result_dir, "full_modifier.hair"),
                      hair_model.eval(clump_scale, add_noise=True, add_cut=True).detach())
    hair_model.save(os.path.join(result_dir, "hair_model.pkl"))

    print(f"finish {hair_name}, time: {time.time() - start_time:.4f}")