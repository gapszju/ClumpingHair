import os
import sys 
import yaml
import time
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np

from torch.nn import functional as F
from pytorch3d.io import load_obj
from pytorch3d.structures import Curves

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
from src.utils import *
from src.hair_model import HairModel
from src.optim import get_references, hairstep_loss

device = torch.device("cuda")


def soft_rasterize(config, strands, curves, cameras, mesh_zbuf):
    W, H = 512, 512
    strands_proj = transform_points_to_ndc(cameras, strands)
    curves.update_packed(strands_proj.reshape(-1, 3))
    image_silh, image_depth, image_orien = render_feature_map(config, curves, (W, H), mesh_zbuf)

    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # axes[0].imshow(image_silh.cpu().numpy())
    # axes[1].imshow(image_depth.cpu().numpy())
    # axes[2].imshow(image_orien.cpu().numpy())
    # plt.show()
    
    return image_silh, image_depth, image_orien


def quad_rasterize(config, rasterizer, strands, cameras):
    image = rasterizer(strands, cameras)[0].permute(1,2,0)
    image_silh = image[..., 2]
    
    image_depth = image[..., 3]
    image_depth = image_depth * image_silh.detach()
    zbuf_min = image_depth.min().item()
    zbuf_range = image_depth.max().item() - zbuf_min
    image_depth = (image_depth - zbuf_min) / zbuf_range
    image_depth = 1 - image_depth
    image_depth = image_depth * image_silh.detach()
    
    image_orien = image[..., :3] * 0.5 + 0.5
    image_orien[..., 2] = 0.5
    image_orien = image_orien * image_silh.detach().unsqueeze(-1)
    
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # axes[0].imshow(image_silh.cpu().numpy())
    # axes[1].imshow(image_depth.cpu().numpy())
    # axes[2].imshow(image_orien.cpu().numpy())
    # plt.show()

    return image_silh, image_depth, image_orien


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./config/config_sample.yml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--method', type=str, default="line")

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    torch.cuda.random.manual_seed(0)

    with open(args.conf, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["epoch"] = 500
    config["lr_guide"] = 0.001
    
    input_dir = "X:/hairstep/Real_Image"
    output_dir = "X:/results/renderer_compare"
    # hair_name = "pexels-rukiye-agacayak-686493765-20271519"
    hair_name = "joao-paulo-de-souza-oliveira-x-FNmzxyQ94-unsplash"
    config["ref_img_path"] = os.path.join(input_dir, "resized_img", hair_name+".png")
    config["head_path"] = os.path.join(ROOT_DIR, "assets", "head_model_metahuman.obj")
    config["hair_path"] = os.path.join(input_dir, "hair3D/resample_32", hair_name+".hair")
    config["camera_path"] = os.path.join(input_dir, "param", hair_name+".npy")
    config["ref_seg_path"] = os.path.join(input_dir, "seg", hair_name+".png")
    config["ref_orien_path"] = os.path.join(input_dir, "strand_map", hair_name+".png")
    config["ref_depth_path"] = os.path.join(input_dir, "depth_map", hair_name+".npy")
    config["output_dir"] = os.path.join(output_dir, hair_name, args.method)

    os.makedirs(config["output_dir"], exist_ok=True)
    print("process on", hair_name)

    # reference
    ref_img, ref_img_depth, ref_img_orien, ref_img_silh = get_references(config)
    ref_img_mask = ref_img_orien[..., :2].sum(-1) > 0
    C, H, W = ref_img.shape

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
    
    # init scene
    cameras = load_cameras(config["camera_path"], device=device)
    head_mesh = load_obj_with_uv(config["head_path"], device=device)
    mesh_zbuf = render_meshes_zbuf(head_mesh, cameras, (W, H))
    curves = Curves(transform_points_to_ndc(cameras, hair_model.hair_strands))
    
    verts, faces, _ = load_obj(config["head_path"], device=device)
    rasterizer = QuadRasterizer(
        render_size=H,
        head_mesh=(verts, faces.verts_idx),
        quad_w=1e-3,
        faces_per_pixel=config["rasterize_medium"]["lines_per_pixel"] * 2,
        blur_radius=(config["rasterize_medium"]["blur_radius"] / H) ** 2,
        sigma=config["rasterize_medium"]["sigma"],
        gamma=config["rasterize_medium"]["gamma"],
    ).to(device)

    # optimization
    hair_guides = hair_model.guides
    hair_guides.requires_grad = True
    optimizer = torch.optim.Adam([hair_guides], lr=config["lr_guide"])

    # main loop
    loss_list = []
    time_start = time.time()
    memory_start = torch.cuda.memory_allocated(device)
    cost_memory_max = 0
    print("start optimization")
    for i in range(config["epoch"]):
        hair_strands = hair_model.eval()
        
        # rendering
        if args.method == "line":
            image_silh, image_depth, image_orien = soft_rasterize(
                config["rasterize_medium"], hair_strands, curves, cameras, mesh_zbuf)
        elif args.method == "quad":
            image_silh, image_depth, image_orien = quad_rasterize(
                config["rasterize_medium"], rasterizer, hair_strands, cameras)
        else:
            raise ValueError("invalid")
        
        # calculate loss
        strands_root_tangent = F.normalize(hair_strands.diff(dim=1)[:, 0, :], dim=-1)
        guides_laplacian = hair_guides.diff(dim=1).diff(dim=1)

        loss_silh = F.mse_loss(image_silh, ref_img_silh)
        loss_geom = F.mse_loss(ref_guides_laplacian, guides_laplacian)
        loss_orien, loss_depth = hairstep_loss(
            ref_img_orien, image_orien, ref_img_depth, image_depth
        )
        loss = loss_silh + loss_orien + 1e4 * loss_geom

        cost_memory_max = max(cost_memory_max, torch.cuda.memory_allocated(device)-memory_start)
        
        # update
        optimizer.zero_grad()
        loss.backward()
        hair_guides.grad *= torch.linspace(0, 1, hair_guides.shape[1], device=device)[:, None]**0.1
        optimizer.step()

        # eval iou
        inter = (image_silh.detach() > 0.5) & ref_img_silh.bool()
        union = (image_silh.detach() > 0.5) | ref_img_silh.bool()
        iou = inter.float().sum().item() / union.float().sum().item()

        loss_list.append(loss.item())
        print(f"iter: {i:04d}, IoU: {iou:.6f}, orien: {loss_orien.item():.6f}")

    print("optimization done")
    print(f"cost time: {time.time()-time_start:.2f}s")
    print("cost memory:", cost_memory_max / 1024**2)
    
    plt.plot(loss_list)
    plt.savefig(os.path.join(config["output_dir"], "loss.png"))