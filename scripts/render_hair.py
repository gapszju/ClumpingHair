import os
import sys 
import torch
import argparse
import matplotlib.pyplot as plt
from pytorch3d.io import load_hair

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils.visualizaton import (
    render_hair_shading,
    render_hair_color,
    render_hair_projection,
    render_hair_template,
)
from src.utils import *


def soft_rasterize(head_path, hair_path, camera_path):
    config = {    
        "blur_radius": 8,
        "lines_per_pixel": 128,
        "bin_size": None,
        "sigma": 0.00001,
        "gamma": 0.0001
    }
    W, H = 512, 512
    device = torch.device("cuda")
    
    hair_strands = torch.stack(load_hair(hair_path, device=device))
    cameras = load_cameras(camera_path, device=device)
    strands_proj = transform_points_to_ndc(cameras, hair_strands)
    head_mesh = load_obj_with_uv(head_path, device=device)
    mesh_zbuf = render_meshes_zbuf(head_mesh, cameras, (W, H))
    curves = Curves(strands_proj)
    image_silh, image_depth, image_orien = render_feature_map(config, curves, (W, H), mesh_zbuf)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_silh.cpu().numpy())
    axes[1].imshow(image_depth.cpu().numpy())
    axes[2].imshow(image_orien.cpu().numpy())
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    head_path = "X:/hairstep/head_model_metahuman.obj"
    hair_path = "X:/hairstep/HiSa_HiDa/hair3D/resample_32/4.hair"
    result_hair_path = "X:/results/reconstruction/hairstep/HiSa_HiDa/4/results/full_modifier.hair"
    camera_path = "X:/hairstep/HiSa_HiDa/param/4.npy"
    output_dir = "X:/results/reconstruction/hairstep/HiSa_HiDa/4"

    # soft_rasterize(head_path, hair_path, camera_path)

    # render_hair_shading(
    #     head_path, hair_path, os.path.join(output_dir, "render_vis", "vis.png"),
    #     img_size=1024, side_view=True, device_idx=args.gpu,
    # )

    # render_hair_template(
    #     hair_path, os.path.join(output_dir, "render", "render.png"), img_size=1024, device_idx=args.gpu,
    # )

    render_hair_projection(
        head_path, hair_path, camera_path,
        os.path.join(output_dir, "projection", "origin.png"),
        img_size=1024, device_idx=args.gpu,
    )
    # render_hair_projection(
    #     head_path, result_hair_path, camera_path,
    #     os.path.join(output_dir, "projection", "result.png"),
    #     img_size=1024, device_idx=args.gpu,
    # )