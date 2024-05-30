import os
import sys
import torch
import yaml
import argparse
import pyexr
import json
import numpy as np

from matplotlib import pyplot as plt

from pytorch3d.structures import Curves
from pytorch3d.io import load_hair, save_hair

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils import *
from src.utils.modifiers import HairModifier

device = torch.device("cuda")


def render_param_dataset(config):
    # load param
    with open(os.path.join(os.path.dirname(__file__), "param_dict.json"), "r") as f:
        param_dict = json.load(f)
    if hair_name not in param_dict:
        print(f"Skip {hair_name}")
        return

    label = param_dict[hair_name]
    
    # reference image
    ref_img_origin = plt.imread(config["ref_img_path"])[..., :3]
    mask = plt.imread(config["ref_seg_path"])
    if len(mask.shape) == 3:
        mask = mask[..., 0]
    ref_img_origin[mask < 0.5] *= 0
    H, W, C = ref_img_origin.shape
    
    # load scene
    cameras = load_cameras(config["camera_path"], device=device)
    head_mesh = load_obj_with_uv(config["head_path"], device=device)
    mesh_zbuf = render_meshes_zbuf(head_mesh, cameras, (W, H))
    hair_strands = resample_strands_fast(load_hair(config["hair_path"]), 32).to(device)
    curves = Curves(hair_strands)

    # modifiers
    modifier = HairModifier(hair_strands, n_clusters=config["n_cluster"])

    hair_strands_mod = modifier.eval(label)
    hair_strands_proj = transform_points_to_ndc(cameras, hair_strands_mod)
    curves.update_packed(hair_strands_proj.reshape(-1, 3))
    image_silh, image_depth, image_orien = render_feature_map(
        config["rasterize_hard"], curves, (W, H), mesh_zbuf, modifier.clump_scale)
    
    tangent = image_orien[..., :2].detach().cpu().numpy()
    tangent = np.concatenate([tangent, np.ones_like(tangent[..., :1]) * 0.5], axis=-1)
    tangent[image_silh.cpu() < 0.5] *= 0
    data = {
        "default": ref_img_origin,
        "orientation": tangent,
        "depth": image_depth.detach().cpu().numpy(),
        "aov": image_orien[..., 2].detach().cpu().numpy(),
    }
    os.makedirs(os.path.dirname(config["output_path"]), exist_ok=True)
    pyexr.write(config["output_path"], data)
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./config/config.yml')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    config = yaml.load(open(args.conf, "r", encoding="utf-8"), Loader=yaml.FullLoader)

    input_dir = "X:/hairstep/HiSa_HiDa"
    ref_img_dir = os.path.join(input_dir, "resized_img")
    output_dir = "X:/contrastive_learning/data/clumping_dataset/real_img_manul_label"

    for hair_name in os.listdir(ref_img_dir)[::-1]:
        hair_name = hair_name.split(".")[0]
        config["n_cluster"] = 1024
        config["ref_img_path"] = os.path.join(input_dir, "resized_img", hair_name+".png")
        config["hair_path"] = os.path.join(input_dir, "hair3D/resample_32", hair_name+".hair")
        config["camera_path"] = os.path.join(input_dir, "param", hair_name+".npy")
        config["ref_seg_path"] = os.path.join(input_dir, "seg", hair_name+".png")
        config["output_path"] = os.path.join(output_dir, hair_name+".exr")

        if os.path.exists(config["output_path"]):
            print(f"Skip {hair_name}")
            continue
        
        render_param_dataset(config)