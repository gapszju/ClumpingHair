import os
import sys
import torch
import yaml
import glob
import argparse
import pyexr
import numpy as np

from matplotlib import pyplot as plt
from torch.nn.functional import normalize
from pytorch3d.structures import Curves
from pytorch3d.io import load_hair, save_hair

device = torch.device("cuda")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
from src.model.network import sNet
from src.optim import get_references, get_model, search_best_param
from src.utils import *


def pred_param(target, random):
    value = torch.tensor([0.5], requires_grad=True, device=device)
    adam = torch.optim.Adam([value], lr=0.002)
    for i in range(200):
        adam.zero_grad()
        result = (value * 10 - 5 + random * 2).sigmoid().mean()
        loss = (result - target)**2
        loss.backward()
        adam.step()
    return value.item()


def regression_param(config):
    ckpt = torch.load(config["ckpt_regression_path"], map_location=device)["state_dict"]
    model = sNet(base_model="resnet18", out_dim=1, in_channels=[1, 1])
    
    model.load_state_dict(ckpt)
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()
    
    ref_img, _, _, _ = get_references(config)
    scale = model(ref_img[None], None)[0].sigmoid().item()
    
    hair_strands = torch.stack(load_hair(config["hair_path"], device=device))
    modifier = HairModifier(hair_strands, config["n_cluster"])
    
    scale = pred_param(scale, modifier.clump_rand)
    print(f"regression scale: {scale}")
    with open(os.path.join(config["output_dir"], "regression_scale.txt"), "w") as f:
        f.write(str(scale))
    
    hair_strands_mod = modifier.eval(scale)
    
    save_hair_strands(os.path.join(config["output_dir"], "regression_result.hair"), hair_strands_mod)
    
    
def valid_clumping_param(config):
    # model
    model = get_model(config)

    # reference
    ref_img, _, _, _ = get_references(config)
    ref_feature = normalize(model(ref_img[None], None), dim=-1)[0]
    C, H, W = ref_img.shape

    # load scene
    hair_strands = torch.stack(load_hair(config["hair_path"], device=device))
    cameras = load_cameras(config["camera_path"], device=device)
    head_mesh = load_obj_with_uv(config["head_path"], device=device)
    mesh_zbuf = render_meshes_zbuf(head_mesh, cameras, (W, H))
    curves = Curves(transform_points_to_ndc(cameras, hair_strands))
    modifier = HairModifier(hair_strands, config["n_cluster"])
    
    # search best_param
    best_param, best_param_tensor = search_best_param(
        config, model, ref_feature, H, W, cameras, mesh_zbuf, curves, modifier)

    scale = best_param_tensor
    scale.requires_grad = True
    optimizer = torch.optim.Adam([scale], lr=config["lr"])

    # optimize
    loss_list = []
    param_list = []
    print("start optimization")
    for i in range(config["epoch"]):
        hair_strands_mod = modifier.eval(scale)
        hair_strands_proj = transform_points_to_ndc(cameras, hair_strands_mod)
        curves.update_packed(hair_strands_proj.reshape(-1, 3))
        image_silh, image_depth, image_orien = render_feature_map(
            config["rasterize_hard"], curves, (W, H), mesh_zbuf, modifier.clump_scale)
        
        # loss
        image_feature = torch.cat([image_depth.unsqueeze(0), image_orien.permute(2, 0, 1)], dim=0)
        render_feature = normalize(model(None, image_feature[None]), dim=-1)[0]
        loss = 1 - torch.sum(ref_feature * render_feature)

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            scale.clamp_(0, 1)
        
        # info
        loss_list.append(loss.item())
        param_list.append(scale.mean().item())
        print(f"epoch: {i}/{config['epoch']}, loss: {loss_list[i]}, scale: {param_list[i]}")

    # visualize
    save_hair_strands(os.path.join(config["output_dir"], "optim_result.hair"), hair_strands_mod)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./scripts/base_config.yml')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    config = yaml.load(open(args.conf, "r", encoding="utf-8"), Loader=yaml.FullLoader)

    config["n_cluster"] = 1024
    config["lr"] = 0.01
    config["epoch"] = 200
    config["ckpt_path"] = "../contrastive_learning/runs/May05_02-47-38_ROG_cluster_1024_use_full_map/model_best.pth.tar"
    config["ckpt_regression_path"] = "../contrastive_learning/runs/May07_11-48-03_DESKTOP-C0FF356_cluster_1024_regression/model_best.pth.tar"
    output_dir = "X:/results/clumping_validation/"
    
    for ref_img_name in os.listdir("X:/contrastive_learning/data/xgen_full_render"):
        if ref_img_name not in [
            "DD_0916_03_zhongchang_hair027.exr",
            "DD_0916_03_zhongchang_hair012.exr",
            "DD_0913_01_zhongchang_hair001.exr",
        ]:
            continue
        ref_img_path = os.path.join("X:/contrastive_learning/data/xgen_full_render", ref_img_name)
        hair_name = os.path.splitext(os.path.basename(ref_img_path))[0]
        # if os.path.isfile(os.path.join(output_dir, hair_name, "regression_result_modified_side.png")):
        #     print("skip", hair_name)
        #     continue
        print("processing", hair_name)
        config["ref_img_path"] = ref_img_path
        config["head_path"] = os.path.join("X:/contrastive_learning/data/assets/scalp_models/A_Pose",
                                        hair_name[:10]+".obj")
        config["hair_path"] = glob.glob(os.path.join("X:/contrastive_learning/data/assets", "*/*",
                                        hair_name+"_Wo_Modifiers_resample_32.hair"))[0]
        config["ref_hair_path"] = glob.glob(os.path.join("X:/contrastive_learning/data/assets", "*/*",
                                        hair_name+"_Full_resample_32.hair"))[0]
        config["camera_path"] = os.path.join(os.path.dirname(config["hair_path"]), "camera.json")
        config["output_dir"] = os.path.join(output_dir, hair_name)
        
        regression_param(config)
        valid_clumping_param(config)
        
        render_hair_shading(config["head_path"], config["ref_hair_path"], os.path.join(config["output_dir"], f"reference.png"),
                            img_size=1024, side_view=True, device_idx=args.gpu)
        
        render_hair_shading(config["head_path"], config["hair_path"], os.path.join(config["output_dir"], f"wo_modifiers.png"),
                            camera_path=config["ref_hair_path"].replace(".hair", "_camera.json"),
                            img_size=1024, side_view=True, device_idx=args.gpu)
        
        render_hair_shading(config["head_path"], os.path.join(config["output_dir"], "optim_result.hair"),
                            os.path.join(config["output_dir"], f"optim_result.png"),
                            camera_path=config["ref_hair_path"].replace(".hair", "_camera.json"),
                            img_size=1024, side_view=True, device_idx=args.gpu)
        
        render_hair_shading(config["head_path"], os.path.join(config["output_dir"], "regression_result.hair"),
                            os.path.join(config["output_dir"], f"regression_result.png"),
                            camera_path=config["ref_hair_path"].replace(".hair", "_camera.json"),
                            img_size=1024, side_view=True, device_idx=args.gpu)

    # input_dir = "X:/hairstep/Real_Image"
    # output_dir = "X:/results/guide_clumping_ablation"

    # hair_name = "christopher-campbell-rDEOVtE7vOs-unsplash"
    # config["ref_img_path"] = os.path.join(input_dir, "resized_img", hair_name+".png")
    # config["head_path"] = "X:/hairstep/head_model_metahuman.obj"
    # config["hair_path"] = os.path.join(input_dir, "hair3D/resample_32", hair_name+".hair")
    # config["camera_path"] = os.path.join(input_dir, "param", hair_name+".npy")
    # config["ref_seg_path"] = os.path.join(input_dir, "seg", hair_name+".png")
    # config["ref_orien_path"] = os.path.join(input_dir, "strand_map", hair_name+".png")
    # config["ref_depth_path"] = os.path.join(input_dir, "depth_map", hair_name+".npy")

    # config["output_dir"] = os.path.join(output_dir, hair_name)

    # valid_clumping_param(config)
    # result_hair_path = os.path.join(config["output_dir"], "optim_result.hair")
    # hair_strands = torch.stack(load_hair(result_hair_path)).numpy()
    # render_final_result(config["hair_path"], hair_strands, os.path.join(config["output_dir"], "render"),
    #                         render_origin=False, device_idx=-1)
