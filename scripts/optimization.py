import os
import gc
import sys 
import torch
import yaml
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
from src.optim import optimize_strands
from src.utils import render_hair_template, render_hair_projection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./config/config_sample.yml')
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    
    input_dir = args.data
    selected = []
    if os.path.exists(os.path.join(input_dir, "resized_img")):
        selected = os.listdir(os.path.join(input_dir, "resized_img"))
    output_dir = args.output

    with open(args.conf, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["ckpt_path"] = os.path.join(ROOT_DIR, "ckpt", "model_best.pth.tar")
    # config["with_modifier"] = False
    config["loss_outline_weights"] = [1, 1, 0.1, 10000, 50, 1]
    # config["clump_scale"] = 0.55
    # config["epoch"] = 100
    # config["vis_interval"] = 1
       
    
    # input_dir = "X:/hairstep/USC_HairSalon"
    # output_dir = "X:/differential_rendering/full_pipline/May05_02-47-38_ROG_cluster_1024_use_full_map/USC_HairSalon"
    
    # input_dir = "X:/hairstep/HiSa_HiDa_test"
    # output_dir = "X:/results/reconstruction/hairstep/HiSa_HiDa"
        
    # input_dir = "X:/hairstep/DD_Hairs"
    # output_dir = "X:/differential_rendering/full_pipline/May05_02-47-38_ROG_cluster_1024_use_full_map/DD_Hairs"
    
    # for hair_name in os.listdir(os.path.join(input_dir, "resized_img")):
    #     hair_name = os.path.splitext(hair_name)[0]
    
    # input_dir = "X:/hairstep/Man_Image"
    # output_dir = "X:/results/reconstruction/hairstep/Man_Image"
    # with open(os.path.join(input_dir, "selected.json"), "r") as f:
    #     selected = json.load(f)
    # selected = [
    #     # "pexels-tr-n-thanh-hung-1721990-18864058"
    #     # "pexels-kenzero14-19473192",
    #     "pexels-rukiye-agacayak-686493765-20271519",
    # ]
    
    input_dir = "X:/hairstep/Real_Image"
    output_dir = "X:/simulation/"
    selected = [
        "lance-reis-0sJ6Hwi9MU0-unsplash",
        # "kenzie-kraft-L3UrNhwBCes-unsplash",
        # "joao-paulo-de-souza-oliveira-x-FNmzxyQ94-unsplash",
        ]

    for hair_name in selected:
        hair_name = os.path.splitext(hair_name)[0]
    
        config["ref_img_path"] = os.path.join(input_dir, "resized_img", hair_name+".png")
        config["head_path"] = os.path.join(ROOT_DIR, "assets", "head_model_metahuman.obj")
        config["hair_path"] = os.path.join(input_dir, "hair3D", hair_name+".hair")
        config["camera_path"] = os.path.join(input_dir, "param", hair_name+".npy")
        config["ref_seg_path"] = os.path.join(input_dir, "seg", hair_name+".png")
        config["ref_orien_path"] = os.path.join(input_dir, "strand_map", hair_name+".png")
        config["ref_depth_path"] = os.path.join(input_dir, "depth_map", hair_name+".npy")
        # config["hida_pair_path"] = os.path.join(input_dir, "relative_depth/pairs", hair_name+".npy")
        # config["hida_label_path"] = os.path.join(input_dir, "relative_depth/labels", hair_name+".npy")
        config["output_dir"] = os.path.join(output_dir, hair_name)
        
        if os.path.exists(config["ref_img_path"]):
            result_hair_path = os.path.join(config["output_dir"], "results", "full_modifier.hair")
            # if os.path.exists(result_hair_path):
            #     print("skip", hair_name)
            #     continue
            
            optimize_strands(config)
            torch.cuda.empty_cache()
            
            # hair_visualizer.run(os.path.join(config["output_dir"], "vis"))
            
            render_hair_template(config["hair_path"], os.path.join(config["output_dir"], "render", "render_origin.png"), device_idx=args.gpu)
            render_hair_template(result_hair_path, os.path.join(config["output_dir"], "render", "render_modified.png"), device_idx=args.gpu)
            render_hair_projection(
                config["head_path"], config["hair_path"], config["camera_path"],
                os.path.join(config["output_dir"], "projection", "origin.png"), img_size=1024, device_idx=args.gpu,
            )
            render_hair_projection(
                config["head_path"], result_hair_path, config["camera_path"],
                os.path.join(config["output_dir"], "projection", "result.png"), img_size=1024, device_idx=args.gpu,
            )