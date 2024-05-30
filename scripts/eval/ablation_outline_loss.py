import os
import sys 
import torch
import yaml
import argparse
import matplotlib.pyplot as plt
from pytorch3d.io import load_hair

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))
from optim import run as full_pipline

sys.path.insert(0, os.path.join(cur_dir, "../contrastive_learning"))
from data_render import visualize_hair, render_final_result, render_projection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./config_full.yml')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    bl_device_idx = torch.cuda.current_device() - torch.cuda.device_count()

    with open(args.conf, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config["ckpt_path"] = "../contrastive_learning/runs/May05_02-47-38_ROG_cluster_1024_use_full_map/model_best.pth.tar"
    
    input_dir = "X:/hairstep/Real_Image"
    output_dir = "X:/results/outline_loss_ablation/"
    
    # hair_name = "joao-paulo-de-souza-oliveira-x-FNmzxyQ94-unsplash"
    # hair_name = "rosa-rafael-vFy6fja-B5M-unsplash"
    # hair_name = "lance-reis-0sJ6Hwi9MU0-unsplash"
    hair_name = "christina-wocintechchat-com-0Zx1bDv5BNY-unsplash"
    # hair_name = "max-felner-6u0xv4j6WKI-unsplash"
    # hair_name = "patrick-malleret-p-v1DBkTrgo-unsplash" 
    
    for weights, label in zip(
            [[1, 1, 0.1, 1e4, 10, 1]],
            ["add_root_constraint"]            
    ):
        config["ref_img_path"] = os.path.join(input_dir, "resized_img", hair_name+".png")
        config["head_path"] = "../HairStep/data/head_model_metahuman.obj"
        config["hair_path"] = os.path.join(input_dir, "hair3D/resample_32", hair_name+".hair")
        config["camera_path"] = os.path.join(input_dir, "param", hair_name+".npy")
        config["ref_seg_path"] = os.path.join(input_dir, "seg", hair_name+".png")
        config["ref_orien_path"] = os.path.join(input_dir, "strand_map", hair_name+".png")
        config["ref_depth_path"] = os.path.join(input_dir, "depth_map", hair_name+".npy")
        
        config["loss_outline_weights"] = weights
        config["output_dir"] = os.path.join(output_dir, hair_name, label)
        
        # full_pipline(config)
            
        # hair_visualizer.run(os.path.join(config["output_dir"], "vis"))
        
        result_hair_path = os.path.join(config["output_dir"], "results", "full_modifier.hair")
        hair_strands = torch.stack(load_hair(result_hair_path)).numpy()
        render_final_result(config["hair_path"], hair_strands, os.path.join(config["output_dir"], "render"),
                            render_origin=False, device_idx=bl_device_idx)
        # render_projection(
        #     config["head_path"], config["hair_path"], config["camera_path"],
        #     os.path.join(config["output_dir"], "projection", "origin.png"), img_size=1024,
        #     device_idx=bl_device_idx,
        # )
        # render_projection(
        #     config["head_path"], result_hair_path, config["camera_path"],
        #     os.path.join(config["output_dir"], "projection", "result.png"), img_size=1024,
        #     device_idx=bl_device_idx,
        # )

