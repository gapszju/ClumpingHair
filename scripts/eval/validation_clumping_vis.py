import os
import sys 
import torch
import yaml
import glob
import argparse
import matplotlib.pyplot as plt
from validation_clumping import valid_clumping_param

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)

from src.utils import (
    HairModifier,
    render_hair_shading,
    save_hair_strands,
)
from src.utils.visualizaton import *
from src.utils.hair_utils import read_hair_cy, write_hair_cy
from scripts.data_gen.gen_snet_dataset import compose_image


def generate_input_data(head_path, hair_path, output_dir, n_cluster, clump_scale, device_idx):
    hair_name = os.path.splitext(os.path.basename(hair_path))[0]
    
    init_scene(device_idx)
    head_obj, hair_obj = build_scene(head_path, hair_path, get_hair_dataset_material(0.4, 0.3))
    export_opengl_camera(os.path.join(output_dir, "camera.json"), bpy.context.scene.camera)
    
    # update hair strands
    hair_strands = get_hair_strands(hair_name)
    hair_strands = torch.tensor(hair_strands)

    output_path = os.path.join(output_dir, "input")

    # render shading
    modifier = HairModifier(hair_strands, n_cluster)
    modified_hair = modifier.eval(clump_scale).cpu().numpy()
    set_hair_strands(hair_name, modified_hair)
    render_scene(os.path.join(output_dir, "reference.png"), img_size=1024)

    # render data
    head_obj.is_holdout = True
    set_hair_aov(hair_name, modifier.clump_scale.expand_as(hair_strands).cpu().numpy())
    render_dataset(output_path, img_size=1024, with_shading=True, with_data=False)
    render_dataset(output_path, img_size=512, with_shading=False, with_data=True)
    
    # compose
    compose_image(output_path+".png", output_path+".exr")
    
    # save data
    export_hair(os.path.join(output_dir, "reference.hair"), hair_name)
    torch.save(modifier.clump_scale, os.path.join(output_dir, "ref_clump_scale.pt"))


def replace_hair_width(hair1, hair2):
    _, wid = read_hair_cy(hair1, coord_transform=False)
    pos, _ = read_hair_cy(hair2, coord_transform=False)
    num = [len(p) for p in pos]
    write_hair_cy(hair2, num, np.concatenate(pos), np.concatenate(wid))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./config/config_sample.yml')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    config = yaml.load(open(args.conf, "r", encoding="utf-8"), Loader=yaml.FullLoader)
    config["epoch"] = 200
    config["ckpt_path"] = os.path.join(ROOT_DIR, "ckpt", "model_best.pth.tar")
    
    data_dir = "Z:/TZS/output/single-view-hair/data"
    output_root = "Z:/TZS/output/single-view-hair/results/clumping_validation_vis"
    
    for clump_scale in [0.5, 0.3, 0.7]:
        for file in os.listdir(os.path.join(data_dir, "xgen_full_render")):
            hair_name = os.path.splitext(file)[0]
            head_path = os.path.join(data_dir, "assets/scalp_models/A_Pose", hair_name[:10]+".obj")
            hair_path = glob.glob(os.path.join(data_dir, "assets", "*/*",
                                            hair_name+"_Wo_Modifiers_resample_32.hair"))[0]
            output_dir = os.path.join(output_root, f"clump_scale_{clump_scale}", hair_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # if os.path.isfile(os.path.join(output_dir, "optim_clump_vis_front.png")):
            #     print("skip", hair_name)
            #     continue

            # print("processing", hair_name)
            # print("generating input data...")
            # generate_input_data(head_path, hair_path, output_dir, config["n_cluster"], clump_scale, args.gpu)
            
            # # optimize hair
            # print("optimizing hair...")
            # config["ref_img_path"] = os.path.join(output_dir, "input.exr")
            # config["head_path"] = head_path
            # config["hair_path"] = hair_path
            # config["ref_hair_path"] = os.path.join(output_dir, "reference.hair")
            config["camera_path"] = os.path.join(output_dir, "camera.json")
            # config["output_dir"] = os.path.join(output_dir, "optimize")
            # valid_clumping_param(config)
            
            # # render results
            # print("rendering results...")
            # clump_data = torch.load(os.path.join(output_dir, "ref_clump_scale.pt")).cpu().numpy()
            # clump_vis_color = plt.cm.viridis(clump_data.reshape(-1, 1).repeat(config["n_sample"], axis=1))
            # render_hair_color(head_path, config["ref_hair_path"], clump_vis_color**2.2,
            #     os.path.join(output_dir, "ref_clump_vis.png"), config["camera_path"], img_size=1024, device_idx=args.gpu
            # )
            
            render_hair_shading(
                head_path, hair_path, os.path.join(output_dir, "base.png"),
                config["camera_path"], img_size=1024, side_view=False, device_idx=args.gpu
            )
            
            result_hair_path = os.path.join(output_dir, "optimize", "optim_result.hair")
            replace_hair_width(hair_path, result_hair_path)
            render_hair_shading(
                head_path, result_hair_path, os.path.join(output_dir, "optim_result.png"),
                config["camera_path"], img_size=1024, side_view=False, device_idx=args.gpu
            )
            clump_data = torch.load(os.path.join(output_dir, "optimize", "optim_clump_scale.pt")).cpu().numpy()
            clump_vis_color = plt.cm.viridis(clump_data.reshape(-1, 1).repeat(config["n_sample"], axis=1))
            render_hair_color(head_path, result_hair_path, clump_vis_color**2.2,
                os.path.join(output_dir, "optim_clump_vis.png"), config["camera_path"], img_size=1024, device_idx=args.gpu
            )
                
