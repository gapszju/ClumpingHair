import os
import sys 
import torch
import argparse
from pytorch3d.io import load_hair

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))
from modifiers.hair_modifiers import HairModifier
sys.path.insert(0, os.path.join(cur_dir, "../../contrastive_learning"))
from data_render import visualize_hair, render_final_result, render_projection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    bl_device_idx = torch.cuda.current_device() - torch.cuda.device_count()


    head_path = "X:/contrastive_learning/data/assets/scalp_models/A_Pose/DD_0916_03.obj"
    hair_path = "X:/contrastive_learning/data/assets/DD_woman/DD_0916_03_zhongchang_hair004/DD_0916_03_zhongchang_hair004_Wo_Modifiers_resample_32.hair"
    output_dir = "X:/results/modifiers/DD_0916_03_zhongchang_hair004"

    hair_strands = torch.stack(load_hair(hair_path))
    modifier = HairModifier(hair_strands, n_clusters=1024)
    
    visualize_hair(
        head_path, hair_path, modifier.eval(0.5, add_noise=False, add_cut=False),
        os.path.join(output_dir, "only_clumping.png"),
        img_size=2048, side_view=True, render_origin=True, device_idx=bl_device_idx,
    )
    visualize_hair(
        head_path, hair_path, modifier.eval(0.5, add_noise=True, add_cut=False),
        os.path.join(output_dir, "clumping_noise.png"),
        img_size=2048, side_view=True, render_origin=False, device_idx=bl_device_idx,
    )
    visualize_hair(
        head_path, hair_path, modifier.eval(0.5, add_noise=True, add_cut=True),
        os.path.join(output_dir, "full_modifiers.png"),
        img_size=2048, side_view=True, render_origin=False, device_idx=bl_device_idx,
    )

    # render_final_result(
    #     hair_path, hair_strands,
    #     os.path.join(output_dir, "render"),
    #     render_origin=True, device_idx=bl_device_idx,
    # )