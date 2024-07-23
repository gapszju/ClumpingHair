import os
import sys 
import torch
import argparse
from pytorch3d.io import load_hair

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils import (
    HairModifier,
    render_hair_shading,
    render_hair_template,
    save_hair_strands,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    head_path = "X:/contrastive_learning/data/assets/scalp_models/A_Pose/DD_0916_03.obj"
    hair_path = "X:/contrastive_learning/data/assets/DD_woman/DD_0916_03_zhongchang_hair004/DD_0916_03_zhongchang_hair004_Wo_Modifiers_resample_32.hair"
    output_dir = "X:/results/modifiers/DD_0916_03_zhongchang_hair004"

    hair_strands = torch.stack(load_hair(hair_path))
    modifier = HairModifier(hair_strands, n_clusters=1024)
    
    render_hair_shading(
        head_path, hair_path, os.path.join(output_dir, "wo_modifiers.png"),
        img_size=2048, side_view=True, device_idx=args.gpu)
    
    output_path = os.path.join(output_dir, "only_clumping.hair")
    save_hair_strands(output_path, modifier.eval(0.5, add_noise=False, add_cut=False))
    render_hair_shading(
        head_path, output_path, output_path.replace(".hair", ".png"), hair_path.replace(".hair", "_camera.json"),
        img_size=2048, side_view=True, device_idx=args.gpu)
    
    output_path = os.path.join(output_dir, "clumping_noise.hair")
    save_hair_strands(output_path, modifier.eval(0.5, add_noise=True, add_cut=False))
    render_hair_shading(
        head_path, output_path, output_path.replace(".hair", ".png"), hair_path.replace(".hair", "_camera.json"),
        img_size=2048, side_view=True, device_idx=args.gpu)
    
    output_path = os.path.join(output_dir, "full_modifiers.hair")
    save_hair_strands(output_path, modifier.eval(0.5, add_noise=True, add_cut=True))
    render_hair_shading(
        head_path, output_path, output_path.replace(".hair", ".png"), hair_path.replace(".hair", "_camera.json"),
        img_size=2048, side_view=True, device_idx=args.gpu)