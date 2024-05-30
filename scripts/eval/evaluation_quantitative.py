import os, sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.evaluation import run_hisa_hida_evaluation, run_synthetic_evaluation


if __name__ == "__main__":
    import yaml
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./config_full.yml')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    with open(args.conf, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    skip_list = [
        "0337cf004e253a2fadaf3c7c51b3b43e",
        "b3d1548d13d867818a53ee36283692a9",
        "cb7ad7e49f359405b39d9293b491d5ca",
    ]

    run_hisa_hida_evaluation(
        config,
        ref_dir="X:/hairstep/HiSa_HiDa",
        hairnet_dir="X:/hairnet/HiSa_HiDa_test",
        neuralhd_dir="X:/neuralhdhair/HiSa_HiDa_test",
        hairstep_dir="X:/hairstep/HiSa_HiDa_test",
        ours_dir="X:/results/reconstruction/hairstep/HiSa_HiDa",
        output_dir="X:/results/evaluation/HiSa_HiDa",
    )

    run_synthetic_evaluation(
        ref_dir="X:/hairstep/DD_Hairs",
        hairnet_dir="X:/hairnet/DD_Hairs",
        neuralhd_dir="X:/neuralhdhair/DD_Hairs",
        hairstep_dir="X:/hairstep/DD_Hairs",
        ours_dir="X:/results/reconstruction/hairstep/DD_Hairs",
        output_dir="X:/results/evaluation/DD_Hairs",
    )
