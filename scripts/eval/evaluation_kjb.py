import os, sys
import torch
import numpy as np
from pytorch3d.io import load_hair

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.evaluation import compare_image_metric
from src.utils import resample_strands_fast

device = torch.device("cuda")

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
  
    ref_dir = "X:/hairstep/HiSa_HiDa"
    hairstep_dir = "X:/hairstep/HiSa_HiDa_test"
    ours_dir = "X:/results/reconstruction/hairstep/HiSa_HiDa"
    output_dir = "X:/results/evaluation/HiSa_HiDa_neuralhd"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "statistic"), exist_ok=True)
    
    hair_names = []
    init_metric_list = []
    result_metric_list = []
    for hair_name in os.listdir(ours_dir):
        hair_names.append(hair_name)
        init_hair_path = os.path.join(hairstep_dir, "hair3D/resample_32", hair_name+".hair")
        result_hair_path = os.path.join(ours_dir, hair_name, "results/full_modifier.hair")
        
        if not os.path.exists(result_hair_path):
            continue
        print("\nProcessing", hair_name)
        
        config["ref_img_path"] = os.path.join(hairstep_dir, "resized_img", hair_name+".png")
        config["camera_path"] = os.path.join(hairstep_dir, "param", hair_name+".npy")
        config["head_path"] = "X:/hairstep/head_model_metahuman.obj"
        config["ref_seg_path"] = os.path.join(ref_dir, "seg", hair_name+".png")
        config["ref_orien_path"] = os.path.join(ref_dir, "strand_map", hair_name+".png")
        config["ref_depth_path"] = os.path.join(ref_dir, "depth_map", hair_name+".npy")
        config["hida_pair_path"] = os.path.join(ref_dir, "relative_depth/pairs", hair_name+".npy")
        config["hida_label_path"] = os.path.join(ref_dir, "relative_depth/labels", hair_name+".npy")
        config["output_dir"] = output_dir
        
        # load hair
        init_strands = resample_strands_fast(load_hair(init_hair_path), 32).to(device)
        result_strands = resample_strands_fast(load_hair(result_hair_path), 32).to(device)
        
        init_hisa, init_hida, result_hisa, result_hida = compare_image_metric(config, init_strands, result_strands)
        init_metric_list.append([init_hisa, init_hida])
        result_metric_list.append([result_hisa, result_hida])
        
        print("init\t", f"hisa: {init_hisa:.4f}, hida: {init_hida:.4f}")
        print("result\t", f"hisa: {result_hisa:.4f}, hida: {result_hida:.4f}")
        
        # save metrics
        init_metrics = np.array(init_metric_list, dtype=np.float32)
        result_metrics = np.array(result_metric_list, dtype=np.float32)
        os.makedirs(output_dir, exist_ok=True)
        
        for metrics, label in zip([init_metrics, result_metrics], ["Hairstep", "Ours"]):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            for i, title in enumerate(["HiSa", "HiDa"]):
                axes[i].hist(metrics[:, i], bins=20)
                axes[i].set_title(title), axes[i].set_xlabel(f"Mean: {np.mean(metrics[:, i])}")
            fig.savefig(os.path.join(output_dir, "statistic", f"{label}_metrics.png"))
            plt.close(fig)
    
    header = "Name, Hisa, Hida(%)"
    hair_names_str = np.array(hair_names).reshape(-1, 1)
    init_metrics[:, 1] *= 100
    init_metrics_str = np.concatenate((hair_names_str, np.char.mod('%.2f', init_metrics)), axis=1)
    result_metrics[:, 1] *= 100
    result_metrics_str = np.concatenate((hair_names_str, np.char.mod('%.2f', result_metrics)), axis=1)
    np.savetxt(os.path.join(output_dir, "statistic", "metrics_hairstep.csv"),
               init_metrics_str,delimiter=',', header=header, fmt='%s', comments='')
    np.savetxt(os.path.join(output_dir, "statistic", "metrics_ours.csv"),
               result_metrics_str, delimiter=',', header=header, fmt='%s', comments='')