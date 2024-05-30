import os
import sys
import torch
import yaml
import argparse
import pyexr
import numpy as np

from torch.nn.functional import normalize, mse_loss, l1_loss
from matplotlib import pyplot as plt

from pytorch3d.structures import Curves
from pytorch3d.io import load_hair, save_hair

device = torch.device("cuda")
cur_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(cur_dir, "../"))
from modifiers.hair_modifiers import HairModifier
from utils import *

sys.path.insert(0, os.path.join(cur_dir, "../../contrastive_learning"))
from simclr import ResNetSimCLR


def calc_ref_features(config, model):
    ref_features = []
    ref_params = []
    for scale in [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]:
        config["ref_img_path"] = os.path.join(config["ref_img_dir"], f"cluster_{n_cluster}_scale_{scale:.2f}.exr")
        
        # reference image
        ref_img_origin = pyexr.read(config["ref_img_path"])[..., :3]
        ref_img = (
            0.299 * ref_img_origin[..., 0]
            + 0.587 * ref_img_origin[..., 1]
            + 0.114 * ref_img_origin[..., 2]
        )
        ref_img = torch.tensor(ref_img).unsqueeze(0).to(device)
        
        ref_feature = normalize(model(ref_img[None], None), dim=-1)[0]
        ref_features.append(ref_feature)
        
        param_img = pyexr.read(config["ref_img_path"], channels="aov")
        ref_scale = param_img.sum() / (np.count_nonzero(param_img) + 1e-8)
        ref_params.append(ref_scale)
    
    return torch.stack(ref_features), torch.tensor(ref_params, dtype=torch.float32).to(device)


def render_clumping_features(config, model):
    W, H, C = pyexr.read(config["ref_img_path"]).shape
    
    # load scene
    cameras = load_cameras(config["camera_path"], device=device)
    head_mesh = load_obj_with_uv(config["head_path"], device=device)
    mesh_zbuf = render_meshes_zbuf(head_mesh, cameras, (W, H))
    hair_strands = resample_strands_fast(load_hair(config["hair_path"]), 32).to(device)
    curves = Curves(hair_strands)

    # modifiers
    modifier = HairModifier(hair_strands, n_clusters=n_cluster)

    # search for global minimum
    render_features = []
    render_params = []
    for scale in np.linspace(0.0, 1.0, 101):
        hair_strands_mod = modifier.eval(scale)
        hair_strands_proj = transform_points_to_ndc(cameras, hair_strands_mod)
        curves.update_packed(hair_strands_proj.reshape(-1, 3))
        image_silh, image_depth, image_orien = render_feature_map(
            config["rasterize_hard"], curves, (W, H), mesh_zbuf, modifier.clump_scale)
        
        image_feature = torch.cat([image_depth.unsqueeze(0), image_orien.permute(2, 0, 1)], dim=0)
        render_feature = normalize(model(None, image_feature[None, config["feature_channels"]]), dim=-1)[0]
        
        param_img = image_orien[..., 2]
        render_scale = param_img.sum() / param_img.count_nonzero().clip(1e-8)
        
        render_features.append(render_feature)
        render_params.append(render_scale)
    
    return torch.stack(render_features), torch.stack(render_params)


def eval_snet_error(config):
    hair_name = os.path.basename(config["ref_img_dir"])
    print(f"Processing {hair_name}...")
    
    # model
    ckpt = torch.load(config["ckpt_path"], map_location=device)["state_dict"]
    in_channels = [ckpt["conv1.weight"].shape[1], ckpt["conv2.weight"].shape[1]]
    model = ResNetSimCLR(base_model="resnet18", out_dim=128, in_channels=in_channels)
    model.load_state_dict(ckpt)
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()
    
    ref_features, ref_params = calc_ref_features(config, model)
    render_features, render_params = render_clumping_features(config, model)
    
    similarity = ref_features @ render_features.T
    pred_params = render_params[similarity.argmax(dim=1)]
    dists = pred_params - ref_params
    
    mean, std = dists.abs().mean(), dists.std()
    print(f"Mean:{mean:.5f}, Std:{std:.5f}")
        
    # visualize
    plt.plot(dists.cpu().numpy())
    plt.xlabel(f"Mean:{mean:.5f}, Std:{std:.5f}")
    os.makedirs(config["output_dir"], exist_ok=True)
    filename = os.path.basename(os.path.dirname(config["ckpt_path"]))
    plt.savefig(os.path.join(config["output_dir"], filename))
    plt.close()
    
    return mean, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./scripts/base_config.yml')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    config = yaml.load(open(args.conf, "r", encoding="utf-8"), Loader=yaml.FullLoader)

    n_cluster = 1024
    input_dir = "X:/hairstep/HiSa_HiDa"
    data_dir = "X:/contrastive_learning/data/clumping_dataset/real_img_cluster_512/test"
    output_dir = "X:/results/snet_ablation"
    # hair_name_list = [
    #     "0d285f5be7fa09c3dbbf1c9334047888",
    #     "29dc6387017754215b532130750e2370",
    #     "97460f1480c7c477919a68ebb1a660ba",
    #     "cc5bf02bc6205f878cde02df9d00e7b9",
    # ]
    hair_name_list = os.listdir(data_dir)

    ckpt_path_list = [
        # "../contrastive_learning/runs/Apr20_16-40-31_ROG_with_optim_dist2/model_best.pth.tar", # 使用所有特征图
        # "../contrastive_learning/runs/Apr25_11-22-59_ROG_use_depth_param/model_best.pth.tar", # 使用 depth + param
        # "../contrastive_learning/runs/Apr27_21-02-57_ROG_use_only_param/model_best.pth.tar", # 仅使用参数
        "../contrastive_learning/runs/May05_02-47-38_ROG_cluster_1024_use_full_map/model_best.pth.tar", # 使用所有特征图
        # "../contrastive_learning/runs/May07_00-36-02_ROG_cluster_1024_use_depth_param/model_best.pth.tar", # 使用 depth + param
        # "../contrastive_learning/runs/May06_15-14-24_DESKTOP-FLDT8US_cluster_1024_use_only_param/model_best.pth.tar", # 仅使用参数
    ]
    feature_channels_list = [
        [0, 1, 2, 3],
        # [0, 3],
        # [3],
    ]

    results = []
    for ckpt_path, feature_channels in zip(ckpt_path_list, feature_channels_list):
        for hair_name in hair_name_list:
            config["n_cluster"] = n_cluster
            config["ckpt_path"] = ckpt_path
            config["feature_channels"] = feature_channels
            config["ref_img_dir"] = os.path.join(data_dir, hair_name)
            config["hair_path"] = os.path.join(input_dir, "hair3D/resample_32", hair_name+".hair")
            config["camera_path"] = os.path.join(input_dir, "hair3D/camera", hair_name+".json")
            config["output_dir"] = os.path.join(output_dir, hair_name)

            if os.path.isfile(config["hair_path"]):
                mean, std = eval_snet_error(config)
                results.append(mean.item())
    
    result_info = f"\nGlobal Mean:{np.mean(results):.5f}, Global Std:{np.std(results):.5f}"
    print(result_info)
    
    # visualize
    plt.plot(results)
    plt.xlabel(result_info)
    filename = os.path.basename(os.path.dirname(config["ckpt_path"]))
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()