import os
import sys 
import torch
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.io import load_hair, load_obj
from pytorch3d.structures import Curves

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
from src.utils import *


def soft_rasterize(head_path, hair_path, camera_path):
    config = {    
        "blur_radius": 2,
        "lines_per_pixel": 128,
        "bin_size": None,
        "sigma": 0.00001,
        "gamma": 0.00001
    }
    W, H = 512, 512
    device = torch.device("cuda")
    
    hair_strands = torch.stack(load_hair(hair_path, device=device))
    cameras = load_cameras(camera_path, device=device)
    strands_proj = transform_points_to_ndc(cameras, hair_strands)
    head_mesh = load_obj_with_uv(head_path, device=device)
    mesh_zbuf = render_meshes_zbuf(head_mesh, cameras, (W, H))
    curves = Curves(strands_proj)
    image_silh, image_depth, image_orien = render_feature_map(config, curves, (W, H), mesh_zbuf)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_silh.cpu().numpy())
    axes[1].imshow(image_depth.cpu().numpy())
    axes[2].imshow(image_orien.cpu().numpy())
    plt.show()
    
    return image_silh, image_depth, image_orien


def quad_rasterize(head_path, hair_path, camera_path):
    config = {    
        "blur_radius": 2,
        "lines_per_pixel": 128,
        "bin_size": None,
        "sigma": 0.00001,
        "gamma": 0.00001
    }
    W, H = 512, 512
    device = torch.device("cuda")

    verts, faces, _ = load_obj(head_path, device=device)
    strands = torch.stack(load_hair(hair_path)).to(device)
    cameras = load_cameras(camera_path, device=device)
    
    rasterizer = QuadRasterizer(
        render_size=H,
        head_mesh=(verts, faces.verts_idx),
        quad_w=1e-3,
        faces_per_pixel=config["lines_per_pixel"],
        blur_radius=(config["blur_radius"] / 512) ** 2,
        sigma=config["sigma"],
        gamma=config["gamma"],
    ).to(device)
    
    image = rasterizer(strands, cameras)[0].permute(1,2,0)
    image_silh = image[..., 2]
    
    image_depth = image[..., 3]
    image_depth = image_depth * image_silh.detach()
    zbuf_min = image_depth.min().item()
    zbuf_range = image_depth.max().item() - zbuf_min
    image_depth = (image_depth - zbuf_min) / zbuf_range
    image_depth = 1 - image_depth
    image_depth = image_depth * image_silh.detach()
    
    image_orien = image[..., :3] * 0.5 + 0.5
    image_orien[..., 2] = 0.5
    image_orien = image_orien * image_silh.detach().unsqueeze(-1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_silh.cpu().numpy()), axes[0].set_title("Silhouette")
    axes[1].imshow(image_depth.cpu().numpy()), axes[1].set_title("Depth")
    axes[2].imshow(image_orien.cpu().numpy()), axes[2].set_title("Orientation")
    plt.show()

    return image_silh, image_depth, image_orien


def render_hairstep(args):
    input_dir = "X:/hairstep/Real_Image"
    output_dir = "X:/results/reconstruction/hairstep/Real_Image"
    # with open(os.path.join(input_dir, "selected.json"), "r") as f:
    #     selected = json.load(f)
    selected = [
        # "joao-paulo-de-souza-oliveira-x-FNmzxyQ94-unsplash",
        # "halil-ibrahim-cetinkaya-WzGC8xSyqfg-unsplash",
        # "midas-hofstra-tidSLv-UaNs-unsplash",
        # "behrouz-sasani-5cDg40slYoc-unsplash",
        
        # "hosein-sediqi-sBkSyfPzakI-unsplash",
        # "kate-townsend-3YwfRKDiC-8-unsplash",
        # "janko-ferlic-GWFffQS5eWU-unsplash",
        # "lance-reis-0sJ6Hwi9MU0-unsplash",
        
        # "aiony-haust-rhVeNHHNbdk-unsplash",
        # "jurica-koletic-7YVZYZeITc8-unsplash",
        # "antonio-friedemann-HKp_HmxUrf8-unsplash",
        # "nickolas-nikolic-87d56FlCOyI-unsplash",
        
        # "4",
        # "22951248a9eedf048adc582ccb04a058",
        "ann-agterberg-WqATKbXqGZQ-unsplash",
    ]
    
    for hair_name in selected:
        ref_img_path = os.path.join(input_dir, "resized_img", hair_name+".png")
        head_path = os.path.join(ROOT_DIR, "assets", "head_model_metahuman.obj")
        hair_path = os.path.join(input_dir, "hair3D/resample_32", hair_name+".hair")
        camera_path = os.path.join(input_dir, "param", hair_name+".npy")
        
        output_subdir = os.path.join(output_dir, hair_name)
        
        if os.path.exists(ref_img_path):
            result_hair_path = os.path.join(output_subdir, "results", "full_modifier.hair")
            # if os.path.exists(result_hair_path):
            #     print("skip", hair_name)
            #     continue

            # render_hair_template(hair_path, os.path.join(output_subdir, "render", "render_origin.png"), device_idx=args.gpu)
            # render_hair_template(result_hair_path, os.path.join(output_subdir, "render", "render_modified.png"), device_idx=args.gpu)
            # render_hair_projection(
            #     head_path, hair_path, camera_path,
            #     os.path.join(output_subdir, "projection", "origin.png"), img_size=1024, device_idx=args.gpu,
            # )
            # render_hair_projection(
            #     head_path, result_hair_path, camera_path,
            #     os.path.join(output_subdir, "projection", "result.png"), img_size=1024, device_idx=args.gpu,
            # )
    
            render_hair_template(result_hair_path, os.path.join(output_subdir, "render_ani", "modified.png"),
                                 animation=True, device_idx=args.gpu)
            render_hair_template(hair_path, os.path.join(output_subdir, "render_ani", "origin.png"),
                                 camera_path=result_hair_path.replace(".hair", "_camera.json"), animation=True, device_idx=args.gpu)


def render_neuralhdhair(args):
    input_dir = "X:/neuralHDhair/Real_Image"
    transform_matrix = np.array((
        (1.058, 0.0, 0.0, 0.0008),
        (0.0, 1.058, 0.0, 0.0016),
        (0.0, 0.0, 1.058, -0.096),
        (0.0, 0.0, 0.0, 1.0)
    ))
    hair_list = os.listdir(input_dir)
    hair_list = [
        # "halil-ibrahim-cetinkaya-WzGC8xSyqfg-unsplash",
        # "midas-hofstra-tidSLv-UaNs-unsplash",
        # "behrouz-sasani-5cDg40slYoc-unsplash",
        # "hosein-sediqi-sBkSyfPzakI-unsplash",
        # "kate-townsend-3YwfRKDiC-8-unsplash",
        "janko-ferlic-GWFffQS5eWU-unsplash",
    ]
    
    for hair_name in hair_list:
        data_dir = os.path.join(input_dir, hair_name)
        head_path = "X:/neuralHDhair/Bust.obj"
        hair_path = os.path.join(data_dir, "hair_cy.hair")
        camera_path = os.path.join(data_dir, "camera.npy")

        output_dir = os.path.join(input_dir, hair_name, "render")

        # render_hair_template(hair_path, os.path.join(output_dir, "render_origin.png"),
        #                      transform=transform_matrix, device_idx=args.gpu)
        # render_hair_projection(
        #     head_path, hair_path, camera_path,
        #     os.path.join(output_dir, "projection.png"), img_size=1024,
        #     device_idx=args.gpu,
        # )
        render_hair_template(hair_path, os.path.join(output_dir, "animation", "render.png"),
                                animation=True, device_idx=args.gpu)


def render_hairnet(args):
    input_dir = "X:/hairnet/Real_Image"
    hairstep_dir = "X:/hairstep/Real_Image"
    hair_dir = os.path.join(input_dir, "hair3D/resample_32")
    hair_list = [name[:-len(".hair")] for name in os.listdir(hair_dir) if name.endswith(".hair")]
    hair_list = [
        # "halil-ibrahim-cetinkaya-WzGC8xSyqfg-unsplash",
        # "midas-hofstra-tidSLv-UaNs-unsplash",
        # "behrouz-sasani-5cDg40slYoc-unsplash",
        # "hosein-sediqi-sBkSyfPzakI-unsplash",
        # "kate-townsend-3YwfRKDiC-8-unsplash",
        "janko-ferlic-GWFffQS5eWU-unsplash",
    ]
    
    for hair_name in hair_list:
        head_path = os.path.join(ROOT_DIR, "assets", "head_model_metahuman.obj")
        hair_path = os.path.join(hair_dir, hair_name+".hair")
        camera_path = os.path.join(hairstep_dir, "param", hair_name+".npy")

        # render_hair_template(hair_path, os.path.join(input_dir, "render", hair_name),
        #                      side_view=False, device_idx=args.gpu)
        # render_hair_projection(
        #     head_path, hair_path, camera_path,
        #     os.path.join(input_dir, "projection", hair_name), img_size=1024,
        #     device_idx=args.gpu,
        # )
        render_hair_template(hair_path, os.path.join(input_dir, "animation", hair_name, "render.png"),
                            animation=True, device_idx=args.gpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    # render_hairstep(args)
    # # render_neuralhdhair(args)
    # # render_hairnet(args)
    # exit()
    
    head_path = "X:/hairstep/head_model_metahuman.obj"
    hair_path = "X:/hairstep/my_data/hair3D/resample_32/IMG_9934.hair"
    result_hair_path = "C:/Users/tangz/Desktop/research/code/hair_modeling/src/hair_reconstraction/output/IMG_9934/results/full_modifier.hair"
    camera_path = "X:/hairstep/my_data/param/IMG_9934.npy"
    output_dir = "X:/results/reconstruction/hairstep/HiSa_HiDa/4"

    soft_rasterize(head_path, result_hair_path, camera_path)
    # quad_rasterize(head_path, result_hair_path, camera_path)

    # render_hair_shading(
    #     head_path, hair_path, os.path.join(output_dir, "render_vis", "vis.png"),
    #     img_size=1024, side_view=True, device_idx=args.gpu,
    # )

    # render_hair_template(
    #     hair_path, os.path.join(output_dir, "render", "render.png"), img_size=1024, device_idx=args.gpu,
    # )

    # render_hair_projection(
    #     head_path, hair_path, camera_path,
    #     os.path.join(output_dir, "projection", "origin.png"),
    #     img_size=1024, device_idx=args.gpu,
    # )
    # render_hair_projection(
    #     head_path, result_hair_path, camera_path,
    #     os.path.join(output_dir, "projection", "result.png"),
    #     img_size=1024, device_idx=args.gpu,
    # )
    
    # input_dir = "X:/results/guide_clumping_ablation/christopher-campbell-rDEOVtE7vOs-unsplash"
    # for hair_name in ["original", "wo_modifier", "wo_guide_optim", "full_modifier"]:
    #     hair_path = os.path.join(input_dir, hair_name+".hair")
    #     camera_path = os.path.join(input_dir, "full_modifier_camera.json")
    #     output_path = os.path.join(input_dir, "render_ani", hair_name, "render.png")
    #     render_hair_template(hair_path, output_path, camera_path, animation=True, device_idx=args.gpu)
    
    
    ### Clumping validation ### 
    # import glob
    # input_dir = "X:/results/clumping_validation"
    # data_dir = "X:/contrastive_learning/data/assets"
    # for hair_name in [
    #     "DD_0916_03_zhongchang_hair027",
    #     "DD_0916_03_zhongchang_hair012",
    #     "DD_0913_01_zhongchang_hair001",
    # ]:
    #     origin_hair_path = glob.glob(os.path.join(data_dir, "*/*", hair_name+"_Wo_Modifiers_resample_32.hair"))[0]
    #     ref_hair_path = glob.glob(os.path.join(data_dir, "*/*", hair_name+"_Full_resample_32.hair"))[0]
    #     optim_hair_path = os.path.join(input_dir, hair_name, "optim_result.hair")
    #     regress_hair_path = os.path.join(input_dir, hair_name, "regression_result.hair")
        
    #     head_path = os.path.join(data_dir, "scalp_models/A_Pose", hair_name[:10]+".obj")
    #     camera_path = ref_hair_path.replace(".hair", "_camera.json")
    #     output_dir = os.path.join(input_dir, hair_name, "render_ani")
        
    #     render_hair_shading(head_path, ref_hair_path, os.path.join(output_dir, "reference.png"),
    #                          animation=True, device_idx=args.gpu)
    #     render_hair_shading(head_path, origin_hair_path, os.path.join(output_dir, "original.png"),
    #                          camera_path=camera_path, animation=True, device_idx=args.gpu)
    #     render_hair_shading(head_path, optim_hair_path, os.path.join(output_dir, "optimize.png"),
    #                          camera_path=camera_path, animation=True, device_idx=args.gpu)
    #     render_hair_shading(head_path, regress_hair_path, os.path.join(output_dir, "regression.png"),
    #                          camera_path=camera_path, animation=True, device_idx=args.gpu)
    
    
    # ### Guide and Clumping ###
    # head_path = "X:/hairstep/head_model_metahuman.obj"
    # hair_dir = "X:/results/guide_clumping_ablation/christopher-campbell-rDEOVtE7vOs-unsplash"
    # camera_path = "X:/hairstep/Real_Image/param/christopher-campbell-rDEOVtE7vOs-unsplash.npy"

    # for hair_name in ["original", "wo_guide_optim", "wo_modifier", "full_modifier"]:
    #     render_hair_projection(
    #         head_path, os.path.join(hair_dir, hair_name+".hair"), camera_path,
    #         os.path.join(hair_dir, "projection", hair_name+".png"), img_size=1024, device_idx=args.gpu,
    #     )