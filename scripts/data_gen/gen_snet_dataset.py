import os, sys
import bpy
import pyexr
import torch
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils.visualizaton import *
from src.utils.modifiers import HairModifier


def compose_image(render_img_path, data_img_path):
    image = Image.open(render_img_path)
    with pyexr.open(data_img_path) as hair_data:
        tangent = hair_data.get("ViewLayer.Combined")[..., :3]
        depth = hair_data.get("ViewLayer.Depth.Z")
    
    mask = tangent[..., :2].sum(axis=-1)>0.1
    depth[~mask] = 0
    depth_min = depth[depth>0].min()
    depth_max = depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min)
    depth_norm = 1 - depth_norm
    depth_norm[depth_norm>1] = 0
    
    aov = tangent[..., 2].copy()
    tangent[..., 2][mask] = 0.5
    
    # resize
    if image.height != depth.shape[0]:
        image = image.resize((depth.shape[1], depth.shape[0]), resample=PIL.Image.LANCZOS)
    image = np.array(image)[..., :3] / 255
    
    data = {'default': image, 'orientation': tangent, 'depth': depth_norm, "aov": aov}
    pyexr.write(data_img_path, data)
    os.remove(render_img_path)


def run(hair_path, model_path, output_dir, device_idx=-1):
    hair_name = os.path.splitext(os.path.basename(hair_path))[0]
    hair_dir = os.path.dirname(hair_path)

    # gen data
    init_scene(device_idx)
    build_scene(model_path, hair_path, get_hair_dataset_material())
    export_opengl_camera( os.path.join(hair_dir, "camera", hair_name+".json"), bpy.context.scene.camera)
    export_hair(os.path.join(hair_dir, "resample_32", hair_name+".hair"), hair_name)
    
    # update hair strands
    hair_strands = get_hair_strands(hair_name)
    hair_strands = torch.tensor(hair_strands).cuda()
    
    for n_cluster in [1024]:
        for scale in [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]:
            output_path = os.path.join(output_dir, hair_name, f"cluster_{n_cluster}_scale_{scale:.2f}")

            # random light and material
            set_env_params(angle=(0, 0, np.random.rand()*2*np.pi))
            set_hair_material(hair_name, (np.random.rand()*0.85)**2+0.15, 0, np.random.rand()*0.2+0.1)

            # render shading
            modified_hair = HairModifier(hair_strands, n_cluster).eval(scale).cpu().numpy()
            set_hair_strands(hair_name, modified_hair)
            render_dataset(output_path, img_size=1024, with_shading=True, with_data=False)

            # render data
            modifier = HairModifier(hair_strands, n_cluster)
            modified_hair = modifier.eval(scale).cpu().numpy()
            set_hair_strands(hair_name, modified_hair)
            set_hair_aov(hair_name, modifier.clump_scale.expand_as(hair_strands).cpu().numpy())
            render_dataset(output_path, img_size=512, with_shading=False, with_data=True)
            
            # compose
            compose_image(output_path+".png", output_path+".exr")


if __name__ == "__main__":
    import fasteners
    import socket
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    hair_dir = "X:/hairstep/HiSa_HiDa/hair3D"
    model_path = "X:/hairstep/head_model_metahuman.obj"
    output_dir = os.path.join(asset_dir, "clumping_dataset", "real_img")
    os.makedirs(output_dir, exist_ok=True)

    with open("X:/hairstep/HiSa_HiDa/split_test.json") as f:
        test_names = json.load(f)
    test_names = [os.path.splitext(name)[0] for name in test_names]

    progress_file = os.path.join(output_dir, "progress.json")
    lock = fasteners.InterProcessLock(os.path.join(output_dir, "file.lock"))
    if not os.path.exists(progress_file):
        with open(progress_file, "w") as fp:
            json.dump({}, fp)

    for hair_name in os.listdir(hair_dir):
        if not hair_name.endswith(".hair"):
            continue

        # check progress
        with lock:
            with open(progress_file, "r") as fp:
                data = json.load(fp)
            progress = data.get(hair_name)
            device_name = socket.gethostname() + f"_{args.gpu}"
            if progress is None or progress == device_name:
                print("\n-=== Rendering", hair_name)
                data[hair_name] = device_name
                with open(progress_file, "w") as fp:
                    json.dump(data, fp, indent=4)
            else:
                print("\n-=== Skip", hair_name)
                continue

        # start render
        hair_path = os.path.join(hair_dir, hair_name)
        output_subdir = (
            os.path.join(output_dir, "test")
            if os.path.splitext(hair_name)[0] in test_names
            else os.path.join(output_dir, "train")
        )
        run(hair_path, model_path, output_subdir, device_idx=args.gpu)

        # done
        with lock:
            with open(progress_file, "r") as fp:
                data = json.load(fp)
            data[hair_name] = "done"
            with open(progress_file, "w") as fp:
                json.dump(data, fp, indent=4)

    exit()
