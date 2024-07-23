import os
import sys 
import torch
import shutil
import argparse
import pyexr
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
from src.hair_model import HairModel
from src.utils import save_hair_strands
from src.utils.visualizaton import *


def collage_pipeline_img(hairstep_dir, result_dir, output_dir):
    hair_name = os.path.basename(output_dir)
    print("Processing", hair_name)

    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(os.path.join(hairstep_dir, "resized_img", hair_name+".png"),
                    os.path.join(output_dir, "reference.png"))
    shutil.copyfile(os.path.join(hairstep_dir, "seg", hair_name+".png"),
                    os.path.join(output_dir, "ref_segment.png"))
    shutil.copyfile(os.path.join(hairstep_dir, "strand_map", hair_name+".png"),
                    os.path.join(output_dir, "ref_strand_map.png"))
    shutil.copyfile(os.path.join(hairstep_dir, "depth_map", hair_name+".npy"),
                    os.path.join(output_dir, "ref_depth.npy"))
    
    strand_map = plt.imread(os.path.join(output_dir, "ref_strand_map.png"))[..., ::-1]
    mask = strand_map[..., :2].sum(-1) > 0.1
    strand_map[mask] = 1 - strand_map[mask]
    strand_map[..., 2] = 0.5
    strand_map[..., 2][~mask] = 0
    plt.imsave(os.path.join(output_dir, "ref_orien_map.png"), strand_map)
    plt.imsave(os.path.join(output_dir, "ref_depth_vis.png"),
                np.load(os.path.join(hairstep_dir, "depth_map", hair_name+".npy"))[..., None].repeat(3, axis=-1))
    
    feature_maps = torch.load(os.path.join(result_dir, hair_name, "vis", "iter_0480.pt"), map_location="cpu")
    orien_map, depth_map, silh_map = feature_maps["image_orien"], feature_maps["image_depth"], feature_maps["image_silh"]
    param_map = orien_map[..., -1:].clone().expand(-1,-1,3).clip(0, 1)
    mask = orien_map[..., :2].sum(-1) > 0.1
    orien_map[..., 2][mask] = 0.5
    plt.imsave(os.path.join(output_dir, "res_silhouette.png"), silh_map[..., None].expand(-1,-1,3).numpy())
    plt.imsave(os.path.join(output_dir, "res_depth.png"), depth_map[..., None].expand(-1,-1,3).numpy())
    plt.imsave(os.path.join(output_dir, "res_orien_map.png"), orien_map.numpy())
    plt.imsave(os.path.join(output_dir, "res_param_map.png"), param_map.numpy())


def render_hair_model(hair_model: HairModel, output_dir, camera_path=None, device_idx=0):
    init_scene(device_idx)
    
    # import head
    verts = hair_model.head_mesh.verts_packed().cpu().numpy()
    faces = hair_model.head_mesh.faces_packed().cpu().numpy()
    mesh = bpy.data.meshes.new('mesh')
    mesh.from_pydata(verts, [], faces)
    mesh.shade_smooth()
    mesh.materials.append(get_bsdf_material())
    
    head_obj = bpy.data.objects.new("Head", mesh)
    head_obj.rotation_euler = (np.pi/2, 0, 0)
    bpy.context.collection.objects.link(head_obj)
    
    # import hair
    hair_list = [
        create_hair_from_strands("Original", hair_model.hair_strands.cpu()),
        create_hair_from_strands("Interpolate", hair_model.get_hair_interp().cpu()),
        create_hair_from_strands("Guide", hair_model.guides.cpu(), 5e-4),
        create_hair_from_strands("Modifiers", hair_model.eval(0.5).cpu()),
    ]
    material = get_hair_dataset_material(melanin=0.4, roughness=0.3)
    for hair in hair_list:
        hair.data.materials.append(material)
        hair.hide_render = True
        hair.rotation_euler = (np.pi/2, 0, 0)
    
    hair_list[2].data.materials.clear()
    hair_list[2].data.materials.append(get_bsdf_material((0.246201, 0.147027, 0.072272, 1)))
    # hair_list[2].data.materials.append(get_bsdf_material((0.4, 0.16, 0, 1)))
    params = hair_model.clump_scale.expand_as(hair_model.hair_strands).cpu().numpy()
    set_hair_aov(hair_list[-1].name, params)
    
    # env and camera
    add_envmap(os.path.join(asset_dir, "ENV.Environment.exr"))
    algin_camera_to_objects([hair_list[0]])
    
    if camera_path is not None:
        import_opengl_camera(camera_path)
    
    # render shading
    for hair in hair_list:
        hair.hide_render = False
        render_scene(os.path.join(output_dir, hair.name+".png"), img_size=1024)
        hair.hide_render = True


def render_optim_visualize(head_path, hair_path, vis_dir, output_dir, camera_path=None, device_idx=0):
    init_scene(device_idx)
    build_scene(head_path, hair_path, get_hair_bsdf_material(melanin=0.4, roughness=0.3))
    if camera_path is not None:
        import_opengl_camera(camera_path)

    hair_name = os.path.splitext(os.path.basename(hair_path))[0]
    hair_obj = bpy.data.objects[hair_name]
    hair_obj.rotation_euler = (np.pi/2, 0, 0)

    iter_files = sorted([f for f in os.listdir(vis_dir) if f.startswith("iter_")])
    for idx, iter_file in enumerate(iter_files[::5]+iter_files[-1:]):
        data = torch.load(os.path.join(vis_dir, iter_file), map_location="cpu")
        strands = data["strands_interp"]

        set_hair_strands(hair_name, strands.numpy())
        render_scene(os.path.join(output_dir, "optim", f"optim_{idx:03d}.png"), img_size=1024)


def create_bpy_vis_scene(
    head_path: str,
    hair_model: HairModel,
    vis_dir: str,
    output_dir: str,
    camera_path: str = None,
    device_idx: int = 0,
):
    init_scene(device_idx)

    # head object
    bpy.ops.wm.obj_import(filepath=head_path)
    bpy.ops.object.shade_smooth()
    head_obj = bpy.context.object
    head_obj.data.materials.append(get_bsdf_material())

    # hair
    iter_data = torch.load(os.path.join(vis_dir, "iter_0000.pt"), map_location="cpu")
    hair_list = [
        create_hair_from_strands("Original", hair_model.hair_strands.cpu()),
        create_hair_from_strands("Guide", hair_model.guides.cpu(), 5e-4),
        create_hair_from_strands("Interpolate", hair_model.get_hair_interp().cpu()),
        create_hair_from_strands("Modifiers", iter_data["strands_interp"]),
    ]
    material = get_hair_dataset_material(melanin=0.4, roughness=0.3)
    for hair in hair_list:
        hair.data.materials.append(material)
        hair.rotation_euler = (np.pi/2, 0, 0)
        
    set_hair_aov("Guide", hair_model.guide_colors[:, None, :3].expand_as(hair_model.guides).cpu().numpy())
    set_hair_aov("Interpolate", hair_model.strand_colors[:, None, :3].expand_as(hair_model.hair_strands).cpu().numpy())

    # world & camera
    add_envmap(os.path.join(asset_dir, "ENV.Environment.exr"))
    algin_camera_to_objects(hair_list[-1:])
    if camera_path is not None:
        import_opengl_camera(camera_path)
    
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(output_dir, "parameterization.blend"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    head_path = os.path.join(ROOT_DIR, "assets", "head_smooth.obj")
    hairstep_dir = "X:/hairstep/Real_Image"
    result_dir = "X:/results/reconstruction/hairstep/Real_Image"
    hair_names = [
        # "lance-reis-0sJ6Hwi9MU0-unsplash",
        # "kenzie-kraft-L3UrNhwBCes-unsplash",
        "joao-paulo-de-souza-oliveira-x-FNmzxyQ94-unsplash",
    ]
    
    for hair_name in hair_names:
        output_dir = "X:/results/pipeline_images/"+hair_name
        hair_path = os.path.join(hairstep_dir, "hair3D/resample_32", hair_name+".hair")
        res_hair_path = os.path.join(result_dir, hair_name, "results", "full_modifier.hair")
        vis_dir = os.path.join(result_dir, hair_name, "vis")
        camera_path = res_hair_path.replace(".hair", "_camera.json")
        
        hair_model = HairModel(hair_path, head_path, device="cuda")
        # render_hair_model(hair_model, output_dir, camera_path, device_idx=args.gpu)
        # render_hair_shading(head_path, res_hair_path, os.path.join(output_dir, "Result.png"),
        #                     camera_path, img_size=1024, side_view=False, device_idx=args.gpu)
        
        # collage_pipeline_img(hairstep_dir, result_dir, output_dir)
        
        # render_optim_visualize(head_path, hair_path, vis_dir, output_dir, camera_path, device_idx=args.gpu)
        
        create_bpy_vis_scene(head_path, hair_model, vis_dir, output_dir, camera_path, device_idx=args.gpu)
