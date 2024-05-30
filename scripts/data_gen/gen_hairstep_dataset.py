import os, sys
import bpy
import pyexr
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils.visualizaton import *


def export_camera(out_path):
    camera = bpy.context.scene.camera
    matrix_world = Matrix.Rotation(-np.pi/2, 4, 'X') @ camera.matrix_world
    R = matrix_world.to_3x3().transposed()
    T = matrix_world.to_translation()
        
    param = {
            "ortho_ratio": camera.data.ortho_scale,
            "scale": np.array([resolution * 2], dtype=np.float32),
            "center": np.array(T, dtype=np.float32),
            "R": np.array(R, dtype=np.float32),
        }
    np.save(out_path, param, allow_pickle=True)
    

def run(hair_path, model_path, output_dir, resolution, device_idx):
    hair_name = os.path.splitext(os.path.basename(hair_path))[0]
    
    # prepare scene
    init_scene(device_idx)
    build_scene(model_path, hair_path, get_hair_dataset_material())
    set_hair_material(hair_name, 0.2)

    camera = bpy.context.scene.camera
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 0.42 * -camera.location[1]

    # export hair and camera
    os.makedirs(os.path.join(os.path.dirname(hair_path), "resample_32"), exist_ok=True)
    export_hair(os.path.join(os.path.dirname(hair_path), "resample_32", hair_name+".hair"), hair_name)
    
    os.makedirs(os.path.join(output_dir, "param"), exist_ok=True)
    export_camera(os.path.join(output_dir, "param", hair_name+".npy"))

    # render
    render_dataset(os.path.join(output_dir, "img", hair_name+".png"), img_size=resolution, with_data=True)

    # extract data img
    with pyexr.open(os.path.join(output_dir, "img", hair_name+".exr")) as data:
        tangent = data.get("ViewLayer.Combined")
        depth = data.get("ViewLayer.Depth.Z")[..., 0]

    hair_mask = tangent[..., :2].sum(axis=-1) > 0.1

    strand_map = np.zeros(tangent.shape[:2]+(3,), dtype=np.float32)
    strand_map[..., :2][hair_mask] = 1 - tangent[..., :2][hair_mask]
    strand_map[..., 2] = tangent[..., 3] * 0.5
    strand_map[..., 2][hair_mask] = 1

    depth[~hair_mask] = 0
    depth_min = depth[depth>0].min()
    depth_max = depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min)
    depth_norm = 1 - depth_norm
    depth_norm[depth_norm>1] = 0

    depth_norm_vis = plt.cm.jet(depth_norm)[..., :3]
    depth_norm_vis[~hair_mask] = 0

    # save data
    for d in ["seg", "strand_map", "depth_map", "depth_vis_map"]:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)

    hair_mask_img = np.stack([hair_mask]*3, axis=-1).astype(np.float32)
    plt.imsave(os.path.join(output_dir, "seg", hair_name+".png"), hair_mask_img)
    plt.imsave(os.path.join(output_dir, "strand_map", hair_name+".png"), strand_map[..., ::-1].clip(0,1))
    np.save(os.path.join(output_dir, "depth_map", hair_name+".npy"), depth_norm)
    plt.imsave(os.path.join(output_dir, "depth_vis_map", hair_name+".png"), depth_norm_vis)

    os.remove(os.path.join(output_dir, "img", hair_name+".exr"))


if __name__ == "__main__":
    resolution = 512
    device_idx = 0
    
    hair_dir = "X:/hairstep/DD_Hairs/hair3D"
    output_dir = "X:/hairstep/DD_Hairs/result"
    model_path = "X:/hairstep/head_model_metahuman.obj"

    for filepath in glob.glob(os.path.join(hair_dir, "*.hair")):
        run(filepath, model_path, output_dir, resolution, device_idx)
    