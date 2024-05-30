import os, sys
import bpy
import pyexr
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils.visualizaton import *


def run(hair_path, model_path, output_dir, resolution, device_idx):
    hair_name = os.path.splitext(os.path.basename(hair_path))[0]
    
    # prepare scene
    init_scene(device_idx)
    build_scene(model_path, hair_path, get_hair_dataset_material())
    set_hair_material(hair_name, 0.2)

    camera = bpy.context.scene.camera
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 0.7243
    
    camera.location = (-0.007, -1, 1.58612)
    camera.rotation_euler = (np.pi/2, 0, 0)
    
    hair_obj = bpy.data.objects[hair_name]
    hair_obj.location = (-0.00076, -0.00151, 0.09074)
    hair_obj.scale = [0.945]*3
    
    # render
    render_dataset(os.path.join(output_dir, hair_name, "image.png"),
                     img_size=resolution*2, with_shading=True, with_data=False, compose=False)
    render_dataset(os.path.join(output_dir, hair_name, "Ori.exr"),
                     img_size=resolution, with_shading=False, with_data=True, compose=False)

    # extract data img
    with pyexr.open(os.path.join(output_dir, hair_name, "Ori.exr")) as data:
        tangent = data.get("ViewLayer.Combined")[..., :3]
    tangent[..., 2] = 0
    image = Image.open(os.path.join(output_dir, hair_name, "image.png")).convert("RGB")
    image = image.resize((resolution, resolution), resample=Image.LANCZOS)
    mask = np.linalg.norm(tangent, axis=-1, keepdims=True) > 0.1
    
    os.makedirs(os.path.join(output_dir, hair_name), exist_ok=True)
    image.save(os.path.join(output_dir, hair_name, "image.jpg"))
    plt.imsave(os.path.join(output_dir, hair_name, "Ori.png"), tangent.clip(0, 1))
    plt.imsave(os.path.join(output_dir, hair_name, "mask.png"), mask.astype(np.float32).repeat(3, axis=-1))
    os.remove(os.path.join(output_dir, hair_name, "Ori.exr"))


if __name__ == "__main__":
    import glob
    resolution = 512
    device_idx = 0
    
    hair_dir = "X:/hairstep/DD_Hairs/hair_gt"
    output_dir = "X:/neuralhdhair/DD_Hairs"

    for filepath in glob.glob(os.path.join(hair_dir, "*.hair")):
        model_path = "X:/neuralhdhair/Bust.obj"
        run(filepath, model_path, output_dir, resolution, device_idx)
    