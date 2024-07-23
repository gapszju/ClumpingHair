import os
import sys 
import torch
import shutil
import argparse
import pyexr
import numpy as np
import matplotlib.pyplot as plt
from fast_pytorch_kmeans import KMeans

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
from src.hair_model import HairModel
from src.utils import save_hair_strands
from src.utils.visualizaton import *


def calc_hair_root_ng():
    ng = bpy.data.node_groups.new("Calculate Root Position", "GeometryNodeTree")
    if bpy.app.version < (4, 0, 0):
        ng.inputs.new("NodeSocketGeometry", "Geometry")
        ng.inputs.new("NodeSocketObject", "Head")
        ng.outputs.new("NodeSocketGeometry", "Geometry")
    else:
        ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        ng.interface.new_socket("Head", in_out="INPUT", socket_type="NodeSocketObject")
        ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    
    nodes = ng.nodes
    links = ng.links
    
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")
    
    attr_surface_uv = nodes.new("GeometryNodeInputNamedAttribute")
    attr_surface_uv.data_type = "FLOAT_VECTOR"
    attr_surface_uv.inputs["Name"].default_value = "surface_uv_coordinate"
    
    attr_uv_map = nodes.new("GeometryNodeInputNamedAttribute")
    attr_uv_map.data_type = "FLOAT_VECTOR"
    attr_uv_map.inputs["Name"].default_value = "UVMap"
    
    sample_uv_surface = nodes.new("GeometryNodeSampleUVSurface")
    sample_uv_surface.data_type = "FLOAT_VECTOR"
    
    object_info = nodes.new("GeometryNodeObjectInfo")
    position = nodes.new("GeometryNodeInputPosition")
    add = nodes.new("ShaderNodeVectorMath")
    set_position = nodes.new("GeometryNodeSetPosition")
    
    links.new(object_info.outputs["Geometry"], sample_uv_surface.inputs["Mesh"])
    links.new(attr_uv_map.outputs["Attribute"], sample_uv_surface.inputs["Source UV Map"])
    links.new(attr_surface_uv.outputs["Attribute"], sample_uv_surface.inputs["Sample UV"])
    links.new(position.outputs["Position"], sample_uv_surface.inputs[3])
    
    links.new(sample_uv_surface.outputs[2], add.inputs[1])
    links.new(group_input.outputs["Geometry"], set_position.inputs["Geometry"])
    links.new(set_position.outputs["Geometry"], group_output.inputs["Geometry"])
    links.new(position.outputs["Position"], add.inputs["Vector"])
    links.new(add.outputs["Vector"], set_position.inputs["Position"])
    links.new(group_input.outputs["Head"], object_info.inputs["Object"])

    return ng


def extract_hairs(hair_model_path, output_dir):
    hair_model = HairModel.load(hair_model_path)
    hair_model.guides.requires_grad = False
    hair_model.clump_scale.requires_grad = False

    kmeans = KMeans(n_clusters=512, mode="euclidean", max_iter=100, verbose=0)
    kmeans.fit_predict(hair_model.hair_strands.reshape(hair_model.num_strands, -1))
    clusters = kmeans.centroids.reshape(512, -1, 3)
    valid = [cluster.sum()>0 for cluster in clusters]
    clusters = clusters[valid]

    save_hair_strands(os.path.join(output_dir, "hairstep_clusters.hair"), clusters)
    save_hair_strands(os.path.join(output_dir, "hairstep_guide.hair"), hair_model.guides_init)
    save_hair_strands(os.path.join(output_dir, "hairstep_strands.hair"), hair_model.hair_strands)
    save_hair_strands(os.path.join(output_dir, "ours_guide.hair"), hair_model.guides)
    save_hair_strands(os.path.join(output_dir, "ours_strands.hair"), hair_model.eval(hair_model.clump_scale))
    np.save(os.path.join(output_dir, "interp_weights.npy"), hair_model.guide_weights.cpu().numpy())
    np.save(os.path.join(output_dir, "interp_index.npy"), hair_model.guide_indices.cpu().numpy())


def render_hair_simulation(
    head_cache_path,
    hair_model_path,
    guide_cache_path,
    output_dir,
    camera_path = None,
    use_clumping=True,
    device_idx=0,
):
    template_path = os.path.join(asset_dir, "render_template.blend")
    init_scene(device_idx)
    bpy.ops.wm.open_mainfile(filepath=template_path)

    # head
    head_obj = bpy.data.objects["head_smooth"]
    head_cache = np.load(head_cache_path)
    head_obj.data.vertices.foreach_set("co", head_cache[0].reshape(-1))
    head_obj.modifiers.clear()
    head_obj.rotation_euler = (np.pi/2, 0, 0)

    # import hair
    hair_model = HairModel.load(hair_model_path)
    hair_model.guides.requires_grad = False
    hair_model.clump_scale.requires_grad = False

    hair_obj = create_hair_from_strands("Hair", hair_model.eval().cpu().numpy())
    hair_obj.data.materials.append(bpy.data.materials["Hair"])
    hair_obj.rotation_euler = (np.pi/2, 0, 0)

    # calc root position
    hair_obj.data.surface = head_obj
    hair_obj.data.surface_uv_map = "UVMap"
    bpy.context.view_layer.objects.active = hair_obj
    bpy.ops.object.select_all(action="DESELECT")
    hair_obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.curves.snap_curves_to_surface(attach_mode="NEAREST")
    bpy.ops.object.mode_set(mode="OBJECT")

    mod = hair_obj.modifiers.new(name="Calc Hair Root", type="NODES")
    mod.node_group = calc_hair_root_ng() 
    mod["Input_1"] = head_obj

    # camera
    hair_eval = hair_model.eval(hair_model.clump_scale).cpu().numpy()
    hair_eval_local = hair_eval - hair_eval[:, 0:1]
    set_hair_strands(hair_obj.name, hair_eval_local)
    algin_camera_to_objects([hair_obj])
    
    if camera_path is not None:
        import_opengl_camera(camera_path)

    guides_cache = torch.tensor(np.load(guide_cache_path)).to(hair_model.device)
    for i in range(guides_cache.shape[0]):
        hair_model.guides = guides_cache[i].reshape(-1, hair_model.n_sample, 3)

        if use_clumping:
            hair_eval = hair_model.eval(hair_model.clump_scale).cpu().numpy()
            hair_eval_local = hair_eval - hair_eval[:, 0:1]
            set_hair_strands(hair_obj.name, hair_eval_local)
            head_obj.data.vertices.foreach_set("co", head_cache[i].reshape(-1))
            render_scene(os.path.join(output_dir, "render", "ours", f"ours_{i:03d}.png"), img_size=1024)
        else:
            hair_eval = hair_model.eval().cpu().numpy()
            hair_eval_local = hair_eval - hair_eval[:, 0:1]
            set_hair_strands(hair_obj.name, hair_eval_local)
            head_obj.data.vertices.foreach_set("co", head_cache[i].reshape(-1))
            render_scene(os.path.join(output_dir, "render", "hairstep", f"hairstep_{i:03d}.png"), img_size=1024)


if __name__ == "__main__":
    # hair_name = "lance-reis-0sJ6Hwi9MU0-unsplash"
    hair_name = "joao-paulo-de-souza-oliveira-x-FNmzxyQ94-unsplash"
    
    hair_model_path = f"X:/simulation/{hair_name}/results/hair_model.pkl"
    output_dir = f"X:/simulation/{hair_name}/simulation"
    head_cache_path = os.path.join(output_dir, "head_cache.npy")
    hairstep_guide_cache_path = os.path.join(output_dir, "hairstep_guide_cache.npy")
    ours_guide_cache_path = os.path.join(output_dir, "ours_guide_cache.npy")
    camera_path = os.path.join(output_dir, "camera.json")

    os.makedirs(output_dir, exist_ok=True)
    extract_hairs(hair_model_path, output_dir)
    # render_hair_simulation(head_cache_path, hair_model_path, ours_guide_cache_path, output_dir, camera_path, use_clumping=True)
    # render_hair_simulation(head_cache_path, hair_model_path, hairstep_guide_cache_path, output_dir, camera_path, use_clumping=False)
