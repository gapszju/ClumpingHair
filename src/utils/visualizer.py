import os 
import cv2
import json
import uuid
import torch
import tempfile
import subprocess
import shlex
import glob
import numpy as np
import matplotlib.pyplot as plt

import bpy
import bgl
import gpu
from mathutils import Matrix, Vector

from pytorch3d.ops import knn_points
from pytorch3d.io import load_hair, save_hair
import yaml
from .utils import stdout_redirected, as_8bit_img, load_ref_imgs_hairstep


def vec3_bl2gl(v: Vector) -> Vector:
    """Converts a vector from Blender's convention to OpenGL's.

    OpenGL's  X,  Y,  Z are
    Blender's X,  Z, -Y
    """
    return Vector((v[0], v[2], -v[1]))


def vec3_gl2bl(v: Vector) -> Vector:
    """Converts a vector from OpenGL's convention to Blender's.

    Blender's X,  Y,  Z are
    OpenGL's  X, -Z,  Y
    """
    return Vector((v[0], -v[2], v[1]))


def get_view3d_area():
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            return area

    
def import_camera(camera_path):
    with open(camera_path) as f:
        cam_params = json.load(f)
    assert cam_params["type"] == "opengl"
    
    pos = vec3_gl2bl(cam_params["position"])
    look_at = vec3_gl2bl(cam_params["look_at"])
    up = vec3_gl2bl(cam_params["up"])
    
    z_axis = (pos - look_at).normalized()
    x_axis = up.cross(z_axis).normalized()
    y_axis = z_axis.cross(x_axis)
    rot_mat = Matrix([x_axis, y_axis, z_axis]).transposed()
    
    bpy.ops.object.camera_add()
    cam_obj = bpy.context.object
    cam_obj.matrix_world = Matrix.LocRotScale(pos, rot_mat, None)
    cam_obj.data.angle_x = np.radians(cam_params["fov"])
    
    return cam_obj


def single_select(obj: bpy.types.Object = None, name: str = None, active=True):
    """Makes the given object the only selected one in the scene.

    If `active` is True, also makes the object the active one in the scene.
    """
    bpy.ops.object.select_all(action="DESELECT")
    if not obj:
        obj = bpy.data.objects[name]
    obj.select_set(True)

    if active:
        bpy.context.view_layer.objects.active = obj


def create_curves_from_strands(name, strands) -> bpy.types.Object:
    """
    Create a curve object from a list of strands.

    Args:
        name (str): The name of the curve object.
        strands (list): A list of strands, where each strand is a list of 3D points.

    Returns:
        bpy.types.Object: The created curve object.
    """
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    
    for strand in strands:
        strand = np.array(strand) @ Matrix.Rotation(np.pi/2, 3, "X").transposed()
        strand = np.hstack((strand, np.ones((strand.shape[0], 1)))).astype(np.float32)
        wid = np.ones(strand.shape[0], dtype=np.float32) * 0.0001
        
        s = curve.splines.new("POLY")
        s.points.add(strand.shape[0] - 1)  # s already had one point
        s.points.foreach_set("co", strand.flatten())
        s.points.foreach_set("radius", wid)

    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    
    # convert to curves
    single_select(obj)
    bpy.ops.object.convert(target="CURVES")
    curve.user_clear()
    bpy.data.curves.remove(curve)
    
    # render settings
    bpy.context.scene.render.hair_type = "STRAND"
    bpy.context.scene.render.hair_subdiv = 3
    
    return obj


def update_curves_points(curve_obj: bpy.types.Object, points: list):
    """Updates the points of the given curve object.

    Args:
        hair_curve: curve object to be updated.
        points: list of 3D point coordinates (tuples of 3 floats).
    """
    points = points @ Matrix.Rotation(np.pi/2, 3, "X").transposed()
    for idx in range(len(points)):
        curve_obj.data.points[idx].position = points[idx]   


def render_blender_viewport_gpu(width, height):
    offscreen = gpu.types.GPUOffScreen(width, height)

    context = bpy.context
    scene = context.scene

    view_matrix = scene.camera.matrix_world.inverted()

    projection_matrix = scene.camera.calc_matrix_camera(
        context.evaluated_depsgraph_get(), x=width, y=height)

    area = get_view3d_area()
    with bpy.context.temp_override(area=area, region=area.regions[-1]):
        offscreen.draw_view3d(
            scene,
            context.view_layer,
            context.space_data,
            context.region,
            view_matrix,
            projection_matrix,
            do_color_management=True)

    fbo = gpu.types.GPUFrameBuffer(color_slots=(offscreen.texture_color,))
    
    buffer_np = np.empty(width * height * 4, dtype=np.float32)
    buffer = bgl.Buffer(bgl.GL_FLOAT, buffer_np.shape, buffer_np)
    with fbo.bind():
        bgl.glReadBuffer(bgl.GL_BACK)
        bgl.glReadPixels(0, 0, width, height, bgl.GL_RGBA, bgl.GL_FLOAT, buffer)

    return buffer_np.reshape(height, width, 4)[::-1, :, :]


def render_blender_viewport(width, height, engine="BLENDER_EEVEE"):
    render = bpy.context.scene.render
    render.engine = "BLENDER_EEVEE"
    render.resolution_x = width
    render.resolution_y = height
    render.resolution_percentage = 100
    render.filepath = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()) + ".png")
    
    with stdout_redirected():
        bpy.ops.render.render(write_still=True)
    image = cv2.imread(render.filepath)[..., ::-1]
    os.remove(render.filepath)
    
    return image


class HairVisualizer:
    def __init__(self, mesh_paths: list, camera_paths: list, width=512, height=512):
        self.hairs = dict()
        self.meshes = []
        self.cameras = []
        
        self.width = width
        self.height = height
        self.engine = "BLENDER_EEVEE"
        
        bpy.ops.wm.read_factory_settings()
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        bpy.context.preferences.filepaths.save_version = 0
        
        # render settings
        render = bpy.context.scene.render
        render.film_transparent = True
        render.image_settings.color_mode = "RGB"
        bpy.context.scene.eevee.use_gtao = True
        try:
            bpy.context.scene.view_settings.view_transform = "Standard"
        except:
            pass
        
        # head obj
        for mesh_path in mesh_paths:
            if os.path.isfile(mesh_path):
                with stdout_redirected():
                    bpy.ops.wm.obj_import(filepath=mesh_path)
                self.meshes += [bpy.context.object]
        
        # camera
        for camera_path in camera_paths:
            if os.path.isfile(camera_path):
                self.cameras += [import_camera(camera_path)]
        
        # world
        node_tree = bpy.context.scene.world.node_tree
        
        environment_texture_node = node_tree.nodes.new('ShaderNodeTexEnvironment')
        background_node = next(n for n in node_tree.nodes if n.type == 'BACKGROUND')
        
        hdr_dir = bpy.utils.system_resource("DATAFILES", path="studiolights/world")
        hdr_path = os.path.join(hdr_dir, "forest.exr")
        environment_texture_node.image = bpy.data.images.load(hdr_path)
        
        node_tree.links.new(environment_texture_node.outputs['Color'], background_node.inputs['Color'])
        
        # material
        self.hair_material = bpy.data.materials.new("hair_mat")
        self.hair_material.use_nodes = True
        node_tree = self.hair_material.node_tree
        
        attribute_node = node_tree.nodes.new('ShaderNodeAttribute')
        bsdf_node = next(n for n in node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
        
        attribute_node.attribute_name = "color"
        
        node_tree.links.new(attribute_node.outputs['Color'], bsdf_node.inputs['Base Color'])

    def add_hair(self, name, strands, colors=None):
        if isinstance(strands, torch.Tensor):
            strands = strands.detach().cpu().numpy()
        
        self.hairs[name] = create_curves_from_strands(name, strands)
        curves = self.hairs[name].data
        
        if not curves.attributes.get("color"):
            curves.attributes.new("color", type="FLOAT_COLOR", domain="POINT")
        
        if colors is not None:
            if isinstance(colors, torch.Tensor):
                colors = colors.detach().cpu().numpy()
            assert colors.shape[-1] == 4
            
            # convert spline domain to point domain
            if colors.ndim == 2 and colors.shape[0] == strands.shape[0]:
                colors = colors[:, None, :].repeat(strands.shape[1], axis=1)
            
            curves.attributes['color'].data.foreach_set("color", colors.reshape(-1))
        
        self.hairs[name].data.materials.append(self.hair_material)
    
    def hide_all_hairs(self):
        for obj in self.hairs.values():
            obj.hide_render = True
    
    def update_hair_points(self, name, points):
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        
        points = points.reshape(-1, 3)
        update_curves_points(self.hairs[name], points)
        
    def update_hair_colors(self, name, colors):
        if isinstance(colors, torch.Tensor):
            colors = colors.detach().cpu().numpy()
        assert colors.shape[-1] == 4
        
        curves = self.hairs[name].data
        curves.attributes['color'].data.foreach_set("color", colors.reshape(-1))
        
    def render(self, name, width=None, height=None, engine=None):
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        if engine is None:
            engine = self.engine
        
        results = []
        for camera in self.cameras:
                self.hide_all_hairs()
                self.hairs[name].hide_render = False
                bpy.context.scene.camera = camera
                results += [render_blender_viewport(width, height, engine)]

        return results
    
    def save(self, out_path):
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=out_path, relative_remap=False)
    

def point_cloud_distance(points1, points2):
    dist1, _, _ = knn_points(points1, points2)
    dist2, _, _ = knn_points(points2, points1)
    return dist1, dist2


def run(data_path):
    out_filename = os.path.realpath(data_path)
    print(f"Visualizing:", out_filename)
    with open(os.path.join(data_path, "config.yml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    const_data = torch.load(os.path.join(data_path, "constants.pt"), map_location="cpu")
    
    # ref images
    ref_img = cv2.imread(config["ref_img_path"])[..., ::-1]
    ref_img_orien, ref_img_silh, ref_img_depth = load_ref_imgs_hairstep(
        config["ref_seg_path"], config["ref_orien_path"], config["ref_depth_path"], device="cpu")
    ref_img_mask = ref_img_orien[..., :2].sum(-1) > 0
    H, W, _ = ref_img_orien.shape
    
    # blender visualizer
    hair_vis = HairVisualizer(
        [config["head_path"]], [config["camera_front_path"], config["camera_side_path"]], W, H
    )
    hair_vis.add_hair("guide", const_data["guides"], const_data["guide_colors"])
    hair_vis.add_hair("hair", const_data["strands_interp"], const_data["strand_colors"])

    # video writer
    cmd = 'ffmpeg -y -s {}x{} -pixel_format rgb24 -f rawvideo -r 10 -i pipe: -vcodec libx265 -pix_fmt yuv420p -crf 19 {}'
    video_sp = subprocess.Popen(
        shlex.split(cmd.format(W * 4, H * 3, out_filename.replace("\\", "/") + ".mp4")),
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    
    is_first = True
    for file in sorted(glob.glob(os.path.join(data_path, "iter_*.pt"))):
        print(f"Processing: {os.path.basename(file)}")
        iter_data = torch.load(file, map_location="cpu")
        image_orien, image_silh, image_depth = (
            iter_data["image_orien"], iter_data["image_silh"], iter_data["image_depth"]
        )
        image_mask = image_orien[..., :2].sum(-1) > 0
        image_mask = image_mask & ref_img_mask
        
        # 2D image visualization
        image_orien_diff = (image_orien - ref_img_orien).abs()
        image_orien_diff = image_orien_diff[..., :2].norm(dim=-1) * image_mask
        image_orien_diff = as_8bit_img(plt.cm.jet(image_orien_diff.numpy())[..., :3])
        image_orien = as_8bit_img(image_orien)
        
        image_depth_diff = (image_depth - ref_img_depth).abs() * image_mask
        image_depth_diff = as_8bit_img(plt.cm.jet(image_depth_diff.numpy())[..., :3])
        image_depth = as_8bit_img(image_depth)[..., None].repeat(3, axis=-1)
        
        image_silh = as_8bit_img(image_silh)
        image_silh_diff = np.stack(
            [as_8bit_img(ref_img_silh), image_silh, np.zeros_like(image_silh)], axis=-1)
        image_silh = image_silh[..., None].repeat(3, axis=-1)
        
        result_img_silh = np.concatenate([image_silh, image_silh_diff], axis=1)
        result_img_orien = np.concatenate([image_orien, image_orien_diff], axis=1)
        result_img_depth = np.concatenate([image_depth, image_depth_diff], axis=1)
        
        # rendering visualization
        hair_vis.update_hair_points("guide", iter_data["guides"])
        hair_vis.update_hair_points("hair", iter_data["strands_interp"])

        guide_render = hair_vis.render("guide")
        hair_render = hair_vis.render("hair")
        
        # write result to video
        row0 = np.concatenate([ref_img, as_8bit_img(ref_img_orien), result_img_silh], axis=1)
        row1 = np.concatenate([result_img_orien, result_img_depth], axis=1)
        row2 = np.concatenate([guide_render[0], hair_render[0], guide_render[1], hair_render[1]], axis=1)
        result_img = np.concatenate([row0, row1, row2], axis=0)
        video_sp.stdin.write(result_img.tobytes())
        
        if is_first:
            plt.imsave(out_filename + "_init.png", result_img)
            is_first = False

    # save final result
    plt.imsave(out_filename + ".png", result_img)
    hair_vis.save(out_filename + ".blend")
    # strands = iter_data["strands_interp"]
    # save_hair(out_filename + ".hair", [strands.shape[1]]*strands.shape[0], strands.reshape(-1, 3))
    video_sp.stdin.close()
    video_sp.wait()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hair Visualizer")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to data directory",
    )
    args = parser.parse_args()

    run(args.data_path)