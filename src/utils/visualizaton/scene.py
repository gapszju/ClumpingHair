import bpy
import os
import json
import torch
import numpy as np

from mathutils import Matrix, Vector, Quaternion

asset_dir = os.path.realpath(os.path.join(__file__, "../"*4, "assets"))

from .assets import *
from ..hair_utils import read_hair_cy, write_hair_cy, read_hair_data


def get_view3d_area():
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            return area


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


def get_objects_bound_box(objects: list[bpy.types.Object]):
    bpy.context.evaluated_depsgraph_get().update()
    boxes = []
    for obj in objects:
        box = np.array([[bb[:] for bb in obj.bound_box]]).reshape(-1, 3)
        box = box @ obj.matrix_world.to_3x3().transposed() + obj.location
        boxes.append(box)
    
    boxes = np.concatenate(boxes, axis=0)
    bound_min = box.min(axis=0)
    bound_max = box.max(axis=0)
    center = (bound_min + bound_max) / 2
    size = bound_max - bound_min
    
    return Vector(center), Vector(size)


def algin_camera_to_objects(objects: list[bpy.types.Object]):
    """
    Adjusts the camera view to focus on a group of objects.

    Args:
        objects (list[bpy.types.Object]): A list of Blender objects to focus on.

    Returns:
        None
    """
    center, size = get_objects_bound_box(objects)
    
    camera = bpy.context.scene.camera
    dist = max(size[0], size[2]) / 2 / np.tan(camera.data.angle / 2)
    
    camera.location = center + Vector([0, -dist-size[1]/2, 0])
    camera.rotation_euler = (np.pi/2, 0, 0)


def rotate_camera_around(objects: list[bpy.types.Object], angle: float, axis: str = "Z"):
    """
    Rotates the camera around a set of objects by a specified angle.

    Args:
        objects (list[bpy.types.Object]): A list of Blender objects to rotate the camera around.
        angle (float): The angle (in radians) by which to rotate the camera.
        axis (str): The axis around which to rotate the camera. Can be "X", "Y", or "Z".

    Returns:
        None
    """
    center, size = get_objects_bound_box(objects)
    camera = bpy.context.scene.camera
    
    rotation = Matrix.Rotation(angle, 4, axis)
    camera.location = rotation @ (camera.location - center) + center
    camera.rotation_euler.rotate(rotation.to_euler())


def create_hair_from_strands(name, strands, widths=None) -> bpy.types.Object:
    """
    Create a hair object from a list of strands.

    Args:
        name (str): The name of the hair object.
        strands (list): A list of strands, where each strand is a list of points.
        widths (list, optional): A list of widths for each strand. Defaults to None.

    Returns:
        bpy.types.Object: The created hair object.
    """
    print(f"Creating curve object from {len(strands)} strands...")

    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    for i in range(len(strands)):
        pos = np.hstack((strands[i], np.ones((len(strands[i]), 1)))).astype(np.float32)
        
        if widths is None:
            wid = np.ones(len(strands[i]), dtype=np.float32) * 4e-5 * np.random.uniform(0.7, 1.0)
        elif isinstance(widths, float):
            wid = np.ones(len(strands[i]), dtype=np.float32) * widths
        else:
            wid = np.array(widths[i], dtype=np.float32) / 2
            
        s = curve.splines.new("POLY")
        s.points.add(len(pos) - 1)  # s already had one point
        s.points.foreach_set("co", pos.flatten())
        s.points.foreach_set("radius", wid)

    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)

    # convert to hair curve
    single_select(obj)
    bpy.ops.object.convert(target='CURVES', keep_original=False)
    new_object = bpy.context.object

    # clean
    curve.user_clear()
    bpy.data.curves.remove(curve)

    return new_object


def import_hair_obj(
    filepath: str,
    resample: bool = True,
    resample_count: int = 32,
    coord_transform: bool = True,
) -> bpy.types.Object:
    if filepath.endswith(".hair"):
        hair_pos, hair_width = read_hair_cy(filepath, coord_transform=coord_transform)
    elif filepath.endswith(".data"):
        hair_pos = read_hair_data(filepath, coord_transform=coord_transform)
        hair_width = None
    else:
        raise ValueError("Unknown hair file format.")

    hair_name = os.path.splitext(os.path.basename(filepath))[0]
    hair_obj = create_hair_from_strands(hair_name, hair_pos, hair_width)

    if filepath.endswith(".data"):
        # usc-hairsalon, duplicate hair curves
        asset_file = os.path.join(bpy.utils.resource_path("LOCAL"),
                                  "datafiles/assets/geometry_nodes/procedural_hair_node_assets.blend")
        bpy.ops.wm.append(filename="NodeTree/Duplicate Hair Curves", directory=asset_file)
        mod = hair_obj.modifiers.new(type="NODES", name="Duplicate Hair Curves")
        mod.node_group = bpy.data.node_groups["Duplicate Hair Curves"]
        mod["Input_2"] = 7      # amount
        mod["Input_5"] = 0.005  # radius
        bpy.ops.object.modifier_apply(modifier=mod.name)

    if resample:
        mod = hair_obj.modifiers.new(type="NODES", name="Resample Hair")
        mod.node_group = get_hair_resample_node(resample_count)
        single_select(hair_obj, active=True)
        bpy.ops.object.modifier_apply(modifier=mod.name)

    return hair_obj


def export_hair(filepath: str, hair_name: str):
    hair_obj = bpy.data.objects[hair_name]
    num_curves = len(hair_obj.data.curves)
    num_points = len(hair_obj.data.points)
    assert num_points % num_curves == 0, "Curve points are not consistent."
    
    positions = np.zeros(num_points*3, dtype=np.float32)
    widths = np.zeros(num_points, dtype=np.float32)
    hair_obj.data.points.foreach_get("position", positions)
    hair_obj.data.points.foreach_get("radius", widths)
    
    positions = positions.reshape(-1,3) @ Matrix.Rotation(-np.pi/2, 3, 'X').transposed()
    widths = widths * 2
    write_hair_cy(filepath, [num_points//num_curves]*num_curves, positions, widths)


def import_opengl_camera(filepath: str):
    print("Importing camera from", filepath)
    
    def vec3_gl2bl(v: Vector) -> Vector:
        return Vector((v[0], -v[2], v[1]))

    with open(filepath, "r") as fp:
        data = json.load(fp)

    camera = bpy.context.scene.camera
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.type = "PERSP"
    
    # intrinsic
    camera.data.angle_x = np.deg2rad(data["fov"])
    
    # extrinsic
    pos = vec3_gl2bl(data["position"])
    look_at = vec3_gl2bl(data["look_at"])
    up = vec3_gl2bl(data["up"])
    
    z_axis = (pos - look_at).normalized()
    x_axis = up.cross(z_axis).normalized()
    y_axis = z_axis.cross(x_axis)

    rot_mat = Matrix([x_axis, y_axis, z_axis]).transposed()
    
    camera.location = pos
    camera.rotation_euler = rot_mat.to_euler()
    

def export_opengl_camera(filepath: str, cam_obj: bpy.types.Object):
    def get_render_fov(cam_obj: bpy.types.Object) -> tuple:
        cam = cam_obj.data

        fov = cam.angle
        image_width = bpy.context.scene.render.resolution_x
        image_height = bpy.context.scene.render.resolution_y
        
        # fov is x direction
        if (cam.sensor_fit == "AUTO" and image_width > image_height
            or cam.sensor_fit =="HORIZONTAL"):
            tan_fov_y_h = np.tan(0.5 * fov) * image_height / image_width
            return fov, 2 * np.arctan(tan_fov_y_h)
        # fov is y direction
        else:
            tan_fov_x_h = np.tan(0.5 * fov) * image_width / image_height
            return 2 * np.arctan(tan_fov_x_h), fov

    def vec3_bl2gl(v: Vector) -> Vector:
        return Vector((v[0], v[2], -v[1]))

    """Extracts the given camera's parameters in OpenGL convention."""
    if type(cam_obj.data) is not bpy.types.Camera:
        raise TypeError("argument is not a camera")

    rot_90 = Quaternion((1, 0, 0), -0.5 * np.pi)
    rot = cam_obj.matrix_basis.to_quaternion() @ rot_90
    pos = cam_obj.matrix_basis.to_translation()
    
    render = bpy.context.scene.render

    data = {
        "name": cam_obj.name,
        "type": "opengl",
        "unit": "m",
        "resolution": (render.resolution_x, render.resolution_y),
        "fov": np.rad2deg(get_render_fov(cam_obj)[0]),
        "position": tuple(vec3_bl2gl(pos)),
        "look_at": tuple(vec3_bl2gl(pos + rot @ Vector((0, 1, 0)))),
        "up": tuple(vec3_bl2gl(rot @ Vector((0, 0, 1)))),
        "z_near": cam_obj.data.clip_start,
        "z_far": cam_obj.data.clip_end,
    }
    
    with open(filepath, "w") as fp:
        json.dump(data, fp, indent=4)


def get_hair_strands(hair_name: str):
    hair_obj = bpy.data.objects[hair_name]
    num_curves = len(hair_obj.data.curves)
    num_points = len(hair_obj.data.points)
    if num_points % num_curves != 0:
        raise ValueError("Curve points are not consistent.")
    
    positions = np.zeros(num_points*3, dtype=np.float32)
    hair_obj.data.points.foreach_get("position", positions)
    strands = positions.reshape(num_curves, -1, 3)

    return strands


def set_hair_strands(hair_name: str, strands: np.ndarray):
    hair_obj = bpy.data.objects[hair_name]
    positions = np.reshape(strands, -1)
    if len(positions) != len(hair_obj.data.points)*3:
        raise ValueError(f"Strands are not consistent. original: {len(hair_obj.data.points)*3}, new: {len(positions)}.")
    
    hair_obj.data.points.foreach_set("position", strands.flatten())
    hair_obj.data.update_tag()
    hair_obj.update_tag()


def set_hair_material(hair_name: str, melanin: float, redness: float = 0.0, roughness: float = 0.15):
    hair_mat = bpy.data.objects[hair_name].data.materials[0]
    shader_node = next(node for node in hair_mat.node_tree.nodes if node.type == "BSDF_HAIR_PRINCIPLED")
    shader_node.inputs['Melanin'].default_value = melanin
    shader_node.inputs['Melanin Redness'].default_value = redness
    shader_node.inputs['Roughness'].default_value = roughness


def set_hair_aov(hair_name: str, aov: np.ndarray):
    hair_obj = bpy.data.objects[hair_name]
    attr = hair_obj.data.attributes.get("hair_aov")
    if not attr:
        attr = hair_obj.data.attributes.new("hair_aov", type="FLOAT_VECTOR", domain="POINT")
    
    aov = aov.reshape(-1, 3)
    assert len(aov) == len(hair_obj.data.points), "AOV length is not consistent."
    attr.data.foreach_set("vector", aov.reshape(-1))


def set_env_params(angle: tuple = None, strength: float = None):
    node_tree = bpy.context.scene.world.node_tree
    if angle:
        maping_node = next(n for n in node_tree.nodes if n.type == "MAPPING")
        maping_node.inputs["Rotation"].default_value = angle
    if strength:
        background_node = next(n for n in node_tree.nodes if n.type == "BACKGROUND")
        background_node.inputs["Strength"].default_value = strength


def init_scene(device_idx: int = 0):
    bpy.ops.wm.read_factory_settings()
    # render preferences
    bpy.context.preferences.filepaths.save_version = 0
    cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
    cycles_prefs.compute_device_type = "OPTIX"
    cycles_prefs.get_devices()
    for device in cycles_prefs.devices:
        device.use = False
    devices = [device for device in cycles_prefs.devices if device.type == "OPTIX"]
    devices[device_idx].use = True
    
    try:
        bpy.context.scene.view_settings.view_transform = "Standard"
    except:
        pass
    
    bpy.data.meshes.remove(bpy.data.meshes[0])
    bpy.data.lights.remove(bpy.data.lights[0])
    bpy.context.scene.camera.data.lens = 85
    bpy.context.scene.camera.data.ortho_scale = 0.45
    
    # render settings
    render = bpy.context.scene.render
    render.resolution_x = 1024
    render.resolution_y = 1024
    render.filter_size = 0.0
    render.film_transparent = True
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.image_settings.color_depth = "8"
    render.hair_type = "STRAND"
    render.hair_subdiv = 3
    
    cycles = bpy.context.scene.cycles
    cycles.use_adaptive_sampling = False
    cycles.samples = 512
    cycles.device = "GPU"
    bpy.context.view_layer.use_pass_z = True
    bpy.context.scene.cycles_curves.subdivisions = 3


def build_scene(obj_file: str, hair_path: str, hair_material: bpy.types.Material = None):
    # head object
    bpy.ops.wm.obj_import(filepath=obj_file)
    bpy.ops.object.shade_smooth()
    head_obj = bpy.context.object
    head_obj.data.materials.append(get_bsdf_material())
    
    # hair
    hair_obj = import_hair_obj(hair_path)
    mod = hair_obj.modifiers.new(type="NODES", name="Set Hair Data")
    mod.node_group = get_hair_geometry_node()
    if hair_material is not None:
        hair_obj.data.materials.append(hair_material)
    
    # aov
    bpy.context.view_layer.aovs.add()
    
    # world
    add_envmap(os.path.join(asset_dir, "ENV.Environment.exr"))
    
    # camera
    algin_camera_to_objects([hair_obj])


def render_dataset(out_path, img_size=512, with_shading=True, with_data=True):
    render = bpy.context.scene.render
    
    # common settings
    render.resolution_x = img_size
    render.resolution_y = img_size
    render.filter_size = 0.0
    render.film_transparent = True
    render.filepath = out_path
    render.image_settings.color_mode = "RGB"
    render.hair_type = "STRAND"
    render.hair_subdiv = 3
    bpy.context.view_layer.use_pass_z = True
    
    # render hair
    if with_shading:
        render.engine = "CYCLES"
        cycles = bpy.context.scene.cycles
        cycles.use_adaptive_sampling = False
        cycles.samples = 512
        cycles.device = "GPU"
        render.image_settings.file_format = "PNG"
        render.image_settings.color_depth = "16"
        bpy.context.scene.cycles_curves.subdivisions = 3
        bpy.ops.render.render(write_still=True)
    
    # orientation and depth
    if with_data:
        render.engine = "BLENDER_EEVEE"
        render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
        render.image_settings.color_depth = "32"
        bpy.ops.render.render(write_still=True)


def render_scene(out_path, img_size=512, engine="CYCLES"):
    render = bpy.context.scene.render
    
    render.engine = engine
    render.resolution_x = img_size
    render.resolution_y = img_size
    render.filepath = os.path.realpath(out_path)

    bpy.ops.render.render(write_still=True)


def render_hair_shading(
    model_path: str,
    hair_path: str,
    out_path: str,
    camera_path: str = None,
    img_size: int = 1024,
    melanin: float = 0.4,
    side_view: bool = False,
    device_idx: int = 0,
    render_engine: str = "CYCLES",
    save_proj: bool = False,
    animation: bool = False,
):
    torch.cuda.empty_cache()

    init_scene(device_idx)
    build_scene(model_path, hair_path, get_hair_bsdf_material(melanin, 0.3))
    
    if camera_path is None:
        camera_path = hair_path.replace(".hair", "_camera.json")
        if os.path.exists(camera_path):
            import_opengl_camera(camera_path)
        else:
            export_opengl_camera(camera_path, bpy.context.scene.camera)
    else:
        import_opengl_camera(camera_path)
    
    hair_name = os.path.splitext(os.path.basename(hair_path))[0]
    hair_obj = bpy.data.objects[hair_name]
    
    filename, ext = os.path.splitext(os.path.realpath(out_path))

    # render
    if not animation:
        render_scene(filename + "_front" + ext, img_size, render_engine)
        if side_view:
            rotate_camera_around([hair_obj], -np.pi/8)
            render_scene(filename + "_side" + ext, img_size, render_engine)
    else:
        # calculate rotation steps
        angles1 = (np.cos(np.linspace(0, np.pi, 50, endpoint=False)) * 0.5 - 0.5) * np.pi/4
        angles2 = np.cos(np.linspace(np.pi, 0, 70, endpoint=False)) * np.pi/4
        angles3 = -(np.cos(np.linspace(np.pi, 0, 51, endpoint=True)) * 0.5 - 0.5) * np.pi/4
        angles = np.concatenate([[0], angles1, angles2, angles3])
        
        for idx, angle in enumerate(np.diff(angles)):
            rotate_camera_around([hair_obj], angle)
            render_scene(filename + f"_{idx:03d}" + ext, img_size)
    
    if save_proj:
        bpy.ops.wm.save_as_mainfile(filepath=filename+".blend")
        
    bpy.ops.wm.quit_blender()


def render_hair_color(
    model_path: str,
    hair_path: str,
    color: np.ndarray,
    out_path: str,
    camera_path: str = None,
    img_size: int = 512,
    device_idx: int = 0,
    side_view: bool = False,
    render_engine: str = "BLENDER_EEVEE",
    save_proj: bool = False,
):   
    torch.cuda.empty_cache()

    init_scene(device_idx)
    build_scene(model_path, hair_path, get_hair_aov_material())
    
    if camera_path is None:
        camera_path = hair_path.replace(".hair", "_camera.json")
        if os.path.exists(camera_path):
            import_opengl_camera(camera_path)
        else:
            export_opengl_camera(camera_path, bpy.context.scene.camera)
    else:
        import_opengl_camera(camera_path)
    
    hair_name = os.path.splitext(os.path.basename(hair_path))[0]
    hair_obj = bpy.data.objects[hair_name]
    
    # set color
    if isinstance(color, torch.Tensor):
        color = color.detach().cpu().numpy()
    if color.shape[-1] == 4:
        color = color[..., :3]
    set_hair_aov(hair_name, color)

    filename, ext = os.path.splitext(os.path.realpath(out_path))
    render_scene(filename+"_front"+ext, img_size, render_engine)
       
    if side_view:
        rotate_camera_around([hair_obj], -np.pi/8)
        render_scene(filename + "_side" + ext, img_size, render_engine)
    
    if save_proj:
        bpy.ops.wm.save_as_mainfile(filepath=filename+".blend")
    
    bpy.ops.wm.quit_blender()


def load_calib(calib_path, loadSize=1024):
    # loading calibration data
    param = np.load(calib_path, allow_pickle=True)
    # pixel unit / world unit
    ortho_ratio = param.item().get('ortho_ratio')
    # world unit / model unit
    scale = param.item().get('scale')
    # camera center world coordinate
    center = param.item().get('center')
    # model rotation
    R = param.item().get('R')

    translate = -np.matmul(R, center).reshape(3, 1)
    extrinsic = np.concatenate([R, translate], axis=1)
    extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
    # Match camera space to image pixel space
    scale_intrinsic = np.identity(4)
    scale_intrinsic[0, 0] = scale / ortho_ratio
    scale_intrinsic[1, 1] = scale / ortho_ratio
    scale_intrinsic[2, 2] = scale / ortho_ratio
    # Match image pixel space to image uv space
    uv_intrinsic = np.identity(4)
    uv_intrinsic[0, 0] = 1.0 / float(loadSize // 2)
    uv_intrinsic[1, 1] = 1.0 / float(loadSize // 2)
    uv_intrinsic[2, 2] = 1.0 / float(loadSize // 2)
    # Transform under image pixel space
    trans_intrinsic = np.identity(4)
    intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
    calib = np.matmul(intrinsic, extrinsic)

    return calib

def render_hair_projection(
    model_path: str,
    hair_path: str,
    camera_path: str,
    out_path: str,
    img_size: int = 512,
    melanin: float = 0.2,
    device_idx: int = 0,
    render_engine: str = "CYCLES",
):
    init_scene(device_idx)

    # set camera
    camera = bpy.context.scene.camera
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 2
    camera.location = (0, 0, 1)
    camera.rotation_euler = (0, 0, 0)

    camera_calib = load_calib(camera_path)
    R, T = camera_calib[:3, :3], camera_calib[:3, 3]

    # head object
    bpy.ops.wm.obj_import(filepath=model_path)
    head_obj = bpy.context.object
    head_obj.data.materials.append(get_bsdf_material())
    head_obj.matrix_world = Matrix.Identity(4)
    head_obj.is_holdout = True
    
    # hair
    hair_obj = import_hair_obj(hair_path, coord_transform=False)
    hair_obj.data.materials.append(get_hair_bsdf_material(melanin))

    # world
    add_envmap(os.path.join(asset_dir, "ENV.Environment.exr"))

    # transform objects
    head_points = np.empty(len(head_obj.data.vertices)*3)
    head_obj.data.vertices.foreach_get("co", head_points)
    head_points = head_points.reshape(-1, 3)
    head_points = head_points @ R.T + T
    head_obj.data.vertices.foreach_set("co", head_points.ravel())

    hair_points = np.empty(len(hair_obj.data.points)*3)
    hair_obj.data.points.foreach_get("position", hair_points)
    hair_points = hair_points.reshape(-1, 3)
    hair_points = hair_points @ R.T + T
    hair_obj.data.points.foreach_set("position", hair_points.ravel())

    # scale hair radius
    hair_radius = np.empty(len(hair_obj.data.points))
    hair_obj.data.points.foreach_get("radius", hair_radius)
    hair_radius = hair_radius * max([np.linalg.norm(R[i]) for i in range(3)])
    hair_obj.data.points.foreach_set("radius", hair_radius)
    hair_obj.data.update_tag()

    set_env_params(angle=(np.pi/2, 0, 0))

    # render
    render_scene(out_path, img_size, render_engine)
    
    bpy.ops.wm.quit_blender()


def render_hair_template(
    hair_path: str,
    output_path: str,
    camera_path: str = None,
    transform: np.ndarray = None,
    img_size: float = 1024,
    side_view: bool = True,
    animation: bool = False,
    device_idx: int = 0,
):
    template_path = os.path.join(asset_dir, "render_template.blend")
    init_scene(device_idx)
    bpy.ops.wm.open_mainfile(filepath=template_path)

    # import hair
    hair_obj = import_hair_obj(hair_path)
    if transform is not None:
        hair_obj.matrix_world = Matrix(transform)
    hair_obj.data.materials.append(bpy.data.materials["Hair"])
    
    # camera
    algin_camera_to_objects([hair_obj])
    rotate_camera_around([hair_obj], -np.pi/32, axis="X")

    if camera_path is None:
        camera_path = hair_path.replace(".hair", "_camera.json")
        if os.path.exists(camera_path):
            import_opengl_camera(camera_path)
        else:
            export_opengl_camera(camera_path, bpy.context.scene.camera)
    else:
        import_opengl_camera(camera_path)

    filename, ext = os.path.splitext(os.path.realpath(output_path))

    # render
    if not animation:
        render_scene(filename + "_front" + ext, img_size)
        if side_view:
            rotate_camera_around([hair_obj], np.pi/8)
            render_scene(filename + "_side_L" + ext, img_size)
            rotate_camera_around([hair_obj], -np.pi/4)
            render_scene(filename + "_side_R" + ext, img_size)
    else:
        # calculate rotation steps
        angles1 = (np.cos(np.linspace(0, np.pi, 50, endpoint=False)) * 0.5 - 0.5) * np.pi/4
        angles2 = np.cos(np.linspace(np.pi, 0, 70, endpoint=False)) * np.pi/4
        angles3 = -(np.cos(np.linspace(np.pi, 0, 51, endpoint=True)) * 0.5 - 0.5) * np.pi/4
        angles = np.concatenate([[0], angles1, angles2, angles3])
        
        for idx, angle in enumerate(np.diff(angles)):
            rotate_camera_around([hair_obj], angle)
            render_scene(filename + f"_{idx:03d}" + ext, img_size)
    
    bpy.ops.wm.quit_blender()