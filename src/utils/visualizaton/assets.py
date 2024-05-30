import bpy
import numpy as np


def get_bsdf_material():
    material = bpy.data.materials.new("Material")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    
    # ceate nodes
    principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled_node.inputs['Base Color'].default_value = (0.4, 0.4, 0.4, 1.0)
    principled_node.inputs['Specular'].default_value = 0
    material_output_node = nodes.new(type="ShaderNodeOutputMaterial")

    # create links
    links.new(principled_node.outputs["BSDF"], material_output_node.inputs["Surface"])
    
    return material


def get_hair_bsdf_material():
    material = bpy.data.materials.new("Hair")
    material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # remove default nodes
    for node in nodes:
        nodes.remove(node)
    
    # ceate nodes
    principled_node = nodes.new(type="ShaderNodeBsdfHairPrincipled")
    principled_node.parametrization = "MELANIN"
    principled_node.inputs['Melanin'].default_value = 0.99
    principled_node.inputs['Melanin Redness'].default_value = 0
    principled_node.inputs['Roughness'].default_value = 0.15
    principled_node.inputs['Radial Roughness'].default_value = 0.3
    principled_node.inputs['Offset'].default_value = np.radians(-2)
    if bpy.app.version >= (4, 0, 0):
        principled_node.model = "HUANG"
        
    hairbsdf_node = nodes.new(type="ShaderNodeBsdfHair")
    
    cycles_output_node = nodes.new(type="ShaderNodeOutputMaterial")
    cycles_output_node.target = "CYCLES"
    
    eevee_output_node = nodes.new(type="ShaderNodeOutputMaterial")
    eevee_output_node.target = "EEVEE"
    
    # create links
    links.new(principled_node.outputs["BSDF"], cycles_output_node.inputs["Surface"])
    links.new(hairbsdf_node.outputs["BSDF"], eevee_output_node.inputs["Surface"])
    
    return material

def get_hair_aov_material():
    mat = bpy.data.materials.new("Hair_vis")
    mat.use_nodes = True
    node_tree = mat.node_tree
    
    attribute_node = node_tree.nodes.new("ShaderNodeAttribute")
    attribute_node.attribute_name = "hair_aov"
    output_node = next(n for n in node_tree.nodes if n.type == "OUTPUT_MATERIAL")
    node_tree.links.new(attribute_node.outputs["Color"], output_node.inputs["Surface"])

    return mat


def get_hair_dataset_material():
    material = bpy.data.materials.new("Hair")
    material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # remove default nodes
    for node in nodes:
        nodes.remove(node)
    
    # ceate nodes
    principled_node = nodes.new(type="ShaderNodeBsdfHairPrincipled")
    principled_node.parametrization = "MELANIN"
    principled_node.inputs['Melanin'].default_value = 0.99
    principled_node.inputs['Melanin Redness'].default_value = 0
    principled_node.inputs['Roughness'].default_value = 0.15
    principled_node.inputs['Radial Roughness'].default_value = 0.3
    principled_node.inputs['Offset'].default_value = np.radians(-2)
    if bpy.app.version >= (4, 0, 0):
        principled_node.model = "HUANG"
    
    attr_tangent_node = nodes.new(type="ShaderNodeAttribute")
    attr_tangent_node.attribute_name = "tangent"
    
    attr_aov_node = nodes.new(type="ShaderNodeAttribute")
    attr_aov_node.attribute_name = "hair_aov"
    
    vector_math_node = nodes.new(type="ShaderNodeVectorMath")
    vector_math_node.operation = "MULTIPLY_ADD"
    vector_math_node.inputs[1].default_value = (0.5, 0.5, 0.5)
    vector_math_node.inputs[2].default_value = (0.5, 0.5, 0.5)
    
    cycles_output_node = nodes.new(type="ShaderNodeOutputMaterial")
    cycles_output_node.target = "CYCLES"
    
    eevee_output_node = nodes.new(type="ShaderNodeOutputMaterial")
    eevee_output_node.target = "EEVEE"
    
    aov_output_node = nodes.new(type="ShaderNodeOutputAOV")
    aov_output_node.name = "AOV"
    
    separate_xyz_node = nodes.new(type="ShaderNodeSeparateXYZ")
    combine_xyz_node = nodes.new(type="ShaderNodeCombineXYZ")

    # create links
    links.new(principled_node.outputs["BSDF"], cycles_output_node.inputs["Surface"])
    links.new(attr_tangent_node.outputs["Vector"], vector_math_node.inputs[0])
    links.new(attr_aov_node.outputs["Vector"], aov_output_node.inputs["Color"])
    links.new(vector_math_node.outputs["Vector"], separate_xyz_node.inputs["Vector"])
    links.new(separate_xyz_node.outputs["X"], combine_xyz_node.inputs["X"])
    links.new(separate_xyz_node.outputs["Y"], combine_xyz_node.inputs["Y"])
    links.new(attr_aov_node.outputs["Fac"], combine_xyz_node.inputs["Z"])
    links.new(combine_xyz_node.outputs["Vector"], eevee_output_node.inputs["Surface"])
    
    return material


def get_hair_resample_node(count: int = 32):
    ng = bpy.data.node_groups.new("Hair Resample", "GeometryNodeTree")
    if bpy.app.version < (4, 0, 0):
        ng.inputs.new("NodeSocketGeometry", "Geometry")
        ng.outputs.new("NodeSocketGeometry", "Geometry")
    else:
        ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    
    nodes = ng.nodes
    links = ng.links
    
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")
    
    delete_geometry = nodes.new("GeometryNodeDeleteGeometry")
    delete_geometry.domain = "CURVE"
    
    compare_node = nodes.new("FunctionNodeCompare")
    compare_node.operation = "LESS_THAN"
    compare_node.inputs["B"].default_value = 0.001
    
    spline_length = nodes.new("GeometryNodeSplineLength")
    
    resample_curve = nodes.new("GeometryNodeResampleCurve")
    resample_curve.inputs['Count'].default_value = count
    
    links.new(group_input.outputs["Geometry"], resample_curve.inputs["Curve"])
     
    links.new(group_input.outputs["Geometry"], resample_curve.inputs["Curve"])
    links.new(resample_curve.outputs["Curve"], group_output.inputs["Geometry"])
    links.new(group_input.outputs["Geometry"], delete_geometry.inputs["Geometry"])
    links.new(delete_geometry.outputs["Geometry"], resample_curve.inputs["Curve"])
    links.new(compare_node.outputs["Result"], delete_geometry.inputs["Selection"])
    links.new(spline_length.outputs["Length"], compare_node.inputs["A"])
    
    return ng


def get_hair_geometry_node():
    ng = bpy.data.node_groups.new("Store Hair Tangent", "GeometryNodeTree")
    if bpy.app.version < (4, 0, 0):
        ng.inputs.new("NodeSocketGeometry", "Geometry")
        ng.outputs.new("NodeSocketGeometry", "Geometry")
    else:
        ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    
    nodes = ng.nodes
    links = ng.links
    
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")
    curve_tangent = nodes.new("GeometryNodeInputTangent")
    separate_xyz = nodes.new("ShaderNodeSeparateXYZ")
    combine_xyz = nodes.new("ShaderNodeCombineXYZ")
    normalize_node = nodes.new("ShaderNodeVectorMath")
    normalize_node.operation = "NORMALIZE"
    store_named_attribute = nodes.new("GeometryNodeStoreNamedAttribute")
    store_named_attribute.data_type = "FLOAT_VECTOR"
    store_named_attribute.inputs['Name'].default_value = "tangent"
    
    links.new(group_input.outputs["Geometry"], store_named_attribute.inputs["Geometry"])
    links.new(store_named_attribute.outputs["Geometry"], group_output.inputs["Geometry"])
    links.new(curve_tangent.outputs["Tangent"], separate_xyz.inputs["Vector"])
    links.new(separate_xyz.outputs["X"], combine_xyz.inputs["X"])
    links.new(separate_xyz.outputs["Z"], combine_xyz.inputs["Y"])
    links.new(combine_xyz.outputs["Vector"], normalize_node.inputs[0])
    links.new(normalize_node.outputs["Vector"], store_named_attribute.inputs[3])
    
    return ng


def add_envmap(hdr_path):
    node_tree = bpy.context.scene.world.node_tree
    
    environment_texture_node = node_tree.nodes.new("ShaderNodeTexEnvironment")
    environment_texture_node.image = bpy.data.images.load(hdr_path)
    maping_node = node_tree.nodes.new("ShaderNodeMapping")
    texcoord_node = node_tree.nodes.new("ShaderNodeTexCoord")
    
    background_node = next(n for n in node_tree.nodes if n.type == "BACKGROUND")
    background_node.inputs["Strength"].default_value = 2
    
    node_tree.links.new(texcoord_node.outputs["Generated"], maping_node.inputs["Vector"])
    node_tree.links.new(maping_node.outputs["Vector"], environment_texture_node.inputs["Vector"])
    node_tree.links.new(environment_texture_node.outputs["Color"], background_node.inputs["Color"])

