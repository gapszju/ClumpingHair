import os
import numpy as np
import torch
from pytorch3d.io import save_hair
from pytorch3d.ops import knn_points

import imath
import imathnumpy
import alembic
import alembic.Abc as abc
import alembic.AbcGeom as abcGeom


def read_hair_abc(filepath: str) -> dict:
    """Read hair data group from Alembic file, and return a curve dict contains contain numverts,
    positions and widths, represented as numpy arrays.
    """
    def get_children_recursive(parent):
        children = []
        for child in parent.children:
            children.append(child)
            children += get_children_recursive(child)
        return children

    def get_property_data(compound_property: alembic.Abc.ICompoundProperty, name: str):
        if not compound_property.valid():
            return None
        try:
            prop = compound_property.getProperty(name)
            return prop.getValue()
        except:
            return None
        
    def get_global_matrix(iobject: abc.IObject):
        """Returns world space matrix for this object.
        """
        matrix = imath.M44d()
        matrix.makeIdentity()
        parent = iobject
        while parent.valid():
            """recursive xform accum"""
            if abcGeom.IXform.matches(parent.getHeader()):
                xform = abcGeom.IXform(parent, abc.WrapExistingFlag.kWrapExisting)
                matrix *= xform.getSchema().getValue().getMatrix()
            parent = parent.getParent()
        return matrix / matrix[3][3]
    
    print(f"Reading hair data from {filepath}")
    
    # Open Alembic file
    archive = abc.IArchive(filepath)
    top = archive.getTop()
    
    data = {}
    for name in ["guide", "hair"]:
        data.setdefault(name, {
            "numverts": np.empty(0, dtype=np.int32),
            "positions": np.empty((0, 3), dtype=np.float32),
            "widths": np.empty(0, dtype=np.float32),
        })

    for child in get_children_recursive(top):
        # Checks if it is a curve object
        if abcGeom.ICurves.matches(child.getHeader()):
            # Get the data and group info for the curve
            curves = abcGeom.ICurves(child, abc.WrapExistingFlag.kWrapExisting)
            schema = curves.getSchema()
            
            assert schema.valid() and \
                   schema.getNumVerticesProperty().valid() and \
                   schema.getPositionsProperty().valid()
            
            groom_guide = get_property_data(schema.getArbGeomParams(), "groom_guide")
            if groom_guide and groom_guide[0]:
                name = "guide"
            else:
                name = "hair"
            
            # Calculate the global matrix to the positions and widths
            global_matrix = get_global_matrix(curves)
            scale = imath.V3d()
            global_matrix.extractScaling(scale)
            
            # Gets position and width data for the curve
            numverts = schema.getNumVerticesProperty().getValue()
            positions = schema.getPositionsProperty().getValue()
            positions = global_matrix.multVecMatrix(positions)
                
            data[name]["numverts"] = np.append(
                data[name]["numverts"], imathnumpy.arrayToNumpy(numverts))
            data[name]["positions"] = np.append(
                data[name]["positions"], imathnumpy.arrayToNumpy(positions), axis=0)
            
            if schema.getWidthsParam().valid():
                widths = schema.getWidthsParam().getExpandedValue().getVals()
                data[name]["widths"] = np.append(
                    data[name]["widths"], imathnumpy.arrayToNumpy(widths * scale[0]))

    return data


def convert(hair_dir, scale):
    hair_name = os.path.basename(hair_dir)
    hair_data = {}
        
    # read hair data from alembic files
    for mod_name in os.listdir(hair_dir):
        mod_path = os.path.join(hair_dir, mod_name)
        if not os.path.isdir(mod_path):
            continue
            
        hair_data[mod_name] = {"strands": [], "widths": []}
            
        for file in os.listdir(mod_path):
            if file.endswith(".abc") and not file.startswith("guides"):
                filepath = os.path.join(mod_path, file)
                data = read_hair_abc(filepath)["hair"]
                indices = np.cumsum(data["numverts"])[:-1]
                hair_data[mod_name]["strands"] += np.split(data["positions"], indices),
                hair_data[mod_name]["widths"] += np.split(data["widths"], indices),
    
    # alignment point order
    for gid in range(len(hair_data["Wo_Modifiers"]["strands"])):
        ref_strands = hair_data["Wo_Modifiers"]["strands"][gid]
        hair_roots_ref = [strand[0] for strand in ref_strands]
        hair_roots_ref = torch.tensor(np.array(hair_roots_ref, dtype=np.float32))[None].cuda()
        
        for mod_name in hair_data.keys():
            if mod_name == "Wo_Modifiers":
                continue
            
            strands = hair_data[mod_name]["strands"][gid]
            hair_roots = [strand[0] for strand in strands]
            hair_roots = torch.tensor(np.array(hair_roots, dtype=np.float32))[None].cuda()
            dists, idx, _ = knn_points(hair_roots_ref, hair_roots, K=2)
            if dists[..., 0].max() > 1e-8:
                raise ValueError("Hair roots are not aligned.")
            if (dists[..., 1] - dists[..., 0]).min() == 0:
                raise ValueError("Hair roots has overlap.")
            
            strands_new = []
            widths_new = []
            for i, sid in enumerate(idx[0,:,0].cpu()):
                num_verts = len(ref_strands[i])
                if num_verts != len(strands[sid]):
                    raise ValueError("Hair strands num_verts are not inconsistent.")
                strands_new.append(strands[sid])
                widths_new.append(hair_data[mod_name]["widths"][gid][sid])
            
            hair_data[mod_name]["strands"][gid] = strands_new
            hair_data[mod_name]["widths"][gid] = widths_new
            
    # save hair data
    for mod_name in hair_data.keys():
        output_path = os.path.join(hair_dir, f"{hair_name}_{mod_name}.hair")
        strands = sum(hair_data[mod_name]["strands"], start=[])
        widths = sum(hair_data[mod_name]["widths"], start=[])
        
        num_verts = [len(strand) for strand in strands]
        positions = np.concatenate(strands, axis=0)
        thicknesses = np.concatenate(widths, axis=0)
        save_hair(output_path, num_verts, positions*scale, thicknesses*scale)


def convert_single_hair(hair_path, out_path, scale):
    data = read_hair_abc(hair_path)["hair"]
    save_hair(out_path, data["numverts"], data["positions"]*scale, data["widths"]*scale)


if __name__ == "__main__":
    import glob
    data_dir = "X:/contrastive_learning/data/assets/adapted"
    for hair_path in glob.glob(os.path.join(data_dir, "**", "*.abc"), recursive=True):
        hair_name = os.path.basename(hair_path).split("_finalHair")[0]
        out_path = os.path.join(data_dir, f"{hair_name}.hair")
        convert_single_hair(hair_path, out_path, 1)
    
    exit()
    
    data_dir = os.path.join(os.path.dirname(__file__), "data", "assets", "DD_woman")
    log_file = os.path.join(data_dir, "convert_successed.txt")
    scale = 0.01
    
    successed = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            successed = f.read().splitlines()

    for hair_name in os.listdir(data_dir):
        hair_dir = os.path.join(data_dir, hair_name)
        if os.path.isdir(hair_dir) and hair_name not in successed:
            print(f"\n-=== Converting {hair_name}")
            convert(hair_dir, scale)
            
            with open(log_file, "a") as f:
                f.write(hair_name + "\n")
        else:
            print(f"\n-=== Skip {hair_name}")