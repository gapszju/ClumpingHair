import os
import torch
import numpy as np
import struct
import multiprocessing
from .mesh_util import load_obj_mesh


def hair_synthesis(net, cuda, root_tensor, calib_tensor, num_sample=100, hair_unit=0.006):
    #root:[3, 1024]
    num_strand = root_tensor.shape[2]
    hair_strands = torch.zeros(num_sample, 3, num_strand).to(device=cuda)
    
    curr_node = root_tensor.squeeze()
    hair_strands[0] = curr_node
    for i in range(1,num_sample-1):
        curr_node_orien = net.query(curr_node.unsqueeze(0), calib_tensor)
        curr_node_orien = curr_node_orien.squeeze()
        hair_strands[i] = hair_strands[i-1] + hair_unit * curr_node_orien
        curr_node = hair_strands[i]

    return hair_strands.permute(2, 0, 1).cpu().detach().numpy()


def hair_synthesis_DSH(net, cuda, root_tensor, calib_tensor, num_sample=100, hair_unit=0.006, threshold=[60,150]):
    #growing algorithm in DeepSketchHair
    #root:[3, 1024]
    num_strand = root_tensor.shape[2]
    hair_strands = torch.zeros(num_sample, 3, num_strand).to(device=cuda)
    
    curr_node = root_tensor.squeeze()
    last_node_orien = 0
    hair_strands[0] = curr_node
    for i in range(1,num_sample-1):
        curr_node_orien = net.query(curr_node.unsqueeze(0), calib_tensor).squeeze()

        if i>1:
            len_cd = torch.norm(curr_node_orien,p=2,dim=0)
            len_pd = torch.norm(last_node_orien,p=2,dim=0)
            in_prod = torch.sum(curr_node_orien * last_node_orien, dim=0)
            theta = torch.acos( in_prod/ (len_cd*len_pd))*180/np.pi

            idx_big_theta = theta > threshold[1]
            idx_mid_theta = ((theta > threshold[0]).float() - idx_big_theta.float()).bool().unsqueeze(0)#60<theta<150
            idx_stop = (idx_big_theta + torch.isnan(theta).float()).bool().unsqueeze(0)#orien=0 or theta>150

            idx_stop = torch.cat((idx_stop,idx_stop,idx_stop),dim=0)
            idx_mid_theta = torch.cat((idx_mid_theta,idx_mid_theta,idx_mid_theta),dim=0)
            

            half_node_orien = (curr_node_orien + last_node_orien)/2

            orien_zeros = torch.zeros_like(curr_node_orien).float().to(device=cuda)
            curr_node_orien = torch.where(idx_stop, orien_zeros, curr_node_orien)
            curr_node_orien = torch.where(idx_mid_theta, half_node_orien, curr_node_orien)

        hair_strands[i] = hair_strands[i-1] + hair_unit * curr_node_orien
        curr_node = hair_strands[i]
        last_node_orien = curr_node_orien

    return hair_strands.permute(2, 0, 1).cpu().detach().numpy()


def write_hair_cy(filepath, num_verts, positions):
    """Saves the given strands to Cem Yuksel's hair format.

    The file format specification can be found in:
    http://www.cemyuksel.com/research/hairmodels/

    Args:
        filepath: path of the saved hair file.
        num_verts: number of vertices per strand.
        positions: positions of the vertices.
    """
    if not filepath.endswith(".hair"):
        raise NotImplementedError(f"{filepath} is not a .hair file!")
    
    num_strands = len(num_verts)
    num_points = len(positions)

    with open(filepath, "wb") as fp:
        fp.write(b"HAIR")
        fp.write(struct.pack("I", num_strands))
        fp.write(struct.pack("I", num_points))
        bit_flags = "11000"
        fp.write(struct.pack("i", int(bit_flags[::-1], 2)))
        fp.write(b"\0" * (40-16)) # skips unused fields
        
        # file information
        info = ""
        fp.write(bytes((info + "\0"*88)[:88], "ascii"))

        fp.write(np.array(num_verts).astype(np.uint16)-1)
        fp.write(np.array(positions).astype(np.float32))


def trim_strands_by_mesh(strands, mesh_path, err=0.3):
    import bpy
    import bmesh
    from mathutils.bvhtree import BVHTree

    lst_pc_all_valid = []
    lst_num_pt = []
    pc_all_valid = []
    num_pt_valid = []

    # read surface data
    bpy.ops.wm.read_factory_settings()
    mesh = bpy.data.meshes.new('mesh')
    verts, faces = load_obj_mesh(mesh_path)
    mesh.from_pydata(verts, [], faces)

    bm = bmesh.new()
    bm.from_mesh(mesh)
    tree = BVHTree.FromBMesh(bm)

    # remove out of bound vertices
    for i in range(strands.shape[0]):
        current_pc_all_valid = []
        first_step = strands[i,0] - strands[i,1]
        first_step = np.dot(first_step, first_step)
        if np.dot(strands[i,0], strands[i,0])<0.001 or np.dot(strands[i,1], strands[i,1])<0.001:
            continue
        num_pt = 2
        current_pc_all_valid.append(strands[i][0])
        current_pc_all_valid.append(strands[i][1])
        
        for j in range(2,strands.shape[1]):
            location, normal, index, distance = tree.find_nearest(strands[i][j])
            direction = strands[i][j] - location
            if direction.dot(normal) < 0:
                current_pc_all_valid.append(strands[i][j])
                num_pt += 1
            else:
                break
        
        lst_pc_all_valid.append(current_pc_all_valid)
        lst_num_pt.append(num_pt)

    min_num_pts = int(sum(lst_num_pt)/len(lst_num_pt)*err)

    # remove short strands
    for i in range(strands.shape[0]):
        if lst_num_pt[i] > min_num_pts:
            for j in range(lst_num_pt[i]):
                pc_all_valid.append(lst_pc_all_valid[i][j])
            num_pt_valid.append(lst_num_pt[i])
    
    return num_pt_valid, pc_all_valid


def save_strands_with_mesh(strands, mesh_path, outputpath, err=0.3):
    num_workers = multiprocessing.cpu_count() // 4
    sub_strands = np.array_split(strands, num_workers)
    
    # use multiprocessing to speed up the process
    with multiprocessing.Pool(num_workers) as p:
        res = p.starmap(trim_strands_by_mesh, [(sub_strands[i], mesh_path, err) for i in range(num_workers)])
    num_verts, points = map(lambda x: sum(x, []), zip(*res))
    
    write_hair_cy(outputpath, num_verts, points)
    

def get_hair_root(filepath='./data/roots80k.obj'):
    root, _ = load_obj_mesh(filepath)
    return root.T


def export_hair_real(net, cuda, data, mesh_path, save_path):
    image_tensor = data['hairstep'].to(device=cuda).unsqueeze(0)
    calib_tensor = data['calib'].to(device=cuda).unsqueeze(0)
    root_tensor = torch.from_numpy(get_hair_root()).to(device=cuda).float().unsqueeze(0)

    net.filter(image_tensor)

    strands = hair_synthesis(net, cuda, root_tensor, calib_tensor, num_sample=100, hair_unit=0.006)
    save_strands_with_mesh(strands, mesh_path, save_path, 0.3)
