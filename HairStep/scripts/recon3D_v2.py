import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import json
import torchvision.transforms as transforms

from tqdm import tqdm
import imageio
from PIL import Image

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.hair_util import *
from lib.model.recon3D import *


def perspective(points, calibrations, transforms=None):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx4x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)

    xyz = torch.cat([xy, points[:, 2:3, :]], 1)
    return xyz

class HGPIFuNet_orien_v2(HGPIFuNet_orien):
    def query(self, points, calibs, depth_maps=None, transforms=None, labels=None):
        '''
        override the query function in HGPIFuNet_orien,
        use perspective projection instead of orthographic projection
        '''
        if labels is not None:
            self.labels = labels

        xyz = perspective(points, calibs, transforms)
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]

        if self.opt.vis_loss:
            depth_proj = self.index(depth_maps.unsqueeze(1), xy)
            vis_ones = torch.ones_like(depth_proj).to(device=depth_proj.device)
            vis_bool = torch.abs(depth_proj - z) < self.opt.depth_vis
            out_mask_bool = (depth_proj<=self.opt.depth_out_mask)
            self.vis_weight = torch.where(vis_bool, vis_ones*10, vis_ones).float()
            self.vis_weight = torch.where(out_mask_bool, vis_ones, self.vis_weight).float()

        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)

        position_feat = self.embedder(points, self.orign_embedder, self.embedder_outDim)

        self.intermediate_preds_list = []
        for idx in range(len(self.im_feat_list)):
            im_feat = self.im_feat_list[idx]
            point_local_feat_list = [self.index(im_feat, xy), position_feat]
            point_local_feat = torch.cat(point_local_feat_list, 1)

            # out of image plane is always set to 0
            pred = in_img[:,None].float() * self.surface_classifier(point_local_feat)
            self.intermediate_preds_list.append(pred)

        self.preds = self.intermediate_preds_list[-1]

        return self.preds

def load_hairstep(orien2d_path, depth_path, seg_path, load_size=512):
    raw_orien2d = Image.open(orien2d_path).convert('RGB')
    img_to_tensor = transforms.Compose([
            transforms.Resize(load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    orien2d = img_to_tensor(raw_orien2d).float()

    out_mask = ((np.array(imageio.imread(seg_path))/255)<0.5)
    if len(out_mask.shape)==3:
        out_mask = out_mask[:,:,0]
    depth = np.load(depth_path)
    depth = depth + out_mask*opt.depth_out_mask
    depth = torch.from_numpy(depth).float()

    hairstep = torch.cat([orien2d, depth.unsqueeze(0)], dim=0)
        
    return hairstep

def load_calib(calib_path, loadSize=1024):
    # loading calibration data
    with open(calib_path) as f:
        params = json.load(f)
    
    # extrinsic
    pos = torch.tensor(params["position"], dtype=torch.float32)
    look_at = torch.tensor(params["look_at"], dtype=torch.float32)
    up = torch.tensor(params["up"], dtype=torch.float32)
    
    z_axis = pos - look_at
    z_axis /= z_axis.norm()
    x_axis = up.cross(z_axis)
    x_axis /= x_axis.norm()
    y_axis = z_axis.cross(x_axis)

    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    T = -R @ pos
    
    # intrisic
    f = 1 / np.tan(np.radians(params["fov"]) / 2)
    intrisic = torch.tensor(
        [[f, 0, 0], [0, -f, 0], [0, 0, -1]], dtype=torch.float32
    )

    calib = torch.zeros(4, 4, dtype=torch.float32)
    calib[:3, :3] = intrisic @ R
    calib[:3, 3] = intrisic @ T
    calib[3, 3] = 1

    return calib

def load_occNet(cuda, opt):
    # create net
    net_path = opt.checkpoint_hairstep2occ
    net = HGPIFuNet_orien_v2(opt).to(device=cuda)

    # load checkpoints
    print('loading for occNet ...', net_path)
    net.load_state_dict(torch.load(net_path, map_location=cuda))
    net.eval()

    return net

def load_orienNet(cuda, opt):
    # create net
    net_path = opt.checkpoint_hairstep2orien
    net = HGPIFuNet_orien_v2(opt, gen_orien=True).to(device=cuda)

    # load checkpoints
    print('loading for orienNet ...', net_path)
    net.load_state_dict(torch.load(net_path, map_location=cuda))
    net.eval()

    return net

def recon3D_from_hairstep(opt):
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    seg_dir = os.path.join(opt.root_real_imgs, 'seg')
    depth_dir = os.path.join(opt.root_real_imgs, 'depth_map')
    strand_dir = os.path.join(opt.root_real_imgs, 'strand_map')
    calib_dir = os.path.join(opt.root_real_imgs, 'param')

    output_mesh_dir = os.path.join(opt.root_real_imgs, 'mesh')
    output_hair3D_dir = os.path.join(opt.root_real_imgs, 'hair3D')

    os.makedirs(output_mesh_dir, exist_ok=True)
    os.makedirs(output_hair3D_dir, exist_ok=True)

    occ_net = load_occNet(cuda, opt)
    orien_net = load_orienNet(cuda, opt)

    items = os.listdir(strand_dir)

    with torch.no_grad():
        for item in tqdm(items):
            strand_path = os.path.join(strand_dir, item[:-3] + 'png')
            seg_path = os.path.join(seg_dir, item[:-3] + 'png')
            mesh_path = os.path.join(output_mesh_dir, item[:-3] + 'obj')
            hair3D_path = os.path.join(output_hair3D_dir, item[:-3] + 'hair')
            calib_path = os.path.join(calib_dir, item[:-3] + 'json')
            depth_path = os.path.join(depth_dir, item[:-3] + 'npy')

            if os.path.exists(hair3D_path):
                continue

            calib = load_calib(calib_path)
            hairstep = load_hairstep(strand_path, depth_path, seg_path, opt.loadSize)

            test_data = {'hairstep': hairstep, 'calib':calib}

            gen_mesh_real(opt, occ_net, cuda, test_data, mesh_path)
            export_hair_real(orien_net, cuda, test_data, mesh_path, hair3D_path)

if __name__ == '__main__':
    opt = BaseOptions().parse()
    opt.root_real_imgs = "./results/hair_0021"
    recon3D_from_hairstep(opt)