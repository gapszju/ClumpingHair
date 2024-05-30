import os
import sys
import cv2
import torch
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from contextlib import contextmanager

from pytorch3d.io import load_obj, save_hair
from pytorch3d.structures import Meshes, Curves
from pytorch3d.renderer.mesh import TexturesUV


@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()              # + implicit flush()
        os.dup2(to.fileno(), fd)        # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different


def as_8bit_img(image: torch.Tensor):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    return (image.clip(0, 1) * 255).astype("uint8")


def load_ref_imgs_hairstep(seg_path, orien_path, depth_path, device="cpu"):
    ref_img_orien = torch.tensor(
        plt.imread(orien_path)[..., :3][..., ::-1].copy()
    ).to(device)
    mask = ref_img_orien[..., 2] > .99

    ref_img_silh = plt.imread(seg_path).copy()
    if ref_img_silh.ndim == 3:
        ref_img_silh = ref_img_silh[..., 0]
    ref_img_silh = torch.tensor(ref_img_silh).to(device)
    
    ref_img_orien = 1-ref_img_orien
    ref_img_orien[..., 2] = 0.5
    ref_img_orien *= mask.unsqueeze(-1)
    
    ref_img_depth = np.load(depth_path).astype(np.float32)
    ref_img_depth = torch.tensor(ref_img_depth).to(device)
    
    return ref_img_orien, ref_img_silh, ref_img_depth


def load_obj_with_uv(obj_path, device="cpu"):
    verts, faces, aux = load_obj(obj_path, load_textures=False)
    mesh = Meshes(
        verts=[verts],
        faces=[faces.verts_idx],
        textures=TexturesUV(
            verts_uvs=[aux.verts_uvs],
            faces_uvs=[faces.textures_idx],
            maps=[torch.empty(0, 0, 3)],
        ),
    ).to(device)

    return mesh


def laplace_op(inputs: torch.Tensor, t: torch.Tensor, eps=1e-10):
    inputs = torch.diff(inputs, dim=1)
    delta = torch.diff(t, dim=1).clip(eps)
    inputs = inputs / delta
    
    inputs = torch.diff(inputs, dim=1)
    delta2 = (delta[:, :-1] + delta[:, 1:]) / 2
    inputs = inputs / delta2
    
    return inputs, t[:, 1:-1]


def hair_smooth_loss(hair_strands: torch.Tensor, w_c=1.0, w_l=1.0, p_c=2, p_l=2, eps=1e-8):
    """
    Calculates the smooth loss for hair strands.

    Args:
        hair_strands (torch.Tensor): Tensor representing the hair strands.
        w_c (float, optional): Weight for the curvature loss term. Defaults to 1.0.
        w_l (float, optional): Weight for the length loss term. Defaults to 1.0.
        p_c (int, optional): Power for the curvature loss term. Defaults to 2.
        p_l (int, optional): Power for the length loss term. Defaults to 2.
        eps (float, optional): Small value for numerical stability. Defaults to 1e-8.

    Returns:
        torch.Tensor: The smooth loss for the hair strands.
    """
    
    seg_len = hair_strands.diff(dim=1).norm(dim=-1)
    t = torch.cumsum(F.pad(seg_len, [1, 0]), dim=1).unsqueeze(-1) * 1e2
    
    result, t = laplace_op(hair_strands, t)
    result, t = laplace_op(result, t)
    
    loss_curvature = generalized_mean(result.norm(dim=-1), p=p_c)
    loss_length = generalized_mean(seg_len.diff(dim=-1).abs(), p=p_l)

    # print(f"loss_length: {loss_length.item():.6f}, loss_curvature: {loss_curvature.item():.6f}, ", end="")
    return w_c * loss_curvature + w_l * loss_length


def generalized_mean(tensor : torch.Tensor, p : float, eps = 1e-20, scale = 1e3):
    # Raise tensor elements to the power of p
    powered = torch.pow(tensor * scale + eps, p)
    # Calculate the mean of these powered elements
    mean_of_powers = powered.mean()
    # Take the p-th root of the result
    return torch.pow(mean_of_powers + eps, 1 / p) / scale



def find_image_transform(image1, image2):
    image1 = np.array(image1)
    image2 = np.array(image2)
    
    orb = cv2.ORB_create()

    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    num_good_matches = int(len(matches) * 0.15)
    good_matches = matches[:num_good_matches]

    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    matrix, mask = cv2.estimateAffinePartial2D(points1, points2)
    return matrix


def transform_affine(image, matrix):
    image = np.array(image)
    transformed_image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    return transformed_image