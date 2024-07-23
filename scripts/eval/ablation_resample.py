import os
import sys 
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.hair_model import HairModel
from src.utils import render_hair_shading, render_hair_color, save_hair_strands


head_path = "X:/hairstep/head_smooth.obj"
# hair_path = "X:/hairstep/HiSa_HiDa/hair3D/resample_32/14babc004f6dc3f4a9b1150ca1399c01.hair"
# out_path = "X:/results/resample/14babc004f6dc3f4a9b1150ca1399c01/mean_loss.png"
# hair_path = "X:/hairstep/HiSa_HiDa/hair3D/resample_32/47f10aca2920b9635ab5f94c324e2061.hair"
# out_path = "X:/results/resample/47f10aca2920b9635ab5f94c324e2061/mean_loss.png"
hair_path = "X:/hairstep/HiSa_HiDa/hair3D/resample_32/cc5bf02bc6205f878cde02df9d00e7b9.hair"
out_path = "X:/results/resample/cc5bf02bc6205f878cde02df9d00e7b9/gem_loss.png"
# hair_path = "X:/hairstep/USC_HairSalon/hair_gt/strands00187.hair"
# out_path = "X:/results/resample/strands00187/mean_loss.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

hair_result_path = out_path.replace(".png", ".hair")
camera_path = hair_path.replace(".hair", "_camera.json")

hair_model = HairModel(hair_path, n_guide=256, device="cuda")
hair_interp = hair_model.eval()
save_hair_strands(hair_result_path, hair_interp)

dists = torch.abs(hair_interp - hair_model.hair_strands).norm(dim=-1).cpu().numpy()
color = plt.cm.jet(dists / 1e-2)

render_hair_color(head_path, hair_result_path, color, out_path.replace(".png", "_vis.png"), camera_path,
            img_size=1024, side_view=False, device_idx=0)
render_hair_shading(head_path, hair_result_path, out_path, camera_path,
            img_size=1024, side_view=False, device_idx=0)
