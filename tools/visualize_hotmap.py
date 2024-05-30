import os
import torch
import pyexr
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons

from model.network import sNet
    
    
def update_ref_slider(val=None):
    global ref_img, ax_ref_img
    img_name = f"cluster_{int(ref_sliders[0].val)}_noise_{ref_sliders[1].val:.4f}_scale_{ref_sliders[2].val:.4f}" +\
               f"_shape_{ref_sliders[3].val:.4f}_noise2_{ref_sliders[4].val:.4f}.exr"
    img_path = os.path.join(data_path_ref, img_name) if ref_img_path is None else ref_img_path
    ref_img = pyexr.read(img_path)[..., :3]
    ref_img = 0.299 * ref_img[..., 0:1] + 0.587 * ref_img[..., 1:2] + 0.114 * ref_img[..., 2:3]

    if ax_ref_img is None:
        ax_ref_img = axes[0].imshow(ref_img.clip(0, 1).repeat(3, -1))
    else:
        ax_ref_img.set_data(ref_img.clip(0, 1).repeat(3, -1))
    fig.canvas.draw_idle()
    
    ref_img = torch.tensor(ref_img, dtype=torch.float16).cuda()
    
    # synchronize sliders
    if setting.get_status()[1]:
        for i in range(len(ref_sliders)):
            if ref_sliders[i].drag_active:
                feat_sliders[i].set_val(ref_sliders[i].val)
                if setting.get_status()[2]:
                    draw_similarities()
    else:
        if setting.get_status()[2]:
            draw_similarities()                


def update_feat_slider(val=None):
    global feature_img, ax_feature_img
    img_name = f"cluster_{int(feat_sliders[0].val)}_noise_{feat_sliders[1].val:.4f}_scale_{feat_sliders[2].val:.4f}" +\
               f"_shape_{feat_sliders[3].val:.4f}_noise2_{feat_sliders[4].val:.4f}.exr"
    image = pyexr.open(os.path.join(data_path_feat, img_name))
    depth_img = image.get("depth")[..., :1]
    orien_img = image.get("orientation")[..., :2]
    feature_img = np.concatenate([depth_img, orien_img], axis=-1)
    
    if ax_feature_img is None:
        ax_feature_img = axes[1].imshow(feature_img.clip(0, 1))
    else:
        ax_feature_img.set_data(feature_img.clip(0, 1))
    fig.canvas.draw_idle()
    
    feature_img = torch.tensor(feature_img, dtype=torch.float16).cuda()
    
    # synchronize sliders
    if setting.get_status()[1]:
        for i in range(len(feat_sliders)):
            if feat_sliders[i].drag_active:
                ref_sliders[i].set_val(feat_sliders[i].val)
                if setting.get_status()[2]:
                    draw_similarities()
    else:
        if setting.get_status()[2]:
            draw_similarities()


def update_index_slider(val=None):
    global data_path_ref, data_path_feat
    data_path_ref = os.path.join(data_dir, dataset, hair_list[int(s_index.val)])
    data_path_feat = os.path.join(data_dir, dataset, hair_list[int(s_index.val)])
    
    axes[0].set_xlabel(os.path.basename(data_path_ref))
    axes[1].set_xlabel(os.path.basename(data_path_feat))
    update_ref_slider()
    update_feat_slider()


def update_dataset(val=None):
    global dataset, hair_list, s_index
    if val == "Train Dataset":
        dataset = "train" if setting.get_status()[0] else "test"
        hair_list = os.listdir(os.path.join(data_dir, dataset))
        
        index = int(s_index.val)
        s_index.ax.remove()
        s_index = Slider(plt.axes([0.95, 0.2, 0.03, 0.6]), 'Index', 0, len(hair_list)-1, valinit=len(hair_list)-1, valstep=1,
                        orientation="vertical", facecolor="lightgray", initcolor="lightgray")
        s_index.on_changed(update_index_slider)
        s_index.set_val(min(index, len(hair_list)-1))
        
        update_index_slider()
    elif val == "Link Parameters":
        if setting.get_status()[1]:
            for i in range(len(ref_sliders)):
                feat_sliders[i].set_val(ref_sliders[i].val)
    else:
        update_index_slider()


def on_click(event):
    global roi
    if event.inaxes == axes[0]:
        if event.button == 3:
            roi = None
        elif event.button == 1:
            min_val = P//2
            max_val = ref_img.shape[0] - P//2
            roi = [int(np.clip(event.ydata, min_val, max_val)),
                   int(np.clip(event.xdata, min_val, max_val))]
        draw_similarities()


def on_press(event):
    if event.key == "control" and ax_similarity.get_visible():
        ax_similarity.set_visible(False)
        ax_feature_resize.set_visible(True)
        fig.canvas.draw_idle()

def on_release(event):
    if event.key == "control" and ax_feature_resize.get_visible():
        ax_feature_resize.set_visible(False)
        ax_similarity.set_visible(True)
        fig.canvas.draw_idle()
    

def draw_similarities(event=None):
    if ref_img is None or feature_img is None:
        return
    ax_ref_img.set_data(ref_img.float().clip(0, 1).cpu().numpy().repeat(3, -1))
    ax_feature_img.set_data(feature_img.float().clip(0, 1).cpu().numpy())

    global ax_feature_resize, ax_similarity, similarity_img, feature_img_resize
    with torch.no_grad():
        ref_img_patches = ref_img.unfold(0, P, 3).unfold(1, P, 3)
        if roi is not None:
            y, x = (roi[0]-P//2)//3, (roi[1]-P//2)//3
            ref_img_patches = ref_img_patches[y, x].unsqueeze(0).unsqueeze(0)
        ref_img_embeddings = torch.stack([model(img, None) for img in ref_img_patches])
        
        feature_img_patches = feature_img.unfold(0, P, 3).unfold(1, P, 3)
        feature_img_embeddings = torch.stack([model(None, img) for img in feature_img_patches])
    
    feature_img_resize = feature_img_patches[:, :, :, P//2, P//2]
    similarity_img = (F.normalize(ref_img_embeddings, dim=-1) * F.normalize(feature_img_embeddings, dim=-1)).sum(dim=-1)
    
    if ax_feature_resize is None:
        ax_feature_resize = axes[2].imshow(feature_img_resize.clip(0, 1).float().cpu().numpy())
    else:
        ax_feature_resize.set_data(feature_img_resize.clip(0, 1).float().cpu().numpy())
    
    if ax_similarity is None:
        ax_similarity = axes[2].imshow(similarity_img.float().cpu().numpy(), cmap="viridis", vmin=0, vmax=1)
    else:
        ax_similarity.set_data(similarity_img.float().cpu().numpy())
    
    axes[2].set_xlabel(f"Min: {similarity_img.min():.4f}, Max: {similarity_img.max():.4f}, Mean: {similarity_img.mean():.4f}")
    
    for ax in axes:
        for patch in ax.patches:
            patch.remove()
        for line in ax.lines:
            line.remove()
        
    if roi is not None:
        axes[0].add_patch(plt.Rectangle((roi[1]-P//2, roi[0]-P//2), P, P, linewidth=1, edgecolor="r", facecolor="none"))
        axes[1].add_patch(plt.Rectangle((roi[1]-P//2, roi[0]-P//2), P, P, linewidth=1, edgecolor="r", facecolor="none", linestyle="--"))
        axes[2].axvline(x=x, color="r", linestyle="--")
        axes[2].axhline(y=y, color="r", linestyle="--")
    
    fig.canvas.draw_idle()


if __name__ == "__main__":
    arch = "resnet18"
    out_dim = 128
    in_channels = [1, 3]
    P = 128 # patch size
    data_dir = "X:/contrastive_learning/data/clumping_dataset/full_params_3"
    ckpt_path = "runs/Mar11_21-59-12_DESKTOP-FLDT8US_wo_pooling/model_best.pth.tar"
    # ckpt_path = "runs/Patch3/Jan14_00-15-29_DESKTOP-FLDT8US/model_best.pth.tar"
    # ckpt_path = "runs/Patch6/Jan25_01-00-15_ROG/model_best.pth.tar"
    ref_img_path = None
    # ref_img_path = "X:/contrastive_learning/data/xgen_full_render/DD_0528_02_changfa_hair002.exr"
    
    # global variables
    index = 1
    roi = None
    dataset = "test"
    hair_list = os.listdir(os.path.join(data_dir, dataset))
    data_path_ref =  os.path.join(data_dir, dataset, hair_list[0])
    data_path_feat = os.path.join(data_dir, dataset, hair_list[0])
    ref_img = None
    feature_img = None
    feature_img_resize = None
    similarity_img = None
    
    # create figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    plt.subplots_adjust(bottom=0.35)
    fig.canvas.mpl_connect("button_press_event", on_click)
        
    ax_ref_img, ax_feature_img, ax_feature_resize, ax_similarity = None, None, None, None
    axes[0].set_xticks([]), axes[0].set_yticks([]), axes[0].set_title("Reference Image")
    axes[1].set_xticks([]), axes[1].set_yticks([]), axes[1].set_title("Feature Image")
    axes[2].set_xticks([]), axes[2].set_yticks([]), axes[2].set_title("Similarities")
    
    # slider to control parameters
    ref_sliders = []
    ref_sliders += Slider(plt.axes([0.16, 0.25, 0.15, 0.03]), 'Cluster', 256, 4096, valinit=256, valstep=[256, 1024, 4096]),
    ref_sliders += Slider(plt.axes([0.16, 0.2, 0.15, 0.03]), 'Noise', 0, 0.04, valinit=0.02, valstep=0.01),
    ref_sliders += Slider(plt.axes([0.16, 0.15, 0.15, 0.03]), 'Scale', 0.2, 1, valinit=1, valstep=0.2),
    ref_sliders += Slider(plt.axes([0.16, 0.1, 0.15, 0.03]), 'Shape', 0.5, 2, valinit=1, valstep=[0.5, 1, 2]),
    ref_sliders += Slider(plt.axes([0.16, 0.05, 0.15, 0.03]), 'Noise2', 0, 0.002, valinit=0.002, valstep=0.002),
    [slider.on_changed(update_ref_slider) for slider in ref_sliders]

    feat_sliders = []
    feat_sliders += Slider(plt.axes([0.44, 0.25, 0.15, 0.03]), 'Cluster', 256, 4096, valinit=256, valstep=[256, 1024, 4096]),
    feat_sliders += Slider(plt.axes([0.44, 0.2, 0.15, 0.03]), 'Noise', 0, 0.04, valinit=0.02, valstep=0.01),
    feat_sliders += Slider(plt.axes([0.44, 0.15, 0.15, 0.03]), 'Scale', 0.2, 1, valinit=1, valstep=0.2),
    feat_sliders += Slider(plt.axes([0.44, 0.1, 0.15, 0.03]), 'Shape', 0.5, 2, valinit=1, valstep=[0.5, 1, 2]),
    feat_sliders += Slider(plt.axes([0.44, 0.05, 0.15, 0.03]), 'Noise2', 0, 0.002, valinit=0.002, valstep=0.002),
    [slider.on_changed(update_feat_slider) for slider in feat_sliders]
    
    # botton to calculate similarities
    bn1 = Button(plt.axes([0.74, 0.22, 0.1, 0.04]), "Calculate")
    bn1.on_clicked(draw_similarities)
    
    # switch result image
    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_connect("key_release_event", on_release)
    
    # choose dataset
    setting = CheckButtons(plt.axes([0.72, 0.05, 0.15, 0.15]),
                            ["Train Dataset", "Link Parameters", "Auto Calculate"],
                            [False, False, False])
    setting.on_clicked(update_dataset)
    
    # data bar
    s_index = Slider(plt.axes([0.95, 0.2, 0.03, 0.6]), 'Index', 0, len(hair_list)-1, valinit=index, valstep=1,
                     orientation="vertical", facecolor="lightgray", initcolor="lightgray")
    s_index.on_changed(update_index_slider)
    
    # load model
    model = sNet(arch, out_dim, in_channels)
    checkpoint = torch.load(ckpt_path, map_location="cuda")["state_dict"]
    model.load_state_dict(checkpoint)
    model.half()
    model.cuda()
    model.eval()
    
    update_dataset()
    draw_similarities()
    
    plt.show()
    
    