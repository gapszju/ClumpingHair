import os
import torch
import numpy as np
from pytorch3d.io import load_hair
from .utils import *

device = torch.device("cuda")


@torch.no_grad()
def compute_hisa(ref_image_orien: torch.Tensor, pred_image_orien: torch.Tensor):
    """
    Compute the HairSale metric.

    Args:
        ref_image_orien (torch.Tensor): Reference orientation map.
        pred_image_orien (torch.Tensor): Predicted orientation map.

    Returns:
        float: HairSale metric.
    """
    ref_mask = ref_image_orien[..., :2].sum(-1) > 0
    pred_mask = pred_image_orien[..., :2].sum(-1) > 0
    mask = ref_mask & pred_mask
    
    ref_orien = ref_image_orien[mask][:, :2] * 2 - 1
    ref_orien /= ref_orien.norm(dim=-1, keepdim=True)
        
    pred_orien = pred_image_orien[mask][:, :2] * 2 - 1
    pred_orien /= pred_orien.norm(dim=-1, keepdim=True)
    
    cos_sim = (ref_orien * pred_orien).sum(dim=-1).clip(0, 1)
    hisa = cos_sim.acos().mean() * 180 / np.pi
    
    return hisa.item()


@torch.no_grad()
def compute_hida(image_depth: torch.Tensor, hida_pairs: torch.Tensor, hida_labels: torch.Tensor):
    """
    Compute the HairRida metric.

    Args:
        image_depth (torch.Tensor): Depth map.
        hida_pairs (torch.Tensor): Pairs of indices for computing relative depth.
        hida_labels (torch.Tensor): Labels for the relative depth pairs.

    Returns:
        float: HairRida metric.
    """
    image_mask = image_depth > 0
    
    sample_labels = image_depth[hida_pairs[:, :, 0], hida_pairs[:, :, 1]].diff(dim=-1)[:, 0] > 0
    sample_masks = image_mask[hida_pairs[:, :, 0], hida_pairs[:, :, 1]].all(dim=-1)
    
    hida = (sample_labels.int() == hida_labels)[sample_masks].float().mean()
    
    return hida.item()


def eval_image_metric(config, hair_strands: torch.Tensor, vis=False):
    """
    Evaluate image metrics for hair modeling.

    Args:
        config (dict): Configuration parameters for evaluation.
        hair_strands (torch.Tensor): Tensor containing hair strand data.

    Returns:
        tuple: A tuple containing the calculated metrics:
            - iou (float): Intersection over Union (IoU) metric.
            - hisa (float): HairSale metric.
            - hida (float): HairRida metric.
    """
    
    # reference
    ref_img_orien, ref_img_silh, ref_img_depth = load_ref_imgs_hairstep(
        config["ref_seg_path"], config["ref_orien_path"], config["ref_depth_path"], device=device)
    ref_img_mask = ref_img_orien[..., :2].sum(-1) > 0
    H, W = ref_img_depth.shape

    # setup renderer
    cameras = load_cameras(config["camera_path"], device=device)
    head_mesh = load_obj_with_uv(config["head_path"], device=device)
    head_verts_ndc = transform_points_to_ndc(cameras, head_mesh.verts_packed())
    hair_strands_ndc = transform_points_to_ndc(cameras, hair_strands)
    gl_render = GlHairRenderer(
        hair_strands_ndc, head_verts_ndc, head_mesh.faces_packed(), image_size=(W, H))

    # render image
    image_orien, image_silh, image_depth = gl_render.render()
    
    # evaluation
    inter = image_silh.bool() & ref_img_mask
    union = image_silh.bool() | ref_img_mask
    iou = inter.float().sum().item() / union.float().sum().item()
    
    hisa = compute_hisa(ref_img_orien, image_orien)
    hida = 0.0
    
    if "hida_pair_path" in config and "hida_label_path" in config:
        hida_pairs = torch.tensor(np.load(config["hida_pair_path"])).to(device)
        hida_labels = torch.tensor(np.load(config["hida_label_path"])).to(device)
        hida = compute_hida(image_depth, hida_pairs, hida_labels)
    
    # visualization
    if vis:
        fig, axes = plt.subplots(2, 3)
        fig.tight_layout()
        axes[0, 0].imshow(ref_img_silh.cpu().numpy())
        axes[0, 1].imshow(ref_img_depth.cpu().numpy())
        axes[0, 2].imshow(ref_img_orien.cpu().numpy())
        axes[1, 0].imshow(image_silh.cpu().numpy())
        axes[1, 1].imshow(image_depth.cpu().numpy())
        axes[1, 2].imshow(image_orien.cpu().numpy())
        plt.show()
    
    return iou, hisa, hida


def compare_image_metric(config, init_strands: torch.Tensor, result_strands: torch.Tensor):   
    # reference
    ref_img_orien, ref_img_silh, ref_img_depth = load_ref_imgs_hairstep(
        config["ref_seg_path"], config["ref_orien_path"], config["ref_depth_path"], device=device)
    H, W = ref_img_depth.shape

    hida_pairs = torch.tensor(np.load(config["hida_pair_path"])).to(device)
    hida_labels = torch.tensor(np.load(config["hida_label_path"])).to(device)
    
    # setup renderer
    cameras = load_cameras(config["camera_path"], device=device)
    head_mesh = load_obj_with_uv(config["head_path"], device=device)
    head_verts_ndc = transform_points_to_ndc(cameras, head_mesh.verts_packed())
    init_strands_ndc = transform_points_to_ndc(cameras, init_strands)
    result_strands_ndc = transform_points_to_ndc(cameras, result_strands)
    gl_render = GlHairRenderer(
        init_strands_ndc, head_verts_ndc, head_mesh.faces_packed(), image_size=(W, H))

    # init evaluation
    init_image_orien, init_image_silh, init_image_depth = gl_render.render()
    init_hisa = compute_hisa(ref_img_orien, init_image_orien)
    init_hida = compute_hida(init_image_depth, hida_pairs, hida_labels)
    
    # result evaluation
    result_image_orien, result_image_silh, result_image_depth = gl_render.render(result_strands_ndc)
    result_hisa = compute_hisa(ref_img_orien, result_image_orien)
    result_hida = compute_hida(result_image_depth, hida_pairs, hida_labels)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.tight_layout()
    for ax in axes.flatten():
        ax.set_xticks([]), ax.set_yticks([])
    axes[0, 0].imshow(ref_img_orien.cpu().numpy()), axes[0, 0].set_title("Reference Orientation")
    axes[1, 0].imshow(ref_img_depth.cpu().numpy()), axes[1, 0].set_title("Reference Depth")
    axes[0, 1].imshow(init_image_orien.cpu().numpy()), axes[0, 1].set_title("Hairstep Orientation"), axes[0, 1].set_xlabel(f"HiSa: {init_hisa:.2f}")
    axes[1, 1].imshow(init_image_depth.cpu().numpy()), axes[1, 1].set_title("Hairstep Depth"), axes[1, 1].set_xlabel(f"HiDa: {init_hida*100:.2f}%")
    axes[0, 2].imshow(result_image_orien.cpu().numpy()), axes[0, 2].set_title("Ours Orientation"), axes[0, 2].set_xlabel(f"HiSa: {result_hisa:.2f}")
    axes[1, 2].imshow(result_image_depth.cpu().numpy()), axes[1, 2].set_title("Ours Depth"), axes[1, 2].set_xlabel(f"HiDa: {result_hida*100:.2f}%")
    
    hair_name = os.path.basename(config["ref_img_path"]).split(".")[0]
    fig.savefig(os.path.join(config["output_dir"], hair_name+".png"))
    plt.close(fig)
    
    return init_hisa, init_hida, result_hisa, result_hida


def strands2pc(strands: list | np.ndarray, step: float = 0.001):
    points = []
    colors = []

    for strand in strands:
        distances = np.linalg.norm(np.diff(strand, axis=0), axis=1)
        cum_distances = np.hstack([0, np.cumsum(distances)])
        if cum_distances[-1] < step:
            continue

        t = np.arange(0, cum_distances[-1], step)

        strand_new = np.zeros((t.shape[0], 3))
        strand_new[:, 0] = np.interp(t, cum_distances, strand[:, 0])
        strand_new[:, 1] = np.interp(t, cum_distances, strand[:, 1])
        strand_new[:, 2] = np.interp(t, cum_distances, strand[:, 2])
        
        tangents = np.diff(strand_new, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = np.concatenate([tangents, tangents[-1:]])

        points.append(strand_new)
        colors.append(tangents)

    points = np.concatenate(points)
    colors = np.concatenate(colors)

    return points, colors


def strands2volume(strands: list | np.ndarray, resolution: int = 128):
    # normalize
    min_bound = np.array([-0.3, 1.0, -0.3])
    max_bound = np.array([0.3, 2.0, 0.3])
    step = (max_bound - min_bound) / resolution

    # sample
    points, colors = strands2pc(strands, step=0.5*step.min())
    distances = np.linalg.norm(points - (np.floor(points)+0.5), axis=1)
    weights = np.exp(-0.5 * (distances / 0.4) ** 2)

    # voxelize
    points = ((points - min_bound) / step).astype(np.int16).clip(0, resolution-1)
    voxels = np.zeros((resolution, resolution, resolution, 4), dtype=np.float32)

    np.add.at(voxels[:, :, :, :3], (points[:, 0], points[:, 1], points[:, 2]), colors * weights[:, None])
    voxels[:, :, :, :3] /= np.linalg.norm(voxels[:, :, :, :3], axis=-1, keepdims=True).clip(1e-8)
    voxels[points[:, 0], points[:, 1], points[:, 2], 3] = 1
    
    return voxels


def eval_volume_metric(pred: np.ndarray, gt: np.ndarray):
    """
    Compute accuracy metrics for predicted and ground truth volumes.

    Args:
        pred (np.ndarray): Predicted volume.
        gt (np.ndarray): Ground truth volume.

    Returns:
        tuple: A tuple containing the following accuracy metrics:
            - iou (float): Intersection over Union.
            - precision (float): Precision.
            - recall (float): Recall.
            - error (float): Orientation error.
    """
    # occupancy
    vol_pred = pred[..., 3].astype(bool)
    vol_gt = gt[..., 3].astype(bool)

    union = vol_pred | vol_gt
    inter = vol_pred & vol_gt
    
    iou = inter.sum() / union.sum().clip(1e-8)
    precision = inter.sum() / vol_pred.sum().clip(1e-8)
    recall = inter.sum() / vol_gt.sum().clip(1e-8)
    
    # orientation
    ori_pred = pred[..., :3]
    ori_gt = gt[..., :3]
    error = np.mean((ori_gt[inter] - ori_pred[inter]) ** 2)
    
    return iou, precision, recall, error


def run_hisa_hida_evaluation(config, ref_dir, hairnet_dir, neuralhd_dir, hairstep_dir, ours_dir, output_dir):
    head_path = "X:/hairstep/head_model_metahuman.obj"
    neuralhd_head_path = "X:/neuralhdhair/Bust.obj"
    
    hairnet_2d_metric_list = []
    neuralhd_2d_metric_list = []
    hairstep_2d_metric_list = []
    ours_2d_metric_list = []
    for hair_name in os.listdir(ours_dir):
        config["ref_img_path"] = os.path.join(ref_dir, "resized_img", hair_name+".png")
        config["ref_seg_path"] = os.path.join(ref_dir, "seg", hair_name+".png")
        config["ref_orien_path"] = os.path.join(ref_dir, "strand_map", hair_name+".png")
        config["ref_depth_path"] = os.path.join(ref_dir, "depth_map", hair_name+".npy")
        config["hida_pair_path"] = os.path.join(ref_dir, "relative_depth/pairs", hair_name+".npy")
        config["hida_label_path"] = os.path.join(ref_dir, "relative_depth/labels", hair_name+".npy")
        
        hairnet_hair_path = os.path.join(hairnet_dir, "hair3D_cy", hair_name+".hair")
        neuralhd_hair_path = os.path.join(neuralhd_dir, hair_name, "hair_cy.hair")
        hairstep_hair_path = os.path.join(hairstep_dir, "hair3D/resample_32", hair_name+".hair")
        ours_hair_path = os.path.join(ours_dir, hair_name, "results/full_modifier.hair")
        
        # 2d metrics
        if os.path.exists(ours_hair_path) and not hair_name in skip_list:
            print("Processing", hair_name)
            print("2D metrics")

            # neuralhdhair
            if os.path.isfile(neuralhd_hair_path):
                neuralhd_strands = resample_strands_fast(load_hair(neuralhd_hair_path), 32).to(device)
                config["head_path"] = neuralhd_head_path
                config["camera_path"] = os.path.join(neuralhd_dir, hair_name, "camera.npy")
                iou, hisa, hida = eval_image_metric(config, neuralhd_strands)
                print("NeuralHDHair\t", f"iou: {iou:.4f}, hisa: {hisa:.4f}, hida: {hida:.4f}")
                neuralhd_2d_metric_list.append([iou, hisa, hida])
            
            # hairstep
            if os.path.isfile(hairstep_hair_path):
                hairstep_strands = resample_strands_fast(load_hair(hairstep_hair_path), 32).to(device)
                config["head_path"] = head_path
                config["camera_path"] = os.path.join(hairstep_dir, "param", hair_name+".npy")
                iou, hisa, hida = eval_image_metric(config, hairstep_strands)
                print("Hairstep\t", f"iou: {iou:.4f}, hisa: {hisa:.4f}, hida: {hida:.4f}")
                hairstep_2d_metric_list.append([iou, hisa, hida])
            
            # hairnet
            if os.path.isfile(hairnet_hair_path):
                hairnet_strands = resample_strands_fast(load_hair(hairnet_hair_path), 32).to(device)
                iou, hisa, hida = eval_image_metric(config, hairnet_strands)
                print("HairNet\t\t", f"iou: {iou:.4f}, hisa: {hisa:.4f}, hida: {hida:.4f}")
                hairnet_2d_metric_list.append([iou, hisa, hida])
            
            # ours
            if os.path.isfile(ours_hair_path):
                ours_strands = resample_strands_fast(load_hair(ours_hair_path), 32).to(device)
                iou, hisa, hida = eval_image_metric(config, ours_strands)
                print("Ours\t\t", f"iou: {iou:.4f}, hisa: {hisa:.4f}, hida: {hida:.4f}")
                ours_2d_metric_list.append([iou, hisa, hida])
            
            print()

            # save metrics
            hairnet_metrics = np.array(hairnet_2d_metric_list, dtype=np.float32)
            neuralhd_metrics = np.array(neuralhd_2d_metric_list, dtype=np.float32)
            hairstep_metrics = np.array(hairstep_2d_metric_list, dtype=np.float32)
            ours_metrics = np.array(ours_2d_metric_list, dtype=np.float32)
            os.makedirs(output_dir, exist_ok=True)
            
            for metrics, label in zip([hairnet_metrics, neuralhd_metrics, hairstep_metrics, ours_metrics],
                                      ["2D_HairNet", "2D_NeuralHDHair", "2D_Hairstep", "2D_Ours"]):
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                for i, title in enumerate(["IoU", "HiSa", "HiDa"]):
                    axes[i].hist(metrics[:, i], bins=20)
                    axes[i].set_title(title), axes[i].set_xlabel(f"Mean: {np.mean(metrics[:, i])}")
                fig.savefig(os.path.join(output_dir, f"{label}_metrics.png"))
                plt.close(fig)


def run_synthetic_evaluation(ref_dir, hairnet_dir, neuralhd_dir, hairstep_dir, ours_dir, output_dir):  
    hairnet_3d_metric_list = []
    neuralhd_3d_metric_list = []
    hairstep_3d_metric_list = []
    ours_3d_metric_list = []
    for hair_name in os.listdir(ours_dir):
        ref_hair_path = os.path.join(ref_dir, "hair_gt", hair_name+".hair")
        hairnet_hair_path = os.path.join(hairnet_dir, "hair3D_cy", hair_name+".hair")
        neuralhd_hair_path = os.path.join(neuralhd_dir, hair_name, "hair_cy.hair")
        hairstep_hair_path = os.path.join(hairstep_dir, "hair3D/resample_32", hair_name+".hair")
        ours_hair_path = os.path.join(ours_dir, hair_name, "results/full_modifier.hair")
                
        if not os.path.isfile(ref_hair_path) or hair_name in skip_list:
            continue
        print("Processing", hair_name)
        print("3D metrics")
        
        ref_hair_strands = load_hair(ref_hair_path)
        ref_hair_volume = strands2volume(ref_hair_strands, 128)
        
        # hairnet
        if os.path.isfile(hairnet_hair_path):
            hairnet_strands = resample_strands_fast(load_hair(hairnet_hair_path), 32)
            hairnet_hair_volume = strands2volume(hairnet_strands.cpu().numpy(), 128)
            iou, precision, recall, error = eval_volume_metric(hairnet_hair_volume, ref_hair_volume)
            print("Hairnet\t\t", f"iou: {iou:.4f}, precision: {precision:.4f}, orien error: {error:.4f}")
            hairnet_3d_metric_list.append([iou, precision, error])
        
        # neuralhdhair
        if os.path.isfile(neuralhd_hair_path):
            neuralhd_strands = resample_strands_fast(load_hair(neuralhd_hair_path), 32)
            # transform to align hairstep
            neuralhd_strands = neuralhd_strands * 1.058 + torch.tensor([0.0008, -0.096, -0.0016])
            neuralhd_hair_volume = strands2volume(neuralhd_strands.cpu().numpy(), 128)
            iou, precision, recall, error = eval_volume_metric(neuralhd_hair_volume, ref_hair_volume)
            print("NeuralHDHair\t", f"iou: {iou:.4f}, precision: {precision:.4f}, orien error: {error:.4f}")
            neuralhd_3d_metric_list.append([iou, precision, error])
        
        # hairstep
        if os.path.isfile(hairstep_hair_path):
            hairstep_strands = resample_strands_fast(load_hair(hairstep_hair_path), 32)
            hairstep_hair_volume = strands2volume(hairstep_strands.cpu().numpy(), 128)
            iou, precision, recall, error = eval_volume_metric(hairstep_hair_volume, ref_hair_volume)
            print("Hairstep\t", f"iou: {iou:.4f}, precision: {precision:.4f}, orien error: {error:.4f}")
            hairstep_3d_metric_list.append([iou, precision, error])
        
        # ours
        if os.path.isfile(ours_hair_path):
            ours_strands = resample_strands_fast(load_hair(ours_hair_path), 32)
            ours_hair_volume = strands2volume(ours_strands.cpu().numpy(), 128)
            iou, precision, recall, error = eval_volume_metric(ours_hair_volume, ref_hair_volume)
            print("Ours\t\t", f"iou: {iou:.4f}, precision: {precision:.4f}, orien error: {error:.4f}")
            ours_3d_metric_list.append([iou, precision, error])
        
        print()
        
        # save metrics
        hairnet_metrics = np.array(hairnet_3d_metric_list, dtype=np.float32)
        neuralhd_metrics = np.array(neuralhd_3d_metric_list, dtype=np.float32)
        hairstep_metrics = np.array(hairstep_3d_metric_list, dtype=np.float32)
        ours_metrics = np.array(ours_3d_metric_list, dtype=np.float32)
        os.makedirs(output_dir, exist_ok=True)
        
        for metrics, label in zip([hairnet_metrics, neuralhd_metrics, hairstep_metrics, ours_metrics],
                                    ["3D_HairNet", "3D_NeuralHDHair", "3D_Hairstep", "3D_Ours"]):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for i, title in enumerate(["IoU", "Precision", "Error"]):
                axes[i].hist(metrics[:, i], bins=20)
                axes[i].set_title(title), axes[i].set_xlabel(f"Mean: {np.mean(metrics[:, i])}")
            fig.savefig(os.path.join(output_dir, f"{label}_metrics.png"))
            plt.close(fig)
