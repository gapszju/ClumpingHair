import os
import sys
import shutil
import pyexr
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def add_background(image, bg_color=(192, 192, 192, 255)):
    bg = Image.new("RGBA", image.size, bg_color)
    return Image.alpha_composite(bg, image).convert("RGB")


def read_image(path: str, with_alpha=False):
    # open image
    if path.endswith(".exr"):
        img = pyexr.read(path)
        img = img.get("default")
        img = Image.fromarray((img * 255).astype(np.uint8))
    else:
        img = Image.open(path)
    
    # alpha processing
    if with_alpha:
        img = img.convert("RGBA")
    else:
        if img.mode == "RGBA":
            add_background(img)
        img = img.convert("RGB")
    
    return img


def copy_origin_result(src_dir, dst_dir):
    for subdir in os.listdir(dst_dir):
        if not os.path.isdir(os.path.join(dst_dir, subdir, "render")):
            continue
        print("Processing", subdir)
        shutil.copy(os.path.join(src_dir, subdir, "render", "render_origin_front.png"),
                    os.path.join(dst_dir, subdir, "render", "render_origin_front.png"))
        shutil.copy(os.path.join(src_dir, subdir, "render", "render_origin_side_L.png"),
                    os.path.join(dst_dir, subdir, "render", "render_origin_side_L.png"))
        shutil.copy(os.path.join(src_dir, subdir, "render", "render_origin_side_R.png"),
                    os.path.join(dst_dir, subdir, "render", "render_origin_side_R.png"))


def compare_hairstep(input_dir, output_dir):
    for subdir in os.listdir(input_dir):
        if not os.path.exists(os.path.join(input_dir, subdir, "vis.png")):
            continue
        print("Processing", subdir)
    
        vis_img = read_image(os.path.join(input_dir, subdir, "vis.png"))
        ref_img = vis_img.crop((0, 0, 512, 512)).resize((1024, 1024))
        orientation_img = vis_img.crop((512, 0, 1024, 512)).resize((1024, 1024))
    
        render_dir = os.path.join(input_dir, subdir, "render")
        origin_front_img = read_image(os.path.join(render_dir, "render_origin_front.png"))
        origin_side_L_img = read_image(os.path.join(render_dir, "render_origin_side_L.png"))
        origin_side_R_img = read_image(os.path.join(render_dir, "render_origin_side_R.png"))
        result_front_img = read_image(os.path.join(render_dir, "render_modified_front.png"))
        result_side_L_img = read_image(os.path.join(render_dir, "render_modified_side_L.png"))
        result_side_R_img = read_image(os.path.join(render_dir, "render_modified_side_R.png"))
    
        ref_group = np.concatenate([ref_img, orientation_img], 0)
        origin_group = np.concatenate([origin_front_img, origin_side_L_img, origin_side_R_img], 1)
        result_group = np.concatenate([result_front_img, result_side_L_img, result_side_R_img], 1)
    
        final = np.concatenate([ref_group, np.concatenate([origin_group, result_group], 0)], 1)
    
        plt.imsave(os.path.join(output_dir, subdir + ".png"), final)


def compare_hairstep2(input_dir, input2_dir, output_dir):
    for subdir in os.listdir(input_dir):
        if not os.path.exists(os.path.join(input_dir, subdir, "vis.png")):
            continue
        print("Processing", subdir)
    
        vis_img = read_image(os.path.join(input_dir, subdir, "vis.png"))
        ref_img = np.array(vis_img.crop((0, 0, 512, 512)).resize((1024, 1024)))[..., :3] / 255.0
        orientation_img = np.array(vis_img.crop((512, 0, 1024, 512)).resize((1024, 1024)))[..., :3] / 255.0
    
        render_dir = os.path.join(input_dir, subdir, "render")
        render2_dir = os.path.join(input2_dir, subdir, "render")
        origin_front_img = plt.imread(os.path.join(render2_dir, "render_modified_front.png"))[..., :3]
        origin_side_L_img = plt.imread(os.path.join(render2_dir, "render_modified_side_L.png"))[..., :3]
        origin_side_R_img = plt.imread(os.path.join(render2_dir, "render_modified_side_R.png"))[..., :3]
        result_front_img = plt.imread(os.path.join(render_dir, "render_modified_front.png"))[..., :3]
        result_side_L_img = plt.imread(os.path.join(render_dir, "render_modified_side_L.png"))[..., :3]
        result_side_R_img = plt.imread(os.path.join(render_dir, "render_modified_side_R.png"))[..., :3]
    
        ref_group = np.concatenate([ref_img, orientation_img], 0)
        origin_group = np.concatenate([origin_front_img, origin_side_L_img, origin_side_R_img], 1)
        result_group = np.concatenate([result_front_img, result_side_L_img, result_side_R_img], 1)
    
        final = np.concatenate([ref_group, np.concatenate([origin_group, result_group], 0)], 1)
    
        plt.imsave(os.path.join(output_dir, subdir + ".png"), final.clip(0, 1))


def screen_blend(img1, img2):
    """Applies screen blend mode to two images."""
    # Convert images to float type for proper division
    img1 = np.array(img1).astype(np.float32) / 255
    img2 = np.array(img2).astype(np.float32) / 255 / 2
    
    # Apply the screen blend formula
    result = 1 - (1 - img1) * (1 - img2)
    
    # Convert back to 8-bit integer type
    result = (result * 255).astype(np.uint8)
    return Image.fromarray(result)


def teaser(input_dir, output_dir):
    for subdir in os.listdir(input_dir):
        if not os.path.exists(os.path.join(input_dir, subdir, "render")):
            continue
        if subdir not in  [
            "4",
            "janko-ferlic-GWFffQS5eWU-unsplash",
        ]:
            continue
        print("Processing", subdir)
        
        separator_subv = np.ones((1024, 2, 3), dtype=np.uint8) * 255
        separator_subh = np.ones((2, 512, 3), dtype=np.uint8) * 255
        separator = np.ones((1024, 10, 3), dtype=np.uint8) * 255
    
        ref_img = read_image(os.path.join(input_dir, subdir, "reference.png")).resize((1024, 1024))
        ref_img_dark = Image.fromarray(np.array(ref_img) // 2).convert("RGBA")
    
        render_dir = os.path.join(input_dir, subdir, "render")
        origin_front_img = read_image(os.path.join(render_dir, "render_origin_front.png"))
        origin_side_L_img = read_image(os.path.join(render_dir, "render_origin_side_L.png")).resize((512, 512))
        origin_side_R_img = read_image(os.path.join(render_dir, "render_origin_side_R.png")).resize((512, 512))
        origin_side_img = np.concatenate([origin_side_R_img.crop((0,0,512,511)), separator_subh, origin_side_L_img.crop((0,1,512,512))], 0)
        
        result_front_img = read_image(os.path.join(render_dir, "render_modified_front.png"))
        result_side_L_img = read_image(os.path.join(render_dir, "render_modified_side_L.png")).resize((512, 512))
        result_side_R_img = read_image(os.path.join(render_dir, "render_modified_side_R.png")).resize((512, 512))
        result_side_img = np.concatenate([result_side_R_img.crop((0,0,512,511)), separator_subh, result_side_L_img.crop((0,1,512,512))], 0)
        
        proj_dir = os.path.join(input_dir, subdir, "projection")
        origin_proj_img = read_image(os.path.join(proj_dir, "origin.png"), with_alpha=True)
        origin_proj_img = Image.alpha_composite(ref_img_dark, origin_proj_img).convert("RGB")
        result_proj_img = read_image(os.path.join(proj_dir, "result.png"), with_alpha=True)
        result_proj_img = Image.alpha_composite(ref_img_dark, result_proj_img).convert("RGB")

        # group1 = np.concatenate([ref_img, np.concatenate([origin_proj_img, result_proj_img], 1)], 0)
        # group2 = np.concatenate([origin_front_img, np.concatenate([origin_side_L_img, origin_side_R_img], 1)], 0)
        # group3 = np.concatenate([result_front_img, np.concatenate([result_side_L_img, result_side_R_img], 1)], 0)
        # group1 = np.concatenate([origin_front_img, separator_subv, origin_side_img], 1)
        # group2 = np.concatenate([result_front_img, separator_subv, result_side_img], 1)
    
        final = np.concatenate([ref_img, separator, origin_front_img, separator, origin_proj_img, separator, result_front_img, separator, result_proj_img], 1)
    
        plt.imsave(os.path.join(output_dir, subdir + ".png"), final)


def collage_our_result(input_dir, output_path):
    final = []
    for hair_name in [
        "joao-paulo-de-souza-oliveira-x-FNmzxyQ94-unsplash",
        "aiony-haust-rhVeNHHNbdk-unsplash",
        "jurica-koletic-7YVZYZeITc8-unsplash",
        "antonio-friedemann-HKp_HmxUrf8-unsplash",
        "22951248a9eedf048adc582ccb04a058",
        "nickolas-nikolic-87d56FlCOyI-unsplash",
        
        # "jurica-koletic-7YVZYZeITc8-unsplash",
        # "dollar-gill-s5Dk_IHgxw4-unsplash",
        # "flemming-fuchs-0toSDPvLjhc-unsplash",
        # "frank-uyt-den-bogaard-NhLQgL8NQ2A-unsplash",
        # "halil-ibrahim-cetinkaya-WzGC8xSyqfg-unsplash",
        # "ving-cam-ixXEGfWBoQY-unsplash",
    ]:
        print("Processing", hair_name)

        ref_img = read_image(os.path.join(input_dir, hair_name, "reference.png")).resize((1024, 1024))
        ref_img_dark = Image.fromarray(np.array(ref_img) // 2).convert("RGBA")

        render_dir = os.path.join(input_dir, hair_name, "render")
        origin_front_img = read_image(os.path.join(render_dir, "render_origin_front.png"))
        result_front_img = read_image(os.path.join(render_dir, "render_modified_front.png"))
        result_side_L_img = read_image(os.path.join(render_dir, "render_modified_side_L.png"))
        result_side_R_img = read_image(os.path.join(render_dir, "render_modified_side_R.png"))
        result_side_img = np.concatenate([result_side_L_img, result_side_R_img], 0)
        
        proj_dir = os.path.join(input_dir, hair_name, "projection")
        result_proj_img = read_image(os.path.join(proj_dir, "result.png"), with_alpha=True)
        result_proj_img = Image.alpha_composite(ref_img_dark, result_proj_img).convert("RGB")
        
        separator_v = np.ones((1024, 10, 3), dtype=np.uint8) * 255
        separator_h = np.ones((10, 1024*5+40, 3), dtype=np.uint8) * 255
        
        group = np.concatenate([ref_img, separator_v, result_proj_img, separator_v, result_front_img, separator_v, result_side_L_img, separator_v, result_side_R_img], 1)
        final += [group, separator_h]
    
    final = np.concatenate(final[:-1], 0)

    plt.imsave(output_path, final)


def xgen_data_compare(input_dir, output_path):
    final = []
    for hair_name in [
        "DD_0916_03_zhongchang_hair027",
        "DD_0916_03_zhongchang_hair012",
        "DD_0913_01_zhongchang_hair001",
    ]:
        print("Processing", hair_name)
        
        data_dir = os.path.join(input_dir, hair_name)
        ref_img = read_image(os.path.join(data_dir, "reference_origin_side.png"))
        init_img = read_image(os.path.join(data_dir, "optim_result_origin_side.png"))
        regression_img = read_image(os.path.join(data_dir, "regression_result_modified_side.png"))
        optim_img = read_image(os.path.join(data_dir, "optim_result_modified_side.png"))
        
        separator_v = np.ones((1024, 10, 3), dtype=np.uint8) * 255
        separator_h = np.ones((20, 1024*4+30, 3), dtype=np.uint8) * 255
        
        group = np.concatenate([ref_img,
                                separator_v,init_img,
                                separator_v,regression_img,
                                separator_v,optim_img], 1)
        final += [group, separator_h]
    
    final = np.concatenate(final[:-1], 0)

    plt.imsave(output_path, final)


def show_dataset(input_dir, output_path):
    final = []
    for hair_name in [
        # "e71d076dbf77985f533ef08cac3b22fb",
        # "e0a8de237f49acd7dca68a5ac2787596",
        "deb4c9ded14a7f2ae324e5b80b4f6b83",
        "c55cb211f1be6a9b8db70849d8551af0",
        "933cdedc9eba95403e43187b0671cb9e",
        # "6bd09eed455592da4799ef7264377f9b",
    ]:
        print("Processing", hair_name)

        data_dir = os.path.join(input_dir, hair_name)
        # img = pyexr.open(os.path.join(data_dir, "cluster_1024_scale_0.00.exr"))
        # img0 = img.get("default")[..., :3]
        # img1 = np.concatenate([img.get("depth.Z"), img.get("orientation")], -1)[..., :3].clip(0, 1)
        
        # img = pyexr.open(os.path.join(data_dir, "cluster_1024_scale_0.55.exr"))
        # img2 = img.get("default")[..., :3]
        # img3 = np.concatenate([img.get("depth.Z"), img.get("orientation")], -1)[..., :3].clip(0, 1)
        
        # img = pyexr.open(os.path.join(data_dir, "cluster_1024_scale_0.90.exr"))
        # img4 = img.get("default")[..., :3]
        # img5 = np.concatenate([img.get("depth.Z"), img.get("orientation")], -1)[..., :3].clip(0, 1)
        
        img0 = read_image(os.path.join(data_dir, "cluster_1024_scale_0.00.png")).resize((512, 512))
        img1 = read_image(os.path.join(data_dir, "cluster_1024_scale_0.40.png")).resize((512, 512))
        img2 = read_image(os.path.join(data_dir, "cluster_1024_scale_0.50.png")).resize((512, 512))
        img3 = read_image(os.path.join(data_dir, "cluster_1024_scale_0.60.png")).resize((512, 512))
        img4 = read_image(os.path.join(data_dir, "cluster_1024_scale_0.70.png")).resize((512, 512))
        img5 = read_image(os.path.join(data_dir, "cluster_1024_scale_0.90.png")).resize((512, 512))

        separator_v = np.ones((512, 10, 3), dtype=np.uint8) * 255
        separator_h = np.ones((10, 512*6+50, 3), dtype=np.uint8) * 255
        
        group = np.concatenate([img0, separator_v, img1, separator_v, img2, separator_v, img3, separator_v, img4, separator_v, img5], 1)
        final += [group, separator_h]
    
    final = np.concatenate(final[:-1], 0)

    plt.imsave(output_path, final)


def compare_with_others(hairstep_dir, neuralhdhair_dir, hairnet_dir, result_dir, selected_hair, output_path):
    final = []
    for hair_name in selected_hair:
        print("Processing", hair_name)

        # reference image
        ref_img = read_image(os.path.join(hairstep_dir, "resized_img", hair_name+".png")).resize((1024, 1024))
        ref_img_dark = Image.fromarray(np.array(ref_img) // 2).convert("RGBA")
        ref_mask = read_image(os.path.join(hairstep_dir, "seg", hair_name+".png")).resize((1024, 1024))
        ref_img = np.array(ref_img)[..., :3]
        ref_mask = np.array(ref_mask) > 128
        ref_img_with_mask = ref_img.copy()
        ref_img_with_mask[~ref_mask] //= 2
        
        # hairnet
        render_dir = os.path.join(hairnet_dir, "render")
        hairnet_img = read_image(os.path.join(render_dir, hair_name+".png"))
        hairnet_proj_img = read_image(os.path.join(render_dir, "projection", hair_name+".png"), with_alpha=True)
        hairnet_proj_img = Image.alpha_composite(ref_img_dark, hairnet_proj_img).convert("RGB").resize((512, 512))
    
        # neuralHDhair
        render_dir = os.path.join(neuralhdhair_dir, hair_name, "render")
        neuralhdhair_img = read_image(os.path.join(render_dir, "render_origin_front.png"))
        neuralhdhair_proj_img = read_image(os.path.join(render_dir, "projection.png"), with_alpha=True)
        neuralhdhair_proj_img = Image.alpha_composite(ref_img_dark, neuralhdhair_proj_img).convert("RGB").resize((512, 512))
        
        # hairstep and ours
        render_dir = os.path.join(result_dir, hair_name, "render")
        hairstep_img = read_image(os.path.join(render_dir, "render_origin_front.png"))
        ours_img = read_image(os.path.join(render_dir, "render_modified_front.png"))

        proj_dir = os.path.join(result_dir, hair_name, "projection")
        hairstep_proj_img = read_image(os.path.join(proj_dir, "origin.png"), with_alpha=True)
        hairstep_proj_img = Image.alpha_composite(ref_img_dark, hairstep_proj_img).convert("RGB").resize((512, 512))
        ours_proj_img = read_image(os.path.join(proj_dir, "result.png"), with_alpha=True)
        ours_proj_img = Image.alpha_composite(ref_img_dark, ours_proj_img).convert("RGB").resize((512, 512))
        
        # separator
        separator_v = np.ones((1024, 10, 3), dtype=np.uint8) * 255
        separator_h = np.ones((10, 1024+(1024+512+10)*4, 3), dtype=np.uint8) * 255
        blank = np.ones((512, 512, 3), dtype=np.uint8) * 255
        
        # combine
        group = np.concatenate([ref_img,
                                separator_v, hairnet_img, np.concatenate([blank, hairnet_proj_img], 0),
                                separator_v, neuralhdhair_img, np.concatenate([blank, neuralhdhair_proj_img], 0),
                                separator_v, hairstep_img, np.concatenate([blank, hairstep_proj_img], 0),
                                separator_v, ours_img, np.concatenate([blank, ours_proj_img], 0)], 1)
        final += [group, separator_h]
    
    final = np.concatenate(final[:-1], 0)

    plt.imsave(output_path, final)


def user_study(hairstep_dir, neuralhdhair_dir, result_dir, selected_hair, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    question_mobile_str = ""
    question_pc_str = ""
    label_list = []
    for hair_name in selected_hair:
        print("Processing", hair_name)

        # reference image
        ref_img = read_image(os.path.join(hairstep_dir, "resized_img", hair_name+".png")).resize((1024, 1024))

        # neuralHDhair
        render_dir = os.path.join(neuralhdhair_dir, hair_name, "render")
        neuralhdhair_front = read_image(os.path.join(render_dir, "render_origin_front.png"))
        neuralhdhair_side_L = read_image(os.path.join(render_dir, "render_origin_side_L.png")).resize((512, 512))
        neuralhdhair_side_R = read_image(os.path.join(render_dir, "render_origin_side_R.png")).resize((512, 512))
        
        # hairstep and ours
        render_dir = os.path.join(result_dir, hair_name, "render")
        hairstep_front = read_image(os.path.join(render_dir, "render_origin_front.png"))
        hairstep_side_L = read_image(os.path.join(render_dir, "render_origin_side_L.png")).resize((512, 512))
        hairstep_side_R = read_image(os.path.join(render_dir, "render_origin_side_R.png")).resize((512, 512))
        
        ours_front = read_image(os.path.join(render_dir, "render_modified_front.png"))
        ours_side_L = read_image(os.path.join(render_dir, "render_modified_side_L.png")).resize((512, 512))
        ours_side_R = read_image(os.path.join(render_dir, "render_modified_side_R.png")).resize((512, 512))
        
        # groups
        names = ["neuralhdhair", "hairstep", "ours"]
        groups_mobile = [
            np.concatenate([neuralhdhair_front, np.concatenate([neuralhdhair_side_L, neuralhdhair_side_R], 0)], 1),
            np.concatenate([hairstep_front, np.concatenate([hairstep_side_L, hairstep_side_R], 0)], 1),
            np.concatenate([ours_front, np.concatenate([ours_side_L, ours_side_R], 0)], 1),
        ]
        groups_pc = [
            np.concatenate([neuralhdhair_front, np.concatenate([neuralhdhair_side_L, neuralhdhair_side_R], 1)], 0),
            np.concatenate([hairstep_front, np.concatenate([hairstep_side_L, hairstep_side_R], 1)], 0),
            np.concatenate([ours_front, np.concatenate([ours_side_L, ours_side_R], 1)], 0),
        ]
        idxs = np.random.permutation(3)
        
        # mobile
        separator_h = np.ones((10, 1024+512, 3), dtype=np.uint8) * 255
        blank = np.ones((1024, 512, 3), dtype=np.uint8) * 255
        result_mobile = sum([[separator_h, groups_mobile[i]] for i in idxs], [])
        result_mobile = np.concatenate([np.concatenate([ref_img, blank], 1)] + [separator_h]*5 + result_mobile, 0)

        final_mobile = Image.new("RGB", (result_mobile.shape[1]+120, result_mobile.shape[0]), "white")
        final_mobile.paste(Image.fromarray(result_mobile), (120, 0))
        draw = ImageDraw.Draw(final_mobile)
        font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), "arialbd.ttf"), size=80)
        for i, name in enumerate(["A", "B", "C"], 1):
            draw.text((0, 1024*i+512), name, fill="gray", font=font)
        
        # pc
        separator_v = np.ones((1024+512, 10, 3), dtype=np.uint8) * 255
        blank = np.ones((512, 1024, 3), dtype=np.uint8) * 255
        result_pc = sum([[separator_v, groups_pc[i]] for i in idxs], [])
        result_pc = np.concatenate([np.concatenate([ref_img, blank], 0)] + [separator_v]*5 + result_pc, 1)
        final_pc = Image.fromarray(result_pc)
        
        # save
        final_mobile.save(os.path.join(output_dir, hair_name+".mobile.webp"), lossless=True)
        final_mobile.save(os.path.join(output_dir, hair_name+"-mobile.webp.webp"), quality=90)
        final_pc.save(os.path.join(output_dir, hair_name+".pc.webp"), lossless=True)
        final_pc.save(os.path.join(output_dir, hair_name+"-pc.webp.webp"), quality=90)
        label_list.append((hair_name, [names[i] for i in idxs]))
        
        question_template = "\n请选择与下面图像最相似和最真实的结果。(%s)[矩阵单选题](![%s]\(https://szr.faceunity.com/hair/%s.webp.webp\){auto,auto})\nA B C\n最相似\n最真实\n"
        question_mobile_str += question_template % (hair_name[:5], hair_name, hair_name+"-mobile")
        question_pc_str += question_template % (hair_name[:5], hair_name, hair_name+"-pc")
    
    subprocess.run(["scp", output_dir+"/*.webp", "tzs@192.168.16.77:~/hair/"], check=True)
    with open(os.path.join(output_dir, "question.md"), "w") as f:
        f.write("\n===分页===\n")
        f.write(question_mobile_str)
        f.write("\n===分页===\n")
        f.write(question_pc_str)
    with open(os.path.join(output_dir, "label.json"), "w") as f:
        json.dump(label_list, f)


if __name__ == "__main__":
    # input_dir = "X:/differential_rendering/full_pipline/Apr20_16-40-31_ROG_with_optim_dist2/Real_Image2_smooth"
    # input2_dir = "X:/differential_rendering/full_pipline/Apr20_16-40-31_ROG_with_optim_dist2/Real_Image2_smooth2"
    # output_dir = "X:/results/compare"
    # os.makedirs(output_dir, exist_ok=True)
    # compare_hairstep(input_dir, output_dir)

    # src_dir = "X:/differential_rendering/full_pipline/Apr20_16-40-31_ROG_with_optim_dist2/Real_Image"
    # dst_dir = "X:/differential_rendering/full_pipline/May05_02-47-38_ROG_cluster_1024_use_full_map/Real_Image"
    # copy_origin_result(src_dir, dst_dir)

    # input_dir = "X:/results/reconstruction/hairstep/Real_Image"
    # output_dir = "X:/results/teaser"
    # os.makedirs(output_dir, exist_ok=True)
    # # teaser(input_dir, output_dir)
    # Image.open(os.path.join(output_dir, "4.png")).convert("RGB").save(
    #     "C:/Users/tangz/Desktop/research/code/hair_modeling/doc/single-view-3d-hair-modeling/results/teaser.jpg",
    #     quality=100,
    # )
    # Image.open( os.path.join(output_dir, "janko-ferlic-GWFffQS5eWU-unsplash.png")).convert("RGB").save(
    #     "C:/Users/tangz/Desktop/research/code/hair_modeling/doc/single-view-3d-hair-modeling/results/teaser2.jpg",
    #     quality=100,
    # )

    input_dir = "X:/results/reconstruction/hairstep/Real_Image"
    output_path = "X:/results/final_results.png"
    collage_our_result(input_dir, output_path)
    Image.open(output_path).convert("RGB").save("C:/Users/tangz/Desktop/research/code/hair_modeling/doc/single-view-3d-hair-modeling/results/final_results.jpg", quality=95)

    # input_dir = "X:/results/clumping_validation"
    # output_path = "X:/results/xgen_data_compare.png"
    # xgen_data_compare(input_dir, output_path)

    # input_dir = "X:/contrastive_learning/data/clumping_dataset/sample_dataset"
    # output_path = "X:/results/snet_dataset.png"
    # show_dataset(input_dir, output_path)
    # Image.open(output_path).convert("RGB").save("C:/Users/tangz/Desktop/research/code/hair_modeling/doc/single-view-3d-hair-modeling/body/snet_dataset.jpg", quality=95)

    # result_dir = "X:/results/reconstruction/hairstep/Real_Image"
    # neuralhdhair_dir = "X:/neuralhdhair/Real_Image"
    # hairnet_dir = "X:/hairnet/Real_Image"
    # hairstep_dir = "X:/hairstep/Real_Image"
    # output_path = "X:/results/comparisons.png"
    # selected_hair = [
    #     "halil-ibrahim-cetinkaya-WzGC8xSyqfg-unsplash",
    #     "midas-hofstra-tidSLv-UaNs-unsplash",
    #     "behrouz-sasani-5cDg40slYoc-unsplash",
    #     "hosein-sediqi-sBkSyfPzakI-unsplash",
    #     "kate-townsend-3YwfRKDiC-8-unsplash",
    # ]
    # compare_with_others(hairstep_dir, neuralhdhair_dir, hairnet_dir, result_dir, selected_hair, output_path)

    # selected_hair = [
    #     "ann-agterberg-WqATKbXqGZQ-unsplash",
    #     "christina-wocintechchat-com-0Zx1bDv5BNY-unsplash",
    #     "rw-studios-mVF9gF7UIqQ-unsplash",
    #     "eric-karim-cornelis-b3oyT44E3LE-unsplash",
    #     "dollar-gill-s5Dk_IHgxw4-unsplash",
    #     "flemming-fuchs-0toSDPvLjhc-unsplash",
    #     "janko-ferlic-Q13lggdvtVY-unsplash",
    #     "luke-southern-yyvx_eYqtKY-unsplash",
    #     "patrick-malleret-p-v1DBkTrgo-unsplash",
    #     "andre-sebastian-X6aMAzoVJzk-unsplash",
    # ]
    # output_dir = "X:/results/user_study"
    # user_study(hairstep_dir, neuralhdhair_dir, result_dir, selected_hair, output_dir)
