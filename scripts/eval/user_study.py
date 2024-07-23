import os
import json
import re
import subprocess
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_image(path: str):
    return Image.open(path).convert("RGB")


def gen_user_study(hairstep_dir, neuralhdhair_dir, result_dir, selected_hair, output_dir):
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
        
        question_template = "\n请选择与第一张图像最相似的结果，以及主观感受最真实的结果。(%s)[矩阵单选题](![%s]\(https://szr.faceunity.com/hair/%s.webp.webp\){auto,auto})\nA B C\n最相似\n最真实\n"
        question_mobile_str += question_template % (hair_name[:5], hair_name, hair_name+"-mobile")
        question_pc_str += question_template % (hair_name[:5], hair_name, hair_name+"-pc")
    
    # distribute
    subprocess.run(["scp", output_dir+"/*.webp", "tzs@192.168.16.77:~/hair/"], check=True)
    with open(os.path.join(output_dir, "question.md"), "w") as f:
        f.write("\n===分页===\n")
        f.write(question_mobile_str)
        f.write("\n===分页===\n")
        f.write(question_pc_str)
    with open(os.path.join(output_dir, "label.json"), "w") as f:
        json.dump(label_list, f)
        

def results_statistic(csv_path: str, label_path: str, output_path: str):
    with open(label_path, "r") as f:
        data = json.load(f)
    labels = {name: label for name, label in data}
    
    statistic = {
        "最相似": [],
        "最真实": []
    }

    df = pd.read_csv(csv_path)
    for col in df.columns:
        # parse question
        match = re.search(r'\(([^)]+)\):(\S+)', col)
        if match:
            name = next(full_name for full_name in labels.keys() if match.group(1) in full_name)
            option = match.group(2)
            
            # process column
            res = {"neuralhdhair": 0, "hairstep": 0, "ours": 0}
            label = labels[name]
            for record in df[col]:
                if not np.isnan(record):
                    res[label[int(record)-1]] += 1
            
            # record
            statistic[option].append({"name": name, **res})
    
    df_sim = pd.DataFrame(statistic["最相似"]).groupby("name", sort=False).sum().reset_index()
    sums = df_sim.iloc[:, 1:].sum()
    sums_percent = sums / sums.sum()
    df_sim = pd.concat([df_sim, sums.to_frame().T.assign(name="Total")], ignore_index=True)
    df_sim = pd.concat([df_sim, sums_percent.to_frame().T.assign(name="Percent")], ignore_index=True)
    
    df_real = pd.DataFrame(statistic["最真实"]).groupby("name", sort=False).sum().reset_index()
    sums = df_real.iloc[:, 1:].sum()
    sums_percent = sums / sums.sum()
    df_real = pd.concat([df_real, sums.to_frame().T.assign(name="Total")], ignore_index=True)
    df_real = pd.concat([df_real, sums_percent.to_frame().T.assign(name="Percent")], ignore_index=True)
    
    print("Similarity:\n", df_sim)
    print("\nRealistic:\n", df_real)
    # save
    pd.concat([pd.DataFrame({"name": ["", "Similarity"]}), df_sim,
               pd.DataFrame({"name": ["", "Realistic"]}), df_real]
    ).to_csv(output_path, index=False)


if __name__ == "__main__":
    result_dir = "X:/results/reconstruction/hairstep/Real_Image"
    neuralhdhair_dir = "X:/neuralhdhair/Real_Image"
    hairnet_dir = "X:/hairnet/Real_Image"
    hairstep_dir = "X:/hairstep/Real_Image"
    output_dir = "X:/results/user_study"

    selected_hair = [
        "ann-agterberg-WqATKbXqGZQ-unsplash",
        "christina-wocintechchat-com-0Zx1bDv5BNY-unsplash",
        "rw-studios-mVF9gF7UIqQ-unsplash",
        "eric-karim-cornelis-b3oyT44E3LE-unsplash",
        "dollar-gill-s5Dk_IHgxw4-unsplash",
        "flemming-fuchs-0toSDPvLjhc-unsplash",
        "janko-ferlic-Q13lggdvtVY-unsplash",
        "luke-southern-yyvx_eYqtKY-unsplash",
        "patrick-malleret-p-v1DBkTrgo-unsplash",
        "andre-sebastian-X6aMAzoVJzk-unsplash",
    ]
    # gen_user_study(hairstep_dir, neuralhdhair_dir, result_dir, selected_hair, output_dir)
    
    results_statistic(os.path.join(output_dir, "results.csv"),
                      os.path.join(output_dir, "label.json"),
                      os.path.join(output_dir, "statistic.csv"))