import shutil
import os

hair_list = [
    "midas-hofstra-tidSLv-UaNs-unsplash",
    "kate-townsend-3YwfRKDiC-8-unsplash",
    "andre-sebastian-X6aMAzoVJzk-unsplash",
    "luke-southern-yyvx_eYqtKY-unsplash",
    "dollar-gill-s5Dk_IHgxw4-unsplash",
    "eric-karim-cornelis-b3oyT44E3LE-unsplash",
    "janko-ferlic-Q13lggdvtVY-unsplash",
]
ours_dir = "X:/results/reconstruction/hairstep/Real_Image"
neuralhd_dir = "X:/neuralhdhair/Real_Image"

for hair_name in hair_list:
    src_camera = os.path.join(ours_dir, hair_name, "results", "full_modifier_camera.json")
    dst_camera = os.path.join(neuralhd_dir, hair_name, "hair_cy_camera.json")
    print(src_camera, dst_camera)
    shutil.copyfile(src_camera, dst_camera)