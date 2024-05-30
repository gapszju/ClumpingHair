import os, sys
import importlib
import multiprocessing

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)
from HairStep.lib.options import BaseOptions


# change to the HairStep directory and run the function
def run(module_name, func_name, opt):
    hairstep_dir = os.path.join(root_dir, "HairStep")
    os.chdir(hairstep_dir)
    sys.path.insert(0, hairstep_dir)
    
    module = importlib.import_module(module_name)
    getattr(module, func_name)(opt)


if __name__ == "__main__":
    opt = BaseOptions().parse()
    tasks = [
        ("scripts.img2masks", "img2masks"),
        ("scripts.img2strand", "img2strand"),
        ("scripts.img2depth", "img2depth"),
        ("scripts.get_lmk", "get_lmk"),
        ("scripts.opt_cam", "opt_cam"),
        ("scripts.recon3D", "recon3D_from_hairstep"),
    ]
    for module, func in tasks:
        process = multiprocessing.Process(target=run, args=(module, func, opt))
        process.start()
        process.join()
