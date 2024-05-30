import os
import glob
import subprocess


def convert(file, time, speed=1):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-t", str(time),
            "-i", file,
            "-vf", f"setpts={1/speed}*PTS",
            "-loop", "0",
            file[: -len(".mp4")] + ".gif",
        ],
        shell=True,
    )


if __name__ == "__main__":
    # file_path = "Z:/制作/TZS/code/hair/hair_dencity_0.1_noise_0.08_orien_radius_8_sigma_1e-05_gamma_0.0001.mp4"
    # convert(file_path, time=10, speed=1)
    
    result_dir = "Z:/制作/TZS/code/hair/guide_spline"
    for file in glob.glob(os.path.join(result_dir, "*.mp4")):
        print(file)
        convert(file, time=5, speed=1)

