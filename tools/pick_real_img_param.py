import os
import pyexr
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons


dataset_dir = "X:/contrastive_learning/data/clumping_dataset/real_img_cluster_512"
ref_img_dir = "X:/hairstep/HiSa_HiDa/resized_img"

render_img_list = [os.path.join(dataset_dir, "train", f) for f in os.listdir(os.path.join(dataset_dir, "train"))] + \
                    [os.path.join(dataset_dir, "test", f) for f in os.listdir(os.path.join(dataset_dir, "test"))]
current_index = 0
val_step = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]

render_img_dir = render_img_list[current_index]
ref_img_path = os.path.join(ref_img_dir, os.path.basename(render_img_dir) + ".png")
param_dict = {}


def draw_img(img_path, ax):
    if img_path.endswith(".png"):
        img = plt.imread(img_path)
    else:
        img = pyexr.read(img_path)
    if ax.images:
        ax.images[0].set_array(img)
    else:
        ax.imshow(img)
    ax.axis('off')
    fig.canvas.draw_idle()
    

def update_slider(val):
    global param_dict
    img_path = os.path.join(render_img_dir, f"cluster_1024_scale_{val:.2f}.exr")
    draw_img(img_path, axes[1])
    param_dict[os.path.basename(render_img_dir)] = val
    

def update_image(index_change):
    global current_index, render_img_dir, ref_img_path, param_dict
    current_index += index_change
    if current_index < 0:
        current_index = len(render_img_list) - 1
    elif current_index >= len(render_img_list):
        current_index = 0
        
    render_img_dir = render_img_list[current_index]
    hair_name = os.path.basename(render_img_dir)
    ref_img_path = os.path.join(ref_img_dir, hair_name + ".png")
    draw_img(ref_img_path, axes[0])
    axes[0].set_title(f"Reference Image: {current_index+1}/{len(render_img_list)}")
    
    if hair_name in param_dict:
        slider.set_val(param_dict[hair_name])
    else:
        slider.set_val(0.5)


def on_key(event):
    if event.key == 'up':
        update_image(-1)
    elif event.key == 'down':
        update_image(1)
    elif event.key == 'left':
        slider.set_val(val_step[max(0, val_step.index(slider.val) - 1)])
    elif event.key == 'right':
        slider.set_val(val_step[min(len(val_step)-1, val_step.index(slider.val) + 1)])


fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.2)
axes[0].set_title("Reference Image")
axes[1].set_title("Rendered Image")

slider = Slider(plt.axes([0.55, 0.1, 0.4, 0.03]), 'param', 0, 1, valinit=0.5, valstep=val_step)
slider.on_changed(update_slider)

fig.canvas.mpl_connect('key_press_event', on_key)
update_image(0)
plt.show()

with open("param_dict.json", "w") as f:
    json.dump(param_dict, f, indent=4)