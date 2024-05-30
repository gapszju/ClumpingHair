import os
import re
import torch
import pyexr
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms


def read_exr_image(filepath: str):
    image = pyexr.open(filepath)

    render_img = image.get("default")[..., :3]
    depth_img = image.get("depth")[..., :1]
    orien_img = image.get("orientation")[..., :2]
    param_img = image.get("aov")[..., :1]

    render_img = (
            0.299 * render_img[..., 0]
            + 0.587 * render_img[..., 1]
            + 0.114 * render_img[..., 2]
    )

    render_img = torch.tensor(render_img, dtype=torch.float16).unsqueeze(0)
    depth_img = torch.tensor(depth_img, dtype=torch.float16).permute(2, 0, 1)
    orien_img = torch.tensor(orien_img, dtype=torch.float16).permute(2, 0, 1)
    param_img = torch.tensor(param_img, dtype=torch.float16).permute(2, 0, 1)
    
    return [render_img, depth_img, orien_img, param_img]


def parse_params(param_str):
    match = re.search(r"scale_([\d.]+)", param_str)
    if match:
        return float(match.group(1))
    else:
        return -1.0


def crop_to_patch(image, center, size):
    top = center[1] - size // 2
    top = max(top, 0)
    left = center[0] - size // 2
    left = max(left, 0)

    return transforms.functional.crop(image, top, left, size, size)


def crop_to_patch(image, center, size):
    top = center[1] - size // 2
    top = max(top, 0)
    left = center[0] - size // 2
    left = max(left, 0)

    return transforms.functional.crop(image, top, left, size, size)


def check_black_pixel_ratio(image, threshold):
    """Check if the non-black pixel ratio is above the threshold in the image."""
    non_black_pixels = torch.count_nonzero(image.sum(axis=0))
    total_pixels = image.shape[1] * image.shape[2]
    ratio = non_black_pixels / total_pixels
    return ratio > threshold


class RandomCropCheckBlack:
    def __init__(self, size, threshold=0.3, max_attempts=10):
        """
        Initialize the parameters for the random cropping with black pixel check.
        :param size: Desired output size of the crop.
        :param threshold: Minimum non-black pixel ratio in the cropped image.
        :param max_attempts: Maximum number of attempts to find a valid crop.
        """
        self.size = size
        self.threshold = threshold
        self.max_attempts = max_attempts
        self.crop_transform = transforms.RandomCrop(self.size)

    def __call__(self, img):
        """
        Call method to perform the random cropping.
        :param img: PIL Image to be cropped.
        :return: Cropped PIL Image.
        """
        for _ in range(self.max_attempts):
            cropped_image = self.crop_transform(img)
            if check_black_pixel_ratio(cropped_image, self.threshold):
                return cropped_image
        # Return the last attempt if none satisfy the condition
        return cropped_image


class HairDataset(Dataset):
    def __init__(
        self,
        folder_path,
        use_depth=True,
        use_orien=True,
        use_param=False,
        augment=True,
        crop_size=None,
        same_crop=True,
    ):
        self.folder_path = folder_path
        self.use_depth = use_depth
        self.use_orien = use_orien
        self.use_param = use_param
        self.augment = augment
        self.crop_size = crop_size
        self.same_crop = same_crop
        
        self.hair_names = os.listdir(folder_path)

        self.crop = None
        self.render_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomAffine(degrees=(-15, 15)),
            transforms.ColorJitter(contrast=0.5),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
            transforms.RandomAdjustSharpness(sharpness_factor=3.0),
        ])
        self.feature_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomAffine(degrees=(-15, 15)),
        ])
        
        if crop_size:
            self.crop = RandomCropCheckBlack(size=crop_size)

        cache_path = os.path.join(folder_path, "cache.pt")
        params_path = os.path.join(folder_path, "params.pt")
        if os.path.isfile(cache_path) and os.path.isfile(params_path):
            self.images = torch.load(cache_path)
            
            # resize = transforms.Resize((224, 224), antialias=True)
            # for i in range(len(self.images)):
            #     for j in range(len(self.images[i])):
            #         self.images[i][j][0] = resize(self.images[i][j][0])
            #         self.images[i][j][1] = resize(self.images[i][j][1])
            #         self.images[i][j][2] = resize(self.images[i][j][2])
            #         self.images[i][j][3] = resize(self.images[i][j][3])
            
            self.params = torch.load(params_path)

        else:
            # if not cached, load from exr files
            self.images = []
            self.params = []
            for hair_name in self.hair_names:
                data_dir = os.path.join(folder_path, hair_name)
                if not os.path.isdir(data_dir):
                    continue
                
                images = []
                params = []
                for param_str in os.listdir(data_dir):
                    if param_str.endswith(".exr"):
                        print(hair_name, param_str)
                        images += read_exr_image(os.path.join(folder_path, hair_name, param_str)),
                        params += parse_params(os.path.splitext(param_str)[0]),
                self.images.append(images)
                self.params.append(params)
                
            assert all(
                len(imgs) == len(self.images[0]) for imgs in self.images
            ), "The number of images should be the same for all hair strands."
            
            torch.save(self.images, os.path.join(folder_path, "cache.pt"))
            torch.save(self.params, os.path.join(folder_path, "params.pt"))
        
        self.n_hair = len(self.images)
        self.n_classes = len(self.images[0])
        
    def __len__(self):
        return self.n_hair * self.n_classes

    def __getitem__(self, idx):
        gid = idx // self.n_classes
        vid = idx % self.n_classes
        
        render_img, depth_img, orien_img, param_img = self.images[gid][vid]
               
        feature_img = []
        if self.use_depth:
            feature_img.append(depth_img)
        if self.use_orien:
            feature_img.append(orien_img)
        if self.use_param:
            feature_img.append(param_img)
        feature_img = torch.cat(feature_img)
        
        if self.crop is not None:
            if self.same_crop:
                patch = self.crop(torch.cat([render_img, feature_img]))
                render_img = patch[:render_img.shape[0]]
                feature_img = patch[-feature_img.shape[0]:]
            else:
                render_img = self.crop(render_img)
                feature_img = self.crop(feature_img)
        
        if self.augment:
            render_img = self.render_transforms(render_img.float()).half()
            feature_img = self.feature_transforms(feature_img.float()).half()
        img_pair = [render_img.clamp(0, 1), feature_img.clamp(0, 1)]
        
        param = self.params[gid][vid]

        return img_pair, param
    
    def get_val(self, idx):
        images = self.images[idx]
        render_imgs = torch.stack([img[0] for img in images])
        depth_imgs = torch.stack([img[1] for img in images])
        orien_imgs = torch.stack([img[2] for img in images])
        param_imgs = torch.stack([img[3] for img in images])
        
        feature_imgs = []
        if self.use_depth:
            feature_imgs.append(depth_imgs)
        if self.use_orien:
            feature_imgs.append(orien_imgs)
        if self.use_param:
            feature_imgs.append(param_imgs)
        feature_imgs = torch.cat(feature_imgs, dim=1)

        img_pair = [render_imgs, feature_imgs]
        
        params = self.params[idx]

        return img_pair, params


class HairImageDataset(Dataset):
    def __init__(self, folder_path, augment=True):
        # assert use_depth or use_orien, "At least one of depth and orientation should be used."

        self.folder_path = folder_path
        self.augment = augment

        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=(0.5, 2.0), contrast=0.5),
            # transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
            transforms.RandomAdjustSharpness(sharpness_factor=3.0),
        ])

        cache_path = os.path.join(folder_path, "cache.pt")
        params_path = os.path.join(folder_path, "params.pt")
        if os.path.isfile(cache_path) and os.path.isfile(params_path):
            params = torch.load(params_path)
            images = torch.load(cache_path)
            
            self.images = []
            self.params = []
            for i in range(len(images)):
                self.images.append([])
                self.params.append([])
                for j in range(len(images[i])):
                    if params[i][j] < 0:
                        continue
                    self.images[i].append(images[i][j][0])
                    
                    param_img = images[i][j][-1].float()
                    param = param_img.sum()
                    param /= np.count_nonzero(param_img) + 1e-8
                    self.params[i].append(param)
        else:
            raise FileNotFoundError("Cache files not found.")

        self.n_hair = len(self.images)
        self.n_classes = len(self.images[0])

    def __len__(self):
        return self.n_hair * self.n_classes

    def __getitem__(self, idx):
        gid = idx // self.n_classes
        vid = idx % self.n_classes
        
        render_img = self.images[gid][vid]
        if self.augment:
            render_img = self.transforms(render_img.float()).half()
        
        param = self.params[gid][vid]
        
        return render_img.clamp(0, 1), param


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    
    data_path = "X:/contrastive_learning/data/clumping_dataset/real_img_train/test"
    dataset = HairDataset(data_path, augment=True, use_param=True, crop_size=0, same_crop=False)
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

    fig, ax = plt.subplots(2, 4)
    plt.subplots_adjust(bottom=0.1)
    
    def update_images(event=None):
        img_pairs, metrics = next(iter(data_loader))
        render_img, feature_img = img_pairs
        for i in range(4):
            ax[0, i].imshow(render_img[i, 0].cpu().float(), cmap="gray"), ax[0, i].set_title(f"{metrics[i][0].item():.4f}"),  ax[0, i].axis("off")
            ax[1, i].imshow(feature_img[i, 3].cpu().float(), cmap="gray"), ax[1, i].set_title("feature"), ax[1, i].axis("off")
        plt.draw()
    
    button = Button(plt.axes([0.81, 0.05, 0.1, 0.03]), 'Next')
    button.on_clicked(update_images)
    update_images()
    
    plt.show()
