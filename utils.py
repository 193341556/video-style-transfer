import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
import os,shutil


def delete_file(name):
    import shutil
    shutil.rmtree(name)
    os.mkdir(name)


def gram(tensor):
    B, C, H, W = tensor.shape
    x = tensor.view(B, C, H*W)
    x_t = x.transpose(1, 2)
    return  torch.bmm(x, x_t) / (C*H*W)


def load_image(path):
    # Images loaded as BGR
    img = cv2.imread(path)
    return img


def show(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = np.array(img/255).clip(0,1)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.show()

def saveimg(img, image_path):
    img = img.clip(0, 255)
    cv2.imwrite(image_path, img)


def itot(img, max_size=None):

    if (max_size==None):
        itot_t = transforms.Compose([

            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])    
    else:
        H, W, C = img.shape
        image_size = tuple([int((float(max_size) / max([H,W]))*x) for x in [H, W]])
        itot_t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

    tensor = itot_t(img)

    tensor = tensor.unsqueeze(dim=0)
    return tensor


def ttoi(tensor):

    tensor = tensor.squeeze()
    img = tensor.cpu().numpy()

    img = img.transpose(1, 2, 0)
    return img
def transfer_color(src, dest):

    src, dest = src.clip(0, 255), dest.clip(0, 255)

    H, W, _ = src.shape
    dest = cv2.resize(dest, dsize=(W, H), interpolation=cv2.INTER_CUBIC)

    dest = dest.astype(np.uint8)

    dest_gray = cv2.cvtColor(dest, cv2.COLOR_BGR2GRAY)

    src_yiq = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)

    src_yiq[..., 0] = dest_gray

    # Convert new image from YCrCb back to BGR
    return cv2.cvtColor(src_yiq, cv2.COLOR_YCrCb2BGR).clip(0, 255)

def plot_loss_hist(c_loss, s_loss, t_loss,total_loss, title="Loss History"):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    plt.plot(c_loss, label="Content Loss", alpha=0.7, linestyle="--")
    plt.plot(s_loss, label="Style Loss", alpha=0.7, linestyle="-.")
    plt.plot(t_loss, label="Temp Loss", alpha=0.7, linestyle="-")
    plt.plot(total_loss, label="Total Loss", alpha=0.9, linewidth=2)

    min_idx = np.argmin(total_loss)
    plt.scatter(min_idx, total_loss[min_idx],
                color='red', zorder=5,
                label=f"Min Loss: {total_loss[min_idx]:.2f}")

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.savefig("training_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

class ImageFolderWithPaths(datasets.ImageFolder):

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]

        tuple_with_path = (*original_tuple, path)
        return tuple_with_path