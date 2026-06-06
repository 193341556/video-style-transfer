import torch
import utils
import transformer_any as transformer
import os
from torchvision import transforms
import time
import cv2
import numpy as np
import vgg
#STYLE_IMAGE_PATH = "images/night.jpg"


PRESERVE_COLOR = False


def read_flo_file(filename):

    with open(filename, 'rb') as f:

        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise ValueError("Magic number incorrect. Invalid .flo file.")


        w, h = np.fromfile(f, np.int32, count=2)


        data = np.fromfile(f, np.float32, count=2 * w * h)
        data2D = np.resize(data, (h, w, 2))

        return data2D

def warp_image(src, flow):
    h, w = flow.shape[:2]
    flow = flow.astype(np.float32)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    flow[:, :, 0] += x
    flow[:, :, 1] += y
    warped_image = cv2.remap(src, flow[:, :, 0], flow[:, :, 1], cv2.INTER_LINEAR)
    return warped_image


def calculate_smooth_weight(flow, threshold=0.5):
  
    flow_magnitude = np.linalg.norm(flow, axis=2)
    smooth_weight = np.where(flow_magnitude < threshold, 1.0, 0.0)
    smooth_weight = smooth_weight[..., np.newaxis]
    return smooth_weight

def stylize_folder_single(STYLE_IMAGE_PATH,style_path, content_folder, save_folder):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    net = transformer.TransformerNetwork(style_feature_dim=512, norm_type="adain")
    net.load_state_dict(torch.load(style_path, weights_only=False))
    net = net.to(device)

  
    style_image = utils.load_image(STYLE_IMAGE_PATH)
    style_tensor = utils.itot(style_image).to(device)
    imagenet_neg_mean = torch.tensor([-103.939, -116.779, -123.68], dtype=torch.float32).view(1, 3, 1, 1).to(device)

    vgg_net = vgg.VGG16().to(device).eval()
    with torch.no_grad():
        style_features = vgg_net(style_tensor.add(imagenet_neg_mean))
        style_input_for_adain = torch.mean(style_features['relu4_3'], dim=[2, 3])

    images = [img for img in os.listdir(content_folder) if img.endswith(".jpg")]
    images.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    flow_files = [f for f in sorted(os.listdir("flow/")) if f.endswith(".flo")]

    with torch.no_grad():
        prev_generated_image = None
        for idx, image_name in enumerate(images):
            content_image = utils.load_image(os.path.join(content_folder, image_name))
            content_tensor = utils.itot(content_image).to(device)
       
            generated_tensor = net(content_tensor, style_feat=style_features['relu4_3'])
            generated_image = utils.ttoi(generated_tensor.detach())


            if idx > 0:
             
                flow = read_flo_file(os.path.join("flow/", flow_files[idx - 1]))

    
                prev_warped = warp_image(prev_generated_image, flow)


            utils.saveimg(generated_image, os.path.join(save_folder, image_name))

            prev_generated_image = generated_image




# stylize()
