import torch
import torch.nn as nn
from torchvision import models, transforms
import utils

class VGG19(nn.Module):
    def __init__(self, vgg_path="vgg19-d01eb7cb.pth"):
        super(VGG19, self).__init__()

        vgg19_features = models.vgg19(pretrained=False)
        vgg19_features.load_state_dict(torch.load(vgg_path), strict=False)
        self.features = vgg19_features.features

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        layers = {'3': 'relu1_2', '8': 'relu2_2', '17': 'relu3_4', '22': 'relu4_2', '26': 'relu4_4', '35': 'relu5_4'}
        features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x

        return features

class VGG16(nn.Module):
    def __init__(self, vgg_path="vgg16-00b39a1b.pth"):
        super(VGG16, self).__init__()
        vgg16_features = models.vgg16(pretrained=False)
        vgg16_features.load_state_dict(torch.load(vgg_path,weights_only=False), strict=False)
        self.features = vgg16_features.features

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        layers = {'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3', '22': 'relu4_3'}
        features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
                if (name=='22'):
                    break

        return features