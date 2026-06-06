import torch
import torch.nn as nn


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def calc_mean_std(self, feat, eps=1e-5):

        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, content_feat, style_mean, style_std):

        content_size = content_feat.size()
        content_mean_calc, content_std_calc = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean_calc.expand(content_size)) / content_std_calc.expand(content_size)
        return normalized_feat * style_std.expand(content_size) + style_mean.expand(content_size)
