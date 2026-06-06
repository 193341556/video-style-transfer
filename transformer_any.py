import torch
import torch.nn as nn

from adain import AdaIN

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm="adain", style_feature_dim=None):
        super(ConvLayer, self).__init__()
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        self.norm_type = norm
        if norm == "instance":
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == "batch":
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)
        elif norm == "adain":
            if style_feature_dim is None:
                raise ValueError("style_feature_dim must be provided for AdaIN normalization in ConvLayer")
            self.adain = AdaIN()
            self.mlp_style_mean = nn.Linear(style_feature_dim, out_channels)
            self.mlp_style_std = nn.Linear(style_feature_dim, out_channels)
        elif norm != "None":
            raise ValueError(f"Unknown norm_type: {norm}")

    def forward(self, x, style_input_for_adain=None):
        x_padded = self.reflection_pad(x)
        x_conv = self.conv_layer(x_padded)

        if self.norm_type == "None":
            out = x_conv
        elif self.norm_type == "adain":
            if style_input_for_adain is None:
                raise ValueError("style_input_for_adain must be provided for AdaIN normalization")
            
            style_mean_params = self.mlp_style_mean(style_input_for_adain)
            style_std_params = self.mlp_style_std(style_input_for_adain)
            
            style_mean = style_mean_params.unsqueeze(-1).unsqueeze(-1)
            style_std = style_std_params.unsqueeze(-1).unsqueeze(-1)
            
            out = self.adain(x_conv, style_mean, style_std)
        else:
            out = self.norm_layer(x_conv)
        return out

class ResidualLayer(nn.Module):
    def __init__(self, channels=128, kernel_size=3, norm="adain", style_feature_dim=None):
        super(ResidualLayer, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride=1, norm=norm, style_feature_dim=style_feature_dim)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride=1, norm=norm, style_feature_dim=style_feature_dim)

    def forward(self, x, style_input_for_adain=None):
        identity = x
        out = self.relu(self.conv1(x, style_input_for_adain=style_input_for_adain))
        out = self.conv2(out, style_input_for_adain=style_input_for_adain)
        out = out + identity
        return out

class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, norm="adain", style_feature_dim=None):
        super(DeconvLayer, self).__init__()
        padding_size = kernel_size // 2
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding_size, output_padding)

        self.norm_type = norm
        if norm == "instance":
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == "batch":
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)
        elif norm == "adain":

            self.adain = AdaIN()
            self.mlp_style_mean = nn.Linear(style_feature_dim, out_channels)
            self.mlp_style_std = nn.Linear(style_feature_dim, out_channels)
        elif norm != "None":
            raise ValueError(f"Unknown norm_type: {norm}")

    def forward(self, x, style_input_for_adain=None):
        x_deconv = self.conv_transpose(x)

        if self.norm_type == "None":
            out = x_deconv
        elif self.norm_type == "adain":
            
            style_mean_params = self.mlp_style_mean(style_input_for_adain)
            style_std_params = self.mlp_style_std(style_input_for_adain)
            
            style_mean = style_mean_params.unsqueeze(-1).unsqueeze(-1)
            style_std = style_std_params.unsqueeze(-1).unsqueeze(-1)
            
            out = self.adain(x_deconv, style_mean, style_std)
        else:  # instance or batch
            out = self.norm_layer(x_deconv)
        return out

class TransformerNetwork(nn.Module):
    def __init__(self, style_feature_dim=512, norm_type="adain"):
        super(TransformerNetwork, self).__init__()
        self.norm_type = norm_type

        self.conv1 = ConvLayer(3, 32, 9, 1, norm=norm_type, style_feature_dim=style_feature_dim)
        self.relu1 = nn.ReLU()
        self.conv2 = ConvLayer(32, 64, 3, 2, norm=norm_type, style_feature_dim=style_feature_dim)
        self.relu2 = nn.ReLU()
        self.conv3 = ConvLayer(64, 128, 3, 2, norm=norm_type, style_feature_dim=style_feature_dim)
        self.relu3 = nn.ReLU()

        self.res_blocks = nn.ModuleList()
        for _ in range(5):
            self.res_blocks.append(ResidualLayer(128, 3, norm=norm_type, style_feature_dim=style_feature_dim))

        self.deconv1 = DeconvLayer(128, 64, 3, 2, 1, norm=norm_type, style_feature_dim=style_feature_dim)
        self.relu4 = nn.ReLU()
        self.deconv2 = DeconvLayer(64, 32, 3, 2, 1, norm=norm_type, style_feature_dim=style_feature_dim)
        self.relu5 = nn.ReLU()
        self.final_conv = ConvLayer(32, 3, 9, 1, norm="None")

    def forward(self, x, style_feat=None):



        def get_style_vec(feat):

            return feat.mean(dim=[2, 3])

        style_vec = get_style_vec(style_feat) if style_feat is not None else None

        x = self.relu1(self.conv1(x, style_input_for_adain=style_vec))
        x = self.relu2(self.conv2(x, style_input_for_adain=style_vec))
        x_conv_out = self.relu3(self.conv3(x, style_input_for_adain=style_vec))

        x_res_out = x_conv_out
        for res_layer in self.res_blocks:
            x_res_out = res_layer(x_res_out, style_input_for_adain=style_vec)

        x = self.relu4(self.deconv1(x_res_out, style_input_for_adain=style_vec))
        x = self.relu5(self.deconv2(x, style_input_for_adain=style_vec))
        out = self.final_conv(x)

        return out

