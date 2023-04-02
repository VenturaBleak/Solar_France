import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # Define a sequential model with two convolution layers,
        # followed by batch normalization and ReLU activation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Apply the sequential model to the input x
        return self.conv(x)

class UNET(nn.Module):
    """U-Net model for image segmentation
    replicated from the following paper: https://arxiv.org/abs/1505.04597
    With the help of this yt tutorial: https://www.youtube.com/watch?v=IHq1t7NxS8k
    """
    def __init__(self, in_channels = 3, out_channels = 1, features = [64, 128, 256, 512]):
        super(UNET, self).__init__()
        # Define a list of modules for the upsampling part of the network
        self.ups = nn.ModuleList()
        # Define a list of modules for the downsampling part of the network
        self.downs = nn.ModuleList()
        # Define a max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of the U-Net
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of the U-Net
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # Bottleneck layers, i.e. the lowest resolution layers, i.e. the embedding
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Apply downsampling layers and store skip connections
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Apply bottleneck layers
        x = self.bottleneck(x)
        # Reverse skip_connections list for correct connections
        skip_connections = skip_connections[::-1]

        # Apply upsampling layers and concatenate skip connections
        for idx in range(0, len(self.ups), 2):
            # Apply transpose convolution for upsampling
            x = self.ups[idx](x)
            # Get corresponding skip connection
            skip_connection = skip_connections[idx // 2]

            # Resize x to match skip connection size if necessary
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:], antialias=True)

            # Concatenate skip connection and x along channel dimension
            concat_skip = torch.cat((skip_connection, x), dim=1)

            # Apply double convolution
            x = self.ups[idx + 1](concat_skip)

        # Apply final 1x1 convolution to produce output
        return self.final_conv(x)

#######################
# Segformer
#######################

from math import sqrt
from torch import einsum
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

# helpers
def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

# classes
class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride = reduction_ratio, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return self.to_out(out)

class MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class MiT(nn.Module):
    def __init__(
        self,
        *,
        channels,
        dims,
        heads,
        ff_expansion,
        reduction_ratio,
        num_layers
    ):
        super().__init__()
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))

        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            get_overlap_patches = nn.Unfold(kernel, stride = stride, padding = padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)

            layers = nn.ModuleList([])

            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim = dim_out, heads = heads, reduction_ratio = reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim = dim_out, expansion_factor = ff_expansion)),
                ]))

            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ]))

    def forward(
        self,
        x,
        return_layer_outputs = False
    ):
        h, w = x.shape[-2:]

        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            x = get_overlap_patches(x)

            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h = h // ratio)

            x = overlap_embed(x)
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x

            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret

class Segformer(nn.Module):
    # link to paper: https://arxiv.org/abs/2105.15203
    def __init__(
            self,
            *,
            dims=(32, 64, 160, 256), # changable
            heads=(1, 2, 5, 8),  # fixed
            ff_expansion=(8, 8, 4, 4),  # changable
            reduction_ratio=(8, 4, 2, 1),  # fixed
            num_layers=(2, 2, 2, 2),  # changable
            channels=3,  # fixed
            decoder_dim=256,  # fixed
            num_classes=1  # changable, according to the BCE or CE loss
    ):
        super().__init__()

        # Cast values to tuples if they are not already
        dims = cast_tuple(dims, 4)
        heads = cast_tuple(heads, 4)
        ff_expansion = cast_tuple(ff_expansion, 4)
        reduction_ratio = cast_tuple(reduction_ratio, 4)
        num_layers = cast_tuple(num_layers, 4)

        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio,
                                                 num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels=channels,
            dims=dims,
            heads=heads,
            ff_expansion=ff_expansion,
            reduction_ratio=reduction_ratio,
            num_layers=num_layers
        )

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor=2 ** i)
        ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

        self.upsample = nn.Upsample(size=(416, 416), mode='bilinear', align_corners=False)

    def forward(self, x):
        layer_outputs = self.mit(x, return_layer_outputs=True)

        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim=1)
        seg_out = self.to_segmentation(fused)

        return self.upsample(seg_out)

def create_segformer(arch, channels=3, num_classes=1):
    architectures = {
        'B0': {'num_layers': (2, 2, 2, 2), 'ff_expansion': (8, 8, 4, 4), 'dims': (32, 64, 160, 256)},
        'B1': {'num_layers': (2, 2, 2, 2), 'ff_expansion': (8, 8, 4, 4), 'dims': (64, 128, 320, 512)},
        'B2': {'num_layers': (3, 3, 6, 3), 'ff_expansion': (8, 8, 4, 4), 'dims': (64, 128, 320, 512)},
        'B3': {'num_layers': (3, 3, 18, 8), 'ff_expansion': (8, 8, 4, 4), 'dims': (64, 128, 320, 512)},
        'B4': {'num_layers': (3, 8, 27, 3), 'ff_expansion': (8, 8, 4, 4), 'dims': (64, 128, 320, 512)},
        'B5': {'num_layers': (3, 6, 40, 3), 'ff_expansion': (4, 4, 4, 4), 'dims': (64, 128, 320, 512)},
    }
    arch_params = architectures[arch]
    return Segformer(channels=channels, num_classes=num_classes, num_layers=arch_params['num_layers'], dims=arch_params['dims'], ff_expansion=arch_params['ff_expansion'])

def test():
    # Create a random input tensor of size (3, 1, 161, 161) - 3 images, 1 channel, 161x161 pixels
    x = torch.randn((3, 3, 416, 416))
    # Create a Segformer model with the B0 architecture
    model = create_segformer('B0', channels=3, num_classes=1)
    preds = model(x)
    print(f'Output shape:{preds.shape}')
    # Check if output shape matches input shape
    assert preds.shape == torch.randn((3, 1, 416, 416)).shape

if __name__ == "__main__":
    test()