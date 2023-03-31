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

def test():
    # Create a random input tensor of size (3, 1, 161, 161) - 3 images, 1 channel, 161x161 pixels
    x = torch.randn((3, 1, 416, 416))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    # Check if output shape matches input shape
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()