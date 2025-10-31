import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Helper Block: Core building block of the 3D UNet ---

class ConvBlock3D(nn.Module):
    # Implements a standard 3D Convolutional Block, which is the foundational 
    # component of the 3D U-Net architecture.
    # The block consists of two sequential operations: 
    # Conv3D -> BatchNorm3D -> ReLU, repeated twice. 
    # A 3x3x3 kernel and padding=1 ensure spatial dimensions (D, H, W) are preserved.
    
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock3D, self).__init__()
        
        # nn.Sequential stacks the operations for a clean, single-pass execution
        self.double_conv = nn.Sequential(
            # First Convolution + BN + ReLU
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels), # Normalizes activations across the batch
            nn.ReLU(inplace=True), # Activation function
            
            # Second Convolution + BN + ReLU
            # Input channels here match the output channels of the first conv.
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Defines the forward computation of the ConvBlock3D.
        # Pass the input tensor through the sequential layers
        return self.double_conv(x)

    # --- 3D UNet ---

class UNet3D(nn.Module):
    """
    A simple 3D UNet architecture for volume segmentation.
    (Normal Difficulty Start)
    """
    def __init__(self, in_channels=1, out_classes=6, base_channels=32):

        super(UNet3D, self).__init__()

        # Encoder (Downsampling path)

        # Level 1: Initial feature extraction. 
        self.down1 = ConvBlock3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Level 2: Deeper layer, double the channel to increase 
        # the models capacity to represent and extract complex 
        # when spartial resolution decreases. 
        self.down2 = ConvBlock3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Level 3: Even deeper level, double base channels in the 
        # convolution blocks. 
        self.down3 = ConvBlock3D(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Deepest layer - bottleneck layer which captures the most abstract, 
        # low resolution information. 
        self.bottleneck = ConvBlock3D(base_channels * 4, base_channels * 8)



