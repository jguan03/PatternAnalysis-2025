%%writefile module.py 

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # Pass the input tensor through the sequential layers. 
        return self.double_conv(x)

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

        #Decoder (Expanding path)


        # Level 3: Upsample from the Bottleneck. 
        # First upsampling step: from bottleneck to level 3. 
        # ConvTranspose3d doubles the spatial dimensions (D, H, W) and halves the channels.
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv3 = ConvBlock3D(base_channels * 8, base_channels * 4) # Input is concatenation of up3 + down3

        # Level 2: Second upsampling step
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv2 = ConvBlock3D(base_channels * 4, base_channels * 2) # Input is concatenation of up2 + down2

        # Level 1: Final upsampling to near-original input size
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBlock3D(base_channels * 2, base_channels) # Input is concatenation of up1 + down1

        # Final 1x1x1 convolution maps the final feature depth (base_channels) 
        # to the required number of output segmentation classes.
        self.final_conv = nn.Conv3d(base_channels, out_classes, kernel_size=1)


    def forward(self, x):

        # Encoder
        x1 = self.down1(x) # -> skip connection 1
        x = self.pool1(x1)

        x2 = self.down2(x) # -> skip connection 2
        x = self.pool2(x2)

        x3 = self.down3(x) # -> skip connection 3
        x = self.pool3(x3)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up3(x)
        # Pad if necessary to match skip connection size (common in 3D UNets)
        x3 = F.interpolate(x3, size=x.shape[2:], mode='nearest')
        x = torch.cat([x, x3], dim=1)
        x = self.conv3(x)

        x = self.up2(x)
        x2 = F.interpolate(x2, size=x.shape[2:], mode='nearest')
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x1 = F.interpolate(x1, size=x.shape[2:], mode='nearest')
        x = torch.cat([x, x1], dim=1)
        x = self.conv1(x)

        # Output
        logits = self.final_conv(x)
        return logits
