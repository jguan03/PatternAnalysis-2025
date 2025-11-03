%%writefile module.py
"""
Contains all of the 3D UNet logic which defines all of the various
convolutional blocks (with batch normalisation) and layers that
are necessary for assembling it into a workable segmentation model.

Key Features:
- 3D convolutional operations for volumetric data
- Encoder-decoder architecture with skip connections
- Handles any input size (D, H, W)
- Outputs per-voxel class probabilities

REF:
Google Gemini AI to assist with developing the 3D UNet task. 

Author: Jiaming Guan
Date 03/11/2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3D(nn.Module):
    """
    Double 3D convolution block.
    
    Two back-to-back 3D convolutions blocks, each followed by batch norm and ReLU.
    Keeps the same spatial size throughout (padding=1, kernel=3).
    """

    def __init__(self, in_channels, out_channels):

        super(ConvBlock3D, self).__init__()

        # First conv: in_channels -> out_channels
        # Second conv: out_channels -> out_channels 
        self.double_conv = nn.Sequential(
            
            # First Convolution + BN + ReLU
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels), 
            nn.ReLU(inplace=True), # Activation function

            # Second Convolution + BN + ReLU
            # Input channels here matches the output of the first 
            # convolution block. 
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Defines the forward computation of the ConvBlock3D.
        return self.double_conv(x)

class UNet3D(nn.Module):
    """
    3D U-Net for segmenting 3D medical images (CT, MRI).
    
    Architecture:
    - Encoder: extracts features, shrinks image
    - Bottleneck: deepest layer 
    - Decoder: uses skip connections from encoder

    Input: (batch, 1, D, H, W)
    Output: (batch, out_classes, D, H, W)
    """
    
    def __init__(self, in_channels=1, out_classes=6, base_channels=32):

        super(UNet3D, self).__init__()

        # Encoder (Downsampling path)

        # Level 1: Initial feature extraction.
        self.down1 = ConvBlock3D(in_channels, base_channels) # 32 channels
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Level 2: Deeper layer, double the channel to see 
        # more patterns. 
        self.down2 = ConvBlock3D(base_channels, base_channels * 2) # 64 channels
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Level 3: Extract even higher level channels. 
        self.down3 = ConvBlock3D(base_channels * 2, base_channels * 4) # 128 channels
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Deepest layer: bottleneck layer which contains the most abstract features. 
        self.bottleneck = ConvBlock3D(base_channels * 4, base_channels * 8) # 256 channels

        # Decoder (Expanding path)
        
        # Level 3 upscale from bottle neck: transposed conv doubles size, halves channels.
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2) 
        self.conv3 = ConvBlock3D(base_channels * 8, base_channels * 4) # 128 channels

        # Level 2: Second upsampling step.
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv2 = ConvBlock3D(base_channels * 4, base_channels * 2) 

        # Level 1: Final upsampling to near-original input size.
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBlock3D(base_channels * 2, base_channels) 

        # 1x1x1 conv: base_channels -> out_classes (no spatial change). 
        self.final_conv = nn.Conv3d(base_channels, out_classes, kernel_size=1)


    def forward(self, x):

        # Encoder.
        x1 = self.down1(x) # -> skip connection 1
        x = self.pool1(x1)

        x2 = self.down2(x) # -> skip connection 2
        x = self.pool2(x2)

        x3 = self.down3(x) # -> skip connection 3
        x = self.pool3(x3)

        # Bottleneck.
        x = self.bottleneck(x)

        # Decoder - upsample and merge with skip connections
        # Level 3: 1/8 -> 1/4 size
        x = self.up3(x)
        # Fix size mismatch if any, then concat with down3 features
        x3 = F.interpolate(x3, size=x.shape[2:], mode='nearest')
        x = torch.cat([x, x3], dim=1)
        x = self.conv3(x)

        # Level 2: 1/4 -> 1/2 size 
        x = self.up2(x)
        x2 = F.interpolate(x2, size=x.shape[2:], mode='nearest')
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)

        # Level 1: 1/2 -> full size
        x = self.up1(x)
        x1 = F.interpolate(x1, size=x.shape[2:], mode='nearest')
        x = torch.cat([x, x1], dim=1)
        x = self.conv1(x)

        # Output.
        logits = self.final_conv(x)
        return logits