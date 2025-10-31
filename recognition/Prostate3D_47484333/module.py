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


