%%writefile predict.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import random

from module import UNet3D
from dataset import NIFTI3DSegmentationDataset, DataLoader, NUM_CLASSES, DEVICE, TARGET_VOLUME_SIZE
from train import dice_score_3d 

# Define the file path for the best model weights
MODEL_PATH = 'best_unet3d_model.pth'

# Class index mapping for visualization labels
CLASS_LABELS = [
    "Background (0)",
    "Body Outline (1)",
    "Bone (2)",
    "Bladder (3)",
    "Rectum (4)",
    "Prostate (5)"
]

def load_model(model_path):
    
    # Ensure the model architecture matches the one used in training.
    model = UNet3D(in_channels=1, out_classes=NUM_CLASSES).to(DEVICE)

    if os.path.exists(model_path):
        
        # Grab the best weights found during training.
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Successfully loaded 3D model weights from {model_path}.")
        
    else:

        # Warn the user when the training script has not been run yet. 
        print(f"Warning: Model weights not found at {model_path}. Please run train.py first.")

    model.eval()
    return model