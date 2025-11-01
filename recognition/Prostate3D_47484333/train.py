%%writefile train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import torch.nn.functional as F
import numpy as np
import os
import random

# Import 3D components
from module import UNet3D
from dataset import NIFTI3DSegmentationDataset, BATCH_SIZE, NUM_CLASSES, DEVICE, LR

# Define the file path for the best model weights.
MODEL_PATH = 'best_unet3d_model.pth'

# Define the prostate label index globally.
PROSTATE_LABEL_IDX = 5

def dice_score_3d(prediction, target, smooth=1e-6):
    
    # If the input is raw model output convert it to discrete class predictions.
    if prediction.ndim == 5:

        # (N, C, D, H, W) to (N, D, H, W)
        prediction = torch.argmax(prediction, dim=1) 

    # Flatten the 3D dimensions (D, H, W) into a single vector for easier per-voxel calculation.
    prediction = prediction.contiguous().view(-1)
    target = target.contiguous().view(-1)

    # Store the calculated Dice score for each class.
    dice_scores = []

    # Loop through all of the different classes: 
    for c in range(NUM_CLASSES):
        # Create simple boolean masks for the current class.
        pred_c = (prediction == c)
        target_c = (target == c)

        # Calculate True Positives (TPs) which is the overlap (intersection).
        intersection = (pred_c * target_c).sum().float()
        
        # Calculate the sum of the predicted and ground truth volumes (union). 
        union = pred_c.sum().float() + target_c.sum().float() 

        # Dice formula: 2 * Intersection / (Total predicted + Total actual). 
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())

    # Return a list of dice scores. 
    return dice_scores 
    
def train_unet_3d(model, train_loader, val_loader, epochs=50):
    return False

def main():
    
    random.seed(42)
    torch.manual_seed(42)

    try:
        train_dataset = NIFTI3DSegmentationDataset(split='train')
        val_dataset = NIFTI3DSegmentationDataset(split='validate')
        test_dataset = NIFTI3DSegmentationDataset(split='test')

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    except RuntimeError as e:
        print(f"\n[FATAL] Data Setup Failed: {e}")
        return
        
    except ValueError as e:
        print(f"\n[FATAL] Splitting Failed: {e}")
        return

    # Initialise the model. 
    model = UNet3D(in_channels=1, out_classes=NUM_CLASSES).to(DEVICE)
    print(f"\nUsing device: {DEVICE}")

    # Train the model. 
    train_unet_3d(model, train_loader, val_loader)
    print("\nTraining complete.")

if __name__ == '__main__':
    main()
