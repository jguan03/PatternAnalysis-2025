%%writefile train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import torch.nn.functional as F
import numpy as np
import os
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg 

# Import 3D components
from module import UNet3D
from dataset import NIFTI3DSegmentationDataset, BATCH_SIZE, NUM_CLASSES, DEVICE, LR

# Define the file path for the best model weights.
MODEL_PATH = 'best_unet3d_model.pth'
PLOT_PATH = 'training_history.png' 

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

class WeightedDiceLoss3D(nn.Module):
    
    def __init__(self, num_classes, weights):
        super(WeightedDiceLoss3D, self).__init__()
        self.num_classes = num_classes
        self.weights = weights # Tensor of size (C,)

    def forward(self, prediction, target, smooth=1e-6):
        # Convert raw output logits into probabilities using Softmax. 
        probs = F.softmax(prediction, dim=1)

        # Convert the ground truth mask into a one-hot vector format for comparison
        # using one-hot encoding. 
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()

        # Reshape the data to combine the D, H, and W dimensions into a single voxel dimension. 
        probs_flat = probs.contiguous().view(probs.shape[0], self.num_classes, -1)
        target_flat = target_one_hot.contiguous().view(target_one_hot.shape[0], self.num_classes, -1)

         # Calculate the intersection and the total sum of volumes per class per batch item.
        intersection = (probs_flat * target_flat).sum(dim=2) 
        sets_sum = probs_flat.sum(dim=2) + target_flat.sum(dim=2) 

        # Compute the dice score. 
        dice = (2. * intersection + smooth) / (sets_sum + smooth)

        # Compute the dice loss. 
        class_losses = 1.0 - dice 

        # Apply the pre-defined weights. 
        weighted_loss = class_losses * self.weights

        # Return the average loss across all classes and items in the batch. 
        return weighted_loss.mean()

def run_epoch(model, data_loader, optimizer, is_training=True, loss_fn=None):

    # training uses model.train()
    # validation uses model.eval()
    model.train() if is_training else model.eval()

    total_loss = 0.0
    all_dice_scores = []

    # Create progress bar to track the amount of time it takes to finish one epoch of training. 
    progress_bar = tqdm(data_loader, desc=f"Epoch {'Train' if is_training else 'Valid'}", leave=False)

    for images, masks_gt in progress_bar:
        # Move the data to the GPU (if available). 
        images, masks_gt = images.to(DEVICE), masks_gt.to(DEVICE)

        if is_training:
            # Zero the gradients before the forward pass. 
            optimizer.zero_grad()

        # Pass the images through the UNet to get logits. 
        output_logits = model(images)

        # Calculate loss and add it to the total loss. 
        loss = loss_fn(output_logits, masks_gt)
        total_loss += loss.item()

        # Backpropagation and optimisation during training. 
        if is_training:
            loss.backward()
            if optimizer:
                optimizer.step()

        # Check the current performance of the training during the training process. 
        with torch.no_grad():

            # Retrieve dice score from every class in the batch. 
            dice = dice_score_3d(output_logits, masks_gt, smooth=1e-6)
            current_prostate_dice = dice[PROSTATE_LABEL_IDX] 

            # If validating, collect all class Dice scores. 
            if not is_training:
                all_dice_scores.append(dice)

        # Update progress to display current metrics. 
        progress_bar.set_postfix(
            Loss=f'{loss.item():.4f}',
            ProstateDice=f'{current_prostate_dice:.4f}'
        )

    # Calculate average loss for the epoch. 
    avg_loss = total_loss / len(data_loader)

    # Calculate mean Dice scores across all validation batches. 
    avg_dice = None
    if not is_training and all_dice_scores:
        
        # Average the collected scores over the entire validation set. 
        avg_dice = np.mean(all_dice_scores, axis=0)

    
    return avg_loss, avg_dice

def plot_training_history(train_losses, val_losses, prostate_dice, mean_dice, save_path):
    """Generates and saves a two-panel plot of training history."""
    epochs = range(1, len(train_losses) + 1)

    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Loss History
    ax1.plot(epochs, train_losses, label='Train Loss', color='blue', marker='o', linestyle='--')
    ax1.plot(epochs, val_losses, label='Validation Loss', color='red', marker='o')
    ax1.set_title('Loss History Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Weighted Dice Loss')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Plot 2: Dice Score History
    ax2.plot(epochs, prostate_dice, label='Prostate Dice (Index 5)', color='green', marker='s')
    ax2.plot(epochs, mean_dice, label='Mean Dice (All Classes)', color='purple', marker='^', linestyle='--')
    ax2.axhline(0.70, color='gray', linestyle='-.', label='Target Dice (0.70)')
    ax2.set_title('Dice Score History Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.suptitle('3D UNet Segmentation Training History', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(save_path)
    plt.close(fig)
    print(f"\nTraining history plotted and saved to {save_path}")

    
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
