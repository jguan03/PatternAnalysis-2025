%%writefile train.py
"""
3D U-Net Training Script for Medical Image Segmentation

Trains the 3D UNet model on the 3D prostate data with weighted Dice loss,
validation monitoring, and early stopping when target Dice is achieved.

Features:
- Weighted Dice loss for class imbalance
- Multi-organ Dice scoring 
- Learning rate scheduling
- Training history visualisation

Usage:
    python train.py

REF:
Google Gemini AI to assist with developing the 3D UNet task. 

Author: Jiaming Guan 
Date 03/11/2025
"""

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

# Import 3D components from module.py and dataset.py. 
from module import UNet3D
from dataset import NIFTI3DSegmentationDataset, BATCH_SIZE, NUM_CLASSES, DEVICE, LR

# Define the file path for the best model weights.
MODEL_PATH = 'best_unet3d_model.pth'
PLOT_PATH = 'training_history.png'

# Define the prostate label index globally. 
PROSTATE_LABEL_IDX = 5

def dice_score_3d(prediction, target, smooth=1e-6):
    """Calculate 3D Dice scores for all classes."""
    
    # Convert logits to class predictions if needed. 
    if prediction.ndim == 5:

        # (N, C, D, H, W) to (N, D, H, W)
        prediction = torch.argmax(prediction, dim=1)

    # Flatten 3D volumes for per-voxel calculation. 
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
    """3D Dice loss with class weighting for imbalanced segmentation."""
    
    def __init__(self, num_classes, weights):
        """
        Args:
            num_classes: Number of segmentation classes
            weights: Tensor of weights for each class
        """
        super(WeightedDiceLoss3D, self).__init__()
        self.num_classes = num_classes
        self.weights = weights

    def forward(self, prediction, target, smooth=1e-6):
        # Convert logits to probabilities. 
        probs = F.softmax(prediction, dim=1)

        # Convert the ground truth mask into a one-hot vector format for comparison
        # using one-hot encoding.
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()

        # Flatten spatial dimensions. 
        probs_flat = probs.contiguous().view(probs.shape[0], self.num_classes, -1)
        target_flat = target_one_hot.contiguous().view(target_one_hot.shape[0], self.num_classes, -1)

        # Calculate the intersection and the total sum of volumes per class per batch item.
        intersection = (probs_flat * target_flat).sum(dim=2)
        sets_sum = probs_flat.sum(dim=2) + target_flat.sum(dim=2)

        # Compute the dice score.
        dice = (2. * intersection + smooth) / (sets_sum + smooth)

        # Apply class weights to losses to improve 
        # training. 
        class_losses = 1.0 - dice
        weighted_loss = class_losses * self.weights

        return weighted_loss.mean()

def run_epoch(model, data_loader, optimizer, is_training=True, loss_fn=None):
    """Run one training or validation epoch."""
    
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

        # Forward pass
        output_logits = model(images)

        # Calculate loss and add it to the total loss.
        loss = loss_fn(output_logits, masks_gt)
        total_loss += loss.item()

        # Backwards pass for training. 
        if is_training:
            loss.backward()
            if optimizer:
                optimizer.step()

        # Check the metrics. 
        with torch.no_grad():

            # Retrieve dice score from every class in the batch.
            dice = dice_score_3d(output_logits, masks_gt, smooth=1e-6)

            # Calculate metrics for logging. 
            organ_dice_scores = np.array(dice[1:]) 
            current_mean_organ_dice = np.mean(organ_dice_scores) if organ_dice_scores.size > 0 else 0.0
            current_min_organ_dice = np.min(organ_dice_scores) if organ_dice_scores.size > 0 else 0.0
            prostate_dice = dice[PROSTATE_LABEL_IDX] if len(dice) > PROSTATE_LABEL_IDX else 0.0

            # If validating, collect all class Dice scores.
            if not is_training:
                all_dice_scores.append(dice)

            # Update organ progress metrics. 
            progress_bar.set_postfix(
                Loss=f'{loss.item():.4f}',
                BodyDice=f'{dice[1]:.4f}',
                BoneDice=f'{dice[2]:.4f}',
                BladderDice=f'{dice[3]:.4f}',
                RectumDice=f'{dice[4]:.4f}',
                ProstateDice=f'{dice[5]:.4f}'
            )

    # Calculate average loss for the epoch.
    avg_loss = total_loss / len(data_loader)
    
    avg_dice = None
    if not is_training and all_dice_scores:

        # Average the scores across the organs. 
        avg_dice = np.mean(all_dice_scores, axis=0)


    return avg_loss, avg_dice

def plot_training_history(train_losses, val_losses, all_class_dice_history, mean_dice_scores, save_path):
    """Generates and saves a two-panel plot of training history,
    displaying loss and individual Dice scores for all organs."""

    # Class names for legend excluding background. 
    class_names = {
        1: 'Body',
        2: 'Bone',
        3: 'Bladder',
        4: 'Rectum',
        5: 'Prostate'
    }

    # Convert list of arrays to a 2D array. 
    dice_history_array = np.array(all_class_dice_history)

    epochs = range(1, len(train_losses) + 1)

    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Loss plot history.
    ax1.plot(epochs, train_losses, label='Train Loss', color='blue', marker='o', linestyle='--')
    ax1.plot(epochs, val_losses, label='Validation Loss', color='red', marker='o')
    ax1.set_title('Loss History Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Weighted Dice Loss')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Dice score history. 
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'] # Distinct colors

    # Loop through classes 1 to 5 (organs) and plot their history. 
    for class_idx in range(1, NUM_CLASSES):
        label = class_names.get(class_idx, f'Class {class_idx}')
        # Plot the column corresponding to the class index
        ax2.plot(epochs, dice_history_array[:, class_idx],
                 label=f'{label} Dice',
                 color=colors[class_idx],
                 linestyle='-')

    # Plot Mean Organ Dices. 
    ax2.plot(epochs, mean_dice_scores,
             label='Mean Organ Dice (1-5)',
             color='black',
             linewidth=2,
             linestyle='--')

    ax2.axhline(0.70, color='gray', linestyle='-.', label='Target Dice (0.70)')
    ax2.set_title('Organ Dice Score History Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.legend(loc='lower right', fontsize='small')
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.set_ylim(0, 1.05) # Ensure y-axis is suitable for Dice scores (0 to 1)

    plt.suptitle('3D UNet Segmentation Training History', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"\nTraining history plotted and saved to {save_path}")


def train_unet_3d(model, train_loader, val_loader, epochs=20):
    """Train 3D U-Net with early stopping when the DSC score
    of 0.7 is met."""
    
    class_weights = torch.tensor([
        0.5,     # 0 - Background
        1.0,     # 1 - Body
        3.0,     # 2 - Bone
        7.0,     # 3 - Bladder
        7.0,     # 4 - Rectum
        10.0     # 5 - Prostate
    ], dtype=torch.float32).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Use the new weighted dice loss with the calculated weights.
    loss_fn = WeightedDiceLoss3D(num_classes=NUM_CLASSES, weights=class_weights)

    # Add a learning rate scheduler.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    TARGET_DICE = 0.70
    best_min_organ_dice = -1.0

    # History tracking lists. 
    train_losses = []
    val_losses = []
    all_val_dice_history = []
    mean_organ_dice_scores = []

    print("\nStarting 3D UNet Training...")
    # Training epoch loop (20 epochs). 
    for epoch in range(1, epochs + 1):

        # Training epoch. 
        train_loss, _ = run_epoch(model, train_loader, optimizer, is_training=True, loss_fn=loss_fn)

        # Validation epoch. 
        with torch.no_grad():
            val_loss, val_dice_scores = run_epoch(model, val_loader, None, is_training=False, loss_fn=loss_fn)

        organ_dice_scores = np.array(val_dice_scores[1:])

        # Calculate key metrics. 
        current_min_organ_dice = np.min(organ_dice_scores)
        current_mean_organ_dice = np.mean(organ_dice_scores)

        # Check if the current model is the best performing by analysing the 
        # previous organ dice scores. 
        is_new_best = current_min_organ_dice > best_min_organ_dice
        if is_new_best:
            best_min_organ_dice = current_min_organ_dice
            
            # Save weights if validation improves. 
            torch.save(model.state_dict(), MODEL_PATH)

        # Set up scheduler to reduce learning rate when loss plateaus.
        scheduler.step(val_loss)

        # Store history for later plotting and analysis. 
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        all_val_dice_history.append(val_dice_scores)
        mean_organ_dice_scores.append(current_mean_organ_dice)


        # Print metrics during epoch training.
        print(f"\n--- Epoch {epoch:02d}/{epochs} ---")
        print(f"| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best Min Organ Dice: {best_min_organ_dice:.4f}")
        print(f"| Val Dice (Mean Organs): {current_mean_organ_dice:.4f} | Min Organ Dice: {current_min_organ_dice:.4f}")
        print("-" * 50)

        # Stop when all of the organs have reached 0.7 or above. 
        target_achieved = np.all(organ_dice_scores >= TARGET_DICE)

        if target_achieved:
            print(f"\nTarget achieved! All organ Dice scores are >= {TARGET_DICE:.2f}. Stopping training.")
            
            # Plot before stopping.
            plot_training_history(train_losses, val_losses, all_val_dice_history, mean_organ_dice_scores, PLOT_PATH)
            return True

    # Plot history after all epochs are completed.
    plot_training_history(train_losses, val_losses, all_val_dice_history, mean_organ_dice_scores, PLOT_PATH)

    return False 


def main():
    """Main training pipeline."""
    
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

    # Final Test Set Evaluation. 
    print("\n--- Final Test Set Evaluation ---")

    # Check if the saved best model file exists before proceeding.
    if os.path.exists(MODEL_PATH):
        print("Loaded best model weights for final testing.")

        # Load the model state dictionary from the best epoch.
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

        # Use unweighted loss for final evaluation. 
        unweighted_dice_loss_fn = WeightedDiceLoss3D(num_classes=NUM_CLASSES, weights=torch.ones(NUM_CLASSES).to(DEVICE))
        test_loss, test_dice_scores = run_epoch(model, test_loader, None, is_training=False, loss_fn=unweighted_dice_loss_fn)
        final_organ_dice_scores = np.array(test_dice_scores[1:])
        final_min_organ_dice = np.min(final_organ_dice_scores)

        print(f"Test Loss: {test_loss:.4f}")

        # Show final performance metric for organs 1 - 5. 
        print(f"Test Set Minimum Organ Dice (1-5): **{final_min_organ_dice:.4f}**")

    else:
        # Handle when models are not saved.
        print("Cannot run final test: Best model weights not saved.")

    # Displays the training history plot. 
    print("\nAttempting to display training history plot...")
    if os.path.exists(PLOT_PATH):
        try:
            # Load the image data. 
            img = mpimg.imread(PLOT_PATH)

            # Create a figure to display the image. 
            plt.figure(figsize=(10, 6))
            plt.title("Training History Plot")

            # Display the image. 
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Could not display plot: {e}")
    else:
        print(f"Error: The image file '{PLOT_PATH}' was not found. Please ensure 'train.py' ran successfully.")

if __name__ == '__main__':
    main()