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

def calculate_all_dice_scores(model, data_loader):

    all_scores = []
    print("\n--- Calculating Dice Scores Across Entire Test Set ---")
    model.eval()
    with torch.no_grad():
        for i, (image_volume, mask_gt_volume) in enumerate(data_loader):
            # Move the volume to the device. 
            image_input = image_volume.to(DEVICE)

            # Inference. 
            output_logits = model(image_input) # (1, C, D, H, W)

            # Convert C-channel output to 1-channel class map. 
            mask_pred_volume = torch.argmax(output_logits.squeeze(0), dim=0).unsqueeze(0) # (1, D, H, W)

            # Calculate Dice scores using the function imported from train.py. 
            dice_scores = dice_score_3d(mask_pred_volume, mask_gt_volume.to(DEVICE), smooth=1e-6)

            # Store results as a NumPy array
            all_scores.append(np.array(dice_scores))

    print(f"Calculation complete. Processed {len(all_scores)} volumes.")
    return np.array(all_scores) # Shape (num_volumes, NUM_CLASSES)

def plot_dice_score_distribution(all_scores):
    
    # The Prostate class is index 5.
    prostate_scores = all_scores[:, 5]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create the box plot.
    ax.boxplot(prostate_scores, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=5))

    # Set labels and title.
    ax.set_title('Prostate Segmentation Dice Score Distribution (Test Set)', fontsize=16)
    ax.set_ylabel('Dice Score (Prostate, Class 5)', fontsize=12)
    ax.set_xticks([1])
    ax.set_xticklabels(['Prostate'])
    ax.set_ylim(0, 1.05) # Ensure y-axis is 0 to 1 for Dice scores

    # Add mean and median text.
    mean_score = np.mean(prostate_scores)
    median_score = np.median(prostate_scores)

    # Text annotation for summary statistics.
    ax.text(1.15, mean_score, f'Mean: {mean_score:.4f}', color='darkgreen', va='center', fontsize=10, weight='bold')
    ax.text(1.15, median_score, f'Median: {median_score:.4f}', color='red', va='center', fontsize=10, weight='bold')

    ax.grid(axis='y', linestyle='--')

    plt.show() 
    plt.close(fig) 

def plot_mean_dice_scores(all_scores):
    
    print("\n--- Plotting Mean Dice Score per Class ---")

    # Calculate the average score for each organ. 
    mean_scores = np.mean(all_scores, axis=0)

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot, excluding the background. 
    classes_to_plot = np.arange(1, NUM_CLASSES)
    mean_scores_to_plot = mean_scores[classes_to_plot]
    class_labels_to_plot = CLASS_LABELS[1:]

    # Plot the results using distinct colors for visualisation 
    # purposes. 
    bars = ax.bar(class_labels_to_plot, mean_scores_to_plot, color=['teal', 'gray', 'orange', 'purple', 'red'])

    # Add labels and title
    ax.set_title('Mean Dice Score per Organ (Excluding Background)', fontsize=16)
    ax.set_ylabel('Mean Dice Score', fontsize=12)
    
    # Dice scores must be between 0 and 1.
    ax.set_ylim(0, 1.0)

    # Label each bar with its exact value.
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 4), ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=15, ha="right")
    ax.grid(axis='y', linestyle='--')

    plt.show()
    plt.close(fig)
    print("Mean Dice score bar chart complete.")
