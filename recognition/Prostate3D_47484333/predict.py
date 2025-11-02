%%writefile predict.py

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import random

# Import 3D components, updated to use the shorter file names
# NOTE: This script assumes 'module.py', 'dataset.py', and 'train.py' exist in the same directory.
from module import UNet3D
from dataset import NIFTI3DSegmentationDataset, DataLoader, NUM_CLASSES, DEVICE, TARGET_VOLUME_SIZE
from train import dice_score_3d # Reuse the score function

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

# --- Model Loading ---

def load_model(model_path):
    """Loads the trained 3D UNet model weights from the specified path."""
    model = UNet3D(in_channels=1, out_classes=NUM_CLASSES).to(DEVICE)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"✅ Successfully loaded 3D model weights from {model_path}.")
    else:
        # Note the change from train_3d.py to train.py in the message below
        print(f"❌ Warning: Model weights not found at {model_path}. Please run train.py first.")

    model.eval()
    return model

# --- Visualization Logic (Qualitative: Slices) ---

def visualize_inference_slice(model, data_loader, num_samples=3):
    """
    Runs inference on random test volumes and plots a central slice (D/2)
    for visual verification.
    """
    print(f"\n--- Visualizing {num_samples} Central Slices from Test Volumes ---")

    # Set up subplots dynamically
    if num_samples > 0:
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    else:
        print("No samples requested for visualization.")
        return

    if num_samples == 1:
        # Ensure axes is a 2D array even for a single sample for consistent indexing
        axes = np.array([axes])

    dataset_size = len(data_loader.dataset)
    if dataset_size == 0:
        print("Dataset is empty. Cannot visualize.")
        plt.close(fig) # Close figure if nothing is plotted
        return

    # Get random indices (which correspond to patient volumes)
    # Note: Using random.sample on the dataset directly for slice visualization
    random_indices = random.sample(range(dataset_size), min(num_samples, dataset_size))

    # Manually retrieve the random samples from the dataset
    sample_data = [data_loader.dataset[i] for i in random_indices]

    # Custom Colormap for segmentation (FIXED to use ListedColormap for proper discretization)
    # 1. Get the continuous colormap
    base_cmap = plt.colormaps.get_cmap('jet')
    # 2. Create the discrete ListedColormap with NUM_CLASSES steps
    cmap = ListedColormap(base_cmap(np.linspace(0, 1, NUM_CLASSES)))

    with torch.no_grad():
        im = None # Initialize im for colorbar scope
        for i, (image_volume, mask_gt_volume) in enumerate(sample_data):

            # 1. Prepare input: Add batch dimension and move to device
            # image_volume shape: (1, D, H, W)
            image_input = image_volume.unsqueeze(0).to(DEVICE)

            # 2. Inference
            output_logits = model(image_input)

            # 3. Process output: Argmax for class prediction and convert to numpy
            # output_logits shape: (1, C, D, H, W) -> mask_pred_volume shape: (D, H, W)
            mask_pred_volume = torch.argmax(output_logits.squeeze(0), dim=0).cpu().numpy()

            # 4. Extract central slice (D/2) for plotting
            D = TARGET_VOLUME_SIZE[0]
            slice_idx = D // 2

            image_slice = image_volume.squeeze().cpu().numpy()[slice_idx, :, :]
            mask_gt_slice = mask_gt_volume.cpu().numpy()[slice_idx, :, :]
            mask_pred_slice = mask_pred_volume[slice_idx, :, :]

            # --- Plotting ---

            # Original Image (Grayscale) - FIX: Min-Max Normalize for visibility
            if image_slice.max() > image_slice.min():
                # Normalize image slice to 0-1 range for reliable grayscale plotting
                norm_image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
            else:
                norm_image_slice = image_slice * 0 # All zeros if data is uniform

            axes[i, 0].imshow(norm_image_slice, cmap='gray')
            axes[i, 0].set_title(f'Original Slice (D={slice_idx})')
            axes[i, 0].axis('off')

            # Ground Truth Mask (Color-mapped)
            axes[i, 1].imshow(mask_gt_slice, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
            axes[i, 1].set_title('Ground Truth Mask')
            axes[i, 1].axis('off')

            # Predicted Mask (Color-mapped)
            im = axes[i, 2].imshow(mask_pred_slice, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
            axes[i, 2].set_title('Predicted Mask')
            axes[i, 2].axis('off')

            # Print Dice Score for this specific volume
            # Note: We create a 4D tensor for prediction (1, D, H, W) and use the dice_score_3d function
            # The mask_gt_volume needs to be unsqueezed as well to (1, D, H, W) for the function signature
            dice_scores = dice_score_3d(torch.from_numpy(mask_pred_volume).to(DEVICE).unsqueeze(0),
                                         mask_gt_volume.to(DEVICE).unsqueeze(0), smooth=1e-6)
            prostate_dice = dice_scores[5]
            axes[i, 2].set_xlabel(f"Volume Prostate Dice: {prostate_dice:.4f}", fontsize=12)

        # Add a common color bar and labels (only once)
        if im is not None:
            cbar = fig.colorbar(im, ax=axes[:, 2].tolist(), ticks=np.arange(NUM_CLASSES), fraction=0.03, pad=0.04)
            cbar.ax.set_yticklabels(CLASS_LABELS)

        # Use fig.subplots_adjust to manually manage padding, which avoids the tight_layout warning
        # when a colorbar is present and ensures all elements fit.
        fig.subplots_adjust(right=0.85, wspace=0.1)

    plt.show()
    print("3D Inference visualization complete.")

# --- Visualization Logic (Quantitative: Graphs) ---

def calculate_all_dice_scores(model, data_loader):
    """
    Calculates the Dice score for all classes for every volume in the data loader.
    Returns a numpy array of shape (num_volumes, NUM_CLASSES).
    """
    all_scores = []
    print("\n--- Calculating Dice Scores Across Entire Test Set ---")
    model.eval()
    with torch.no_grad():
        for i, (image_volume, mask_gt_volume) in enumerate(data_loader):
            # image_volume shape: (1, 1, D, H, W)
            image_input = image_volume.to(DEVICE)

            # Inference
            output_logits = model(image_input) # (1, C, D, H, W)

            # Argmax for class prediction
            # Convert C-channel output to 1-channel class map
            mask_pred_volume = torch.argmax(output_logits.squeeze(0), dim=0).unsqueeze(0) # (1, D, H, W)

            # Calculate Dice scores for the volume (scores shape: (NUM_CLASSES,))
            dice_scores = dice_score_3d(mask_pred_volume, mask_gt_volume.to(DEVICE), smooth=1e-6)

            # FIX: dice_scores is a Python list, so convert it directly to a NumPy array.
            all_scores.append(np.array(dice_scores))

    print(f"Calculation complete. Processed {len(all_scores)} volumes.")
    return np.array(all_scores) # Shape (num_volumes, NUM_CLASSES)

def plot_dice_score_distribution(all_scores):
    """
    Generates a box plot for the Prostate Dice score distribution across all test volumes.
    """
    # The Prostate class is index 5
    prostate_scores = all_scores[:, 5]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create the box plot
    ax.boxplot(prostate_scores, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=5))

    # Set labels and title
    ax.set_title('Prostate Segmentation Dice Score Distribution (Test Set)', fontsize=16)
    ax.set_ylabel('Dice Score (Prostate, Class 5)', fontsize=12)
    ax.set_xticks([1])
    ax.set_xticklabels(['Prostate'])
    ax.set_ylim(0, 1.05) # Ensure y-axis is 0 to 1 for Dice scores

    # Add mean and median text
    mean_score = np.mean(prostate_scores)
    median_score = np.median(prostate_scores)

    # Text annotation for summary statistics
    ax.text(1.15, mean_score, f'Mean: {mean_score:.4f}', color='darkgreen', va='center', fontsize=10, weight='bold')
    ax.text(1.15, median_score, f'Median: {median_score:.4f}', color='red', va='center', fontsize=10, weight='bold')

    ax.grid(axis='y', linestyle='--')

    plt.show() # Using show() again, as savefig didn't help rendering last time.
    plt.close(fig) # Close the figure to free up memory

    print("Dice score distribution box plot complete.")

def plot_mean_dice_scores(all_scores):
    """
    Generates a bar chart showing the mean Dice score for every class across all test volumes.
    """
    print("\n--- Plotting Mean Dice Score per Class ---")

    # Calculate the mean score for each of the 6 classes across all 22 volumes
    mean_scores = np.mean(all_scores, axis=0)

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot, excluding the background class (index 0) for clarity
    classes_to_plot = np.arange(1, NUM_CLASSES) # Indices 1 through 5
    mean_scores_to_plot = mean_scores[classes_to_plot]
    class_labels_to_plot = CLASS_LABELS[1:]

    bars = ax.bar(class_labels_to_plot, mean_scores_to_plot, color=['teal', 'gray', 'orange', 'purple', 'red'])

    # Add labels and title
    ax.set_title('Mean Dice Score per Organ (Excluding Background)', fontsize=16)
    ax.set_ylabel('Mean Dice Score', fontsize=12)
    ax.set_ylim(0, 1.0)

    # Add the value on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 4), ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=15, ha="right")
    ax.grid(axis='y', linestyle='--')

    plt.show()
    plt.close(fig)
    print("Mean Dice score bar chart complete.")


# --- Main Execution ---

def main():
    # 1. Load the Test Data Loader
    try:
        test_dataset = NIFTI3DSegmentationDataset(split='test')
        # Use BATCH_SIZE=1 for 3D prediction/visualization.
        # Set shuffle=False for deterministic score calculation across the full test set.
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    except RuntimeError as e:
        print(f"\n[FATAL] Data Setup Failed: {e}")
        return

    # 2. Load the trained model
    model = load_model(MODEL_PATH)

    if os.path.exists(MODEL_PATH) and len(test_loader.dataset) > 0:

        # 3. Qualitative Visualization (Images)
        # Note: This function samples randomly from the dataset, independent of the DataLoader's shuffle state.
        visualize_inference_slice(model, test_loader, num_samples=3)

        # 4. Quantitative Visualization (Graphs)
        all_scores = calculate_all_dice_scores(model, test_loader)
        plot_dice_score_distribution(all_scores)
        plot_mean_dice_scores(all_scores)

if __name__ == '__main__':
    main()
