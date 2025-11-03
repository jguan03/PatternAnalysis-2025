%%writefile dataset.py
"""
Segmentation Dataset for Prostate 3D medical scans. 

Handles loading and preprocessing of 3D NIfTI medical images for segmentation tasks.
Supports training, validation, and test splits with data augmentation for training.

Features:
- Loads paired 3D MRI scans and segmentation masks
- Applies min-max normalisation and resampling
- Data augmentation like flips and rotations 
- Automatic train/val/test split (80/10/10)

REF:
Google Gemini AI to assist with developing the 3D UNet task. 

Author: Jiaming Guan 
Date: 03/11/2025
"""

import os
import torch
import numpy as np
import nibabel as nib
import random
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom, rotate

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Zipped file containing data. 
LOCAL_UNZIPPED_BASE_DIR = "/tmp/data_3d"

# Segmenting 6 different tissue types
# These tissue types are:
# Background - 0, Body Outline - 1, Bone - 2, Bladder - 3, Rectum - 4, Prostate - 5
NUM_CLASSES = 6

# Make batch size smaller due to memory constraints. 
BATCH_SIZE = 1
LR = 1e-4

# Downsample images to (64, 128, 128)
# to fit on the GPU. 
TARGET_VOLUME_SIZE = (64, 128, 128)

# File pathing and naming conventions. 
IMAGE_DIR = os.path.join(LOCAL_UNZIPPED_BASE_DIR, 'semantic_MRs')
LABEL_DIR = os.path.join(LOCAL_UNZIPPED_BASE_DIR, 'semantic_labels_only')

# File name suffixes used for extracting and parsing the data for the
# dataset.
IMAGE_SUFFIX = "LFOV"
LABEL_SUFFIX = "SEMANTIC"

def normalize_volume(volume):
    """Applies simple min-max normalization to the image volume."""
    
    volume = volume.astype(np.float32)
    min_val = volume.min()
    max_val = volume.max()
    if max_val > min_val:
        # Simple formula: (value - minimum) / (maximum - minimum)
        volume = (volume - min_val) / (max_val - min_val)
    return volume

def resample_volume(volume, target_size):
    """Resamples the volume to the target size using scipy.ndimage.zoom."""
    
    current_size = volume.shape
    zoom_factors = [target_size[i] / current_size[i] for i in range(3)]

    # Check if the volume contains only integers (a proxy for a mask/label volume). 
    is_mask = np.issubdtype(volume.dtype, np.integer) or volume.max() > 1.0

    # Use nearest neighbor interpolation for masks to preserve discrete labels. 
    if is_mask:
        resampled = zoom(volume, zoom_factors, order=0)
    else:
        # Use cubic interpolation (order=3) for image data
        resampled = zoom(volume, zoom_factors, order=3)

    return resampled

def augment_volume(image_data, label_data):
    """
    Applies random data augmentation (flips) to both image and label volumes.
    """
    # Random Z-axis (Depth) flip
    if random.random() < 0.5:
        image_data = np.flip(image_data, axis=0).copy()
        label_data = np.flip(label_data, axis=0).copy()

    # Random Y-axis (Height) flip
    if random.random() < 0.5:
        image_data = np.flip(image_data, axis=1).copy()
        label_data = np.flip(label_data, axis=1).copy()

    # Random X-axis (Width) flip
    if random.random() < 0.5:
        image_data = np.flip(image_data, axis=2).copy()
        label_data = np.flip(label_data, axis=2).copy()

    # Random rotations in axial plane
    if random.random() < 0.75:
        k = random.randint(1, 3)
        image_data = np.rot90(image_data, k=k, axes=(1, 2)).copy()
        label_data = np.rot90(label_data, k=k, axes=(1, 2)).copy()

    # Random intensity variations 
    if random.random() < 0.75:
        # Randomly scale intensity
        contrast_factor = random.uniform(0.85, 1.15)
        image_data = image_data * contrast_factor

        # Randomly shift intensity 
        brightness_factor = random.uniform(-0.1, 0.1)
        image_data = image_data + brightness_factor

        # Re-clip to ensure image remains in the normalized range [0, 1]
        image_data = np.clip(image_data, 0.0, 1.0)

    return image_data, label_data

class NIFTI3DSegmentationDataset(Dataset):
    """Dataset for 3D NIfTI medical image segmentation."""
    
    def __init__(self, split='train'):
        """Initialize dataset with specified split (train/validate/test)."""
        self.split = split

        # Check if the folder paths exist before attempting to load data.
        if not os.path.exists(IMAGE_DIR) or not os.path.exists(LABEL_DIR):
            raise RuntimeError(
                f"3D data not found. Please run prepare_3d_data.sh. "
                f"Missing {IMAGE_DIR} or {LABEL_DIR}"
            )

        # Collect valid patient IDs with both image and label files. 
        all_files = sorted(os.listdir(IMAGE_DIR))

        # Extract the base ID from the image name.
        all_base_ids = []
        for f in all_files:
            if f.endswith('.nii.gz'):

                # Remove the file extension.
                base_id_with_suffix = f.replace('.nii.gz', '')

                # Check for and strip the expected image suffix (_LFOV). 
                if base_id_with_suffix.endswith(f'_{IMAGE_SUFFIX}'):
                    base_id = base_id_with_suffix[:-len(f'_{IMAGE_SUFFIX}')]
                else:
                    base_id = base_id_with_suffix

                # Construct the expected file name for the label mask.
                label_filename = f"{base_id}_{LABEL_SUFFIX}.nii.gz"
                label_path = os.path.join(LABEL_DIR, label_filename)

                # Only add the base ID if both the image and the label file exist. 
                if os.path.exists(label_path):
                    all_base_ids.append(base_id)
                else:
                    print(f"Warning: Skipping {f}. Corresponding label {label_filename} not found.")

        # Setting the random seed to a 
        # fixed number for reproducability. 
        random.seed(42)

        # Shuffle the list of patient IDs in that
        # consistent, randomised order.
        random.shuffle(all_base_ids)

        # Defining the boundaries of data splitting. 
        total_patients = len(all_base_ids)
        train_end = int(total_patients * 0.8)
        val_end = int(total_patients * 0.9)

        # Train dataset takes up the first 80% of the
        # list.
        if split == 'train':
            self.patient_ids = all_base_ids[:train_end]

        # Train validate takes up the next 10% of the
        # list.
        elif split == 'validate':
            self.patient_ids = all_base_ids[train_end:val_end]

        # Train test takes up the next 10% of the
        # list.
        elif split == 'test':
            self.patient_ids = all_base_ids[val_end:]
        else:
            raise ValueError("Split must be 'train', 'validate', or 'test'.")

    def __len__(self):
        """Return number of patients in dataset."""
        
        return len(self.patient_ids)

    def __getitem__(self, idx):
        """Load and preprocess single patient's image and label data."""
        
        # Build file paths. 
        patient_id = self.patient_ids[idx]

        # Continue building the file paths.
        image_filename = f"{patient_id}_{IMAGE_SUFFIX}.nii.gz"
        label_filename = f"{patient_id}_{LABEL_SUFFIX}.nii.gz"
        image_path = os.path.join(IMAGE_DIR, image_filename)
        label_path = os.path.join(LABEL_DIR, label_filename)

        # Load NIfTI files. 
        image_nii = nib.load(image_path)
        label_nii = nib.load(label_path)
        image_data = image_nii.get_fdata().astype(np.float32)
        label_data = label_nii.get_fdata().astype(np.uint8)

        # Remove extra dimensions if present. 
        if image_data.ndim == 4:
            image_data = image_data.squeeze()

        # Normalise the image data by applying the 0-1 scaling function.
        image_data = normalize_volume(image_data)

        # Resize the labels and images to match the target volume size.
        image_resampled = resample_volume(image_data, TARGET_VOLUME_SIZE)
        label_resampled = resample_volume(label_data, TARGET_VOLUME_SIZE)

        # Apply augmentations for training data. 
        if self.split == 'train':
             image_resampled, label_resampled = augment_volume(image_resampled, label_resampled)

        # Convert image array to PyTorch tensor.
        image_tensor = torch.from_numpy(image_resampled[np.newaxis, ...].copy()).float()

        # Labels should remain (D, H, W) for loss function/argmax.
        label_tensor = torch.from_numpy(label_resampled.copy()).long()

        # Check if the tensor size is correct.
        if image_tensor.shape[1:] != TARGET_VOLUME_SIZE:
             print(f"Error: Final image shape {image_tensor.shape[1:]} does not match target {TARGET_VOLUME_SIZE}")

        return image_tensor, label_tensor

if __name__ == '__main__':

    random.seed(42)
    train_dataset = NIFTI3DSegmentationDataset(split='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
