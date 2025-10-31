%%writefile dataset.py
import os
import torch
import numpy as np
import nibabel as nib
import random
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Zipped file containing all of of the semantic MRs and 
# and semantic labels were 
LOCAL_UNZIPPED_BASE_DIR = "/tmp/data_3d"

# Segmenting 6 different tissue types
# These tissue types are: 
# Background - 0, Body Outline - 1, Bone - 2, Bladder - 3, Rectum - 4, Prostate - 5
NUM_CLASSES = 6 

# Make the batch size equal to 1 since 3D models use a lot of memory. 
BATCH_SIZE = 1 
LR = 1e-4

# Make the images smaller to fit on the GPU. (Z, Y, X)
# Downsample to (64, 128, 128).
TARGET_VOLUME_SIZE = (64, 128, 128)

IMAGE_DIR = os.path.join(LOCAL_UNZIPPED_BASE_DIR, 'semantic_MRs')
LABEL_DIR = os.path.join(LOCAL_UNZIPPED_BASE_DIR, 'semantic_labels_only')

# File name suffixes used for extracting and parsing the data for the 
# dataset. 
IMAGE_SUFFIX = "LFOV"
LABEL_SUFFIX = "SEMANTIC"

# --- Utility Functions ---
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

    # Check if the volume contains only integers (a proxy for a mask/label volume)
    # We use volume.max() > 1.0 to distinguish normalized image floats (0-1) from integer labels
    is_mask = np.issubdtype(volume.dtype, np.integer) or volume.max() > 1.0

    # Use nearest neighbor interpolation (order=0) for masks to preserve discrete labels
    if is_mask:
        resampled = zoom(volume, zoom_factors, order=0)
    # Use cubic interpolation (order=3) for image data
    else:
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

    return image_data, label_data

