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