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

# dataset.py - Continuing the NIFTI3DSegmentationDataset class definition
class NIFTI3DSegmentationDataset(Dataset):
    
    def __init__(self, split='train'):
        # This line keeps track of what data type we are storing: 
        # (train, validate, and test). 
        self.split = split 

        # Check if the folder paths exist before attempting to load data. 
        if not os.path.exists(IMAGE_DIR) or not os.path.exists(LABEL_DIR):
            raise RuntimeError(
                f"3D data not found. Please run prepare_3d_data.sh. "
                f"Missing {IMAGE_DIR} or {LABEL_DIR}"
            )

        # Retrieve a list with all of image files in the 
        # directory. 
        all_files = sorted(os.listdir(IMAGE_DIR))

        # Extract the base ID from the image name. 
        all_base_ids = []
        for f in all_files:
            if f.endswith('.nii.gz'):
                
                # Remove the file extension. 
                base_id_with_suffix = f.replace('.nii.gz', '')
                
                # Check for and strip the expected image suffix (_LFOV)
                if base_id_with_suffix.endswith(f'_{IMAGE_SUFFIX}'):
                    base_id = base_id_with_suffix[:-len(f'_{IMAGE_SUFFIX}')]
                else:
                    base_id = base_id_with_suffix

                # Construct the expected file name for the label mask. 
                label_filename = f"{base_id}_{LABEL_SUFFIX}.nii.gz"
                label_path = os.path.join(LABEL_DIR, label_filename)

                # Only add the base ID if both the image and the label file exist
                if os.path.exists(label_path):
                    all_base_ids.append(base_id)
                else:
                    print(f"Warning: Skipping {f}. Corresponding label {label_filename} not found.")

        # Setting the random seed to a 
        # fixed number makes it produce the 
        # same randomised order of patient IDs. 
        random.seed(42)
        
        # Shuffle the list of patient IDs in that 
        # consistent, randomised order.  
        random.shuffle(all_base_ids) 

        # Defining the boundaries of splitting. 
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
        # Function returns the number of patient IDs. 
        return len(self.patient_ids)

    def __getitem__(self, idx):

        # Retrieve unique patient ID for this item (volume)
        # in the list of patient IDs. 
        patient_id = self.patient_ids[idx]
        
        # Use the previously obtained patient ID to build the 
        # suffixes for the file names. 
        image_filename = f"{patient_id}_{IMAGE_SUFFIX}.nii.gz"
        label_filename = f"{patient_id}_{LABEL_SUFFIX}.nii.gz"

        # Complete building the full image path by joining. 
        image_path = os.path.join(IMAGE_DIR, image_filename)
        label_path = os.path.join(LABEL_DIR, label_filename)
        
        # Use nibabel to load the NIfTI files for both the 
        # images and labels. 
        image_nii = nib.load(image_path)
        label_nii = nib.load(label_path)

        # Extract numerical data from the NIfTI objects. 
        image_data = image_nii.get_fdata().astype(np.float32)
        label_data = label_nii.get_fdata().astype(np.uint8)
        
        # Ensure image data has 3 dimensions (Depth, Height, Width)
        # If the image data contains 4, remove it. 
        if image_data.ndim == 4:
            image_data = image_data.squeeze()
        
        # Normalise the image data by applying the 0-1 scaling function. 
        # This is done to standardise the brightness and contrast of the 
        # image data. 
        image_data = normalize_volume(image_data)
        
        # Resize the labels and images to match the target volume size. 
        image_resampled = resample_volume(image_data, TARGET_VOLUME_SIZE)
        label_resampled = resample_volume(label_data, TARGET_VOLUME_SIZE)
        
        # Apply data augmentation (flipping) to the training data set. 
        if self.split == 'train':
             image_resampled, label_resampled = augment_volume(image_resampled, label_resampled)
        
        # Convert image array to PyTorch tensor. Add a channel dimension (C=1).
        image_tensor = torch.from_numpy(image_resampled[np.newaxis, ...].copy()).float()
        
        # Labels should remain (D, H, W) for loss function/argmax. 
        label_tensor = torch.from_numpy(label_resampled.copy()).long()

        # Check if the tensor size is correct (matches the target volume size). 
        if image_tensor.shape[1:] != TARGET_VOLUME_SIZE:
             print(f"Error: Final image shape {image_tensor.shape[1:]} does not match target {TARGET_VOLUME_SIZE}")

        # Return the processed image and its corresponding label mask. 
        return image_tensor, label_tensor


            

        
