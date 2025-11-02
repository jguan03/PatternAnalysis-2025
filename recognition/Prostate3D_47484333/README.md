# 3D Medical Volume Segmentation using Improved UNet3D

## 1. Algorithm Description and Problem Solved

### Problem Statement

The accurate and efficient segmentation of anatomical structures in three dimensional medical imaging is important for clinical applications. Examples of these applications include diagnostic analysis and radiation therapy planning. This report aims to address the challenges of attempting to segment various different organ types located in the pelvic region, which include: prostate gland, rectum, bone, and the body outline from downsampled data obtained from the downsampled 3D NIfTI volumes. This task is considered to be inherently difficult due to the low constrast nature of MRI/CT scans, which is also worsened by the inherent noise in the medical system and variability in organ shapes from patient to patient. The significant class imbalance of organs is deemed true as the target (the prostate) occupies only a small fraction of the total volume of the scans. The main objective of this project is to produce segmentations with high fidelity, highlighting all of the organs with ground and masks, and to further quantify our successes by achieving a a minimum dice similarity coefficient (DSC) of 0.7 for every segmented structure.  

### Algorithm Overview: Improved 3D UNet Architecture

This project uses a modified 3D UNet architecture to tackle the medical volume segmentation of the prostate gland. The network follows a contracting and expanding structure that processes 3D scan data like from an MRI scan. 

On the downsampling path on the 3D UNet architecture implementation, the system gradually increases the models capacity and reduces the spatial dimensions of the input volume while extracting increasingly more and more complex features. This downsampling processes helps capture entire hierarchical features at multiple scales which enables distinguish between multiple organs. The bottleneck layer of the architecture acts like a junction that efficiently transfers this overall information to aid in the reconstruction. 

During the expansion phase in the UNet3D block, the network reconstructs the segmentation map by progressively upsampling the compressed features back to the original dimensions. The architecture for the UNet3D block utilises skip connections, which are shortcuts between corresponding compression and expansion stages which allow the decoder to recover fine spatial details, such as organ edges and textures that would be lost during the downsampling process. This is crucial for accurately outlining organ boundaries, which is essential for clinical use. 

The improvement in our model comes from integrating residual connections. Often times, deep networks can suffer from vanishing gradients, a problem where gradients which are used for updating the network weights become extremely small when they propagate upwards. These connections act as skip routes, which allow the gradients to flow directly through the network. This overall training process enables us to successfully build and train a more complex and capable model.

## 2. Dependencies and Reproducibility

This project is implemented in Python and relies heavily on the PyTorch deep learning framework and complementary scientific libraries to effectively segment the required 3D medical image processing and deep learning workflows. It relies on various key packages in order to fulfil processes such as data loading, model training, pre-processing, and the overall evaluation of the model. 

### Required Dependencies

| Package Name | Minimum Version | Description |
|--------------|-----------------|-------------|
| torch | 2.0.0 | Core deep learning library |
| torchvision | 0.15.0 | For common image/data utilities |
| numpy | 1.23.0 | Fundamental package for scientific computing |
| matplotlib | 3.7.0 | For visualisation and plotting results |
| nibabel | 5.1.0 | For loading and handling NIfTI format medical images |
| tqdm | 4.65.0 | For displaying training progress bars |
| scipy | 1.9.0 | For 3D image resampling and scientific computing |

### Reproducibility of Results

To ensure the reproducibility of the reported results (specifically, DSC ≥ 0.7 for all organ classes), the following steps are critical:

- **Seed Configuration**: All random seeds for NumPy, PyTorch, and Python's random module must be fixed at the start of the train.py script.
- **Dataset Integrity**: Use the exact same version of the downsampled Prostate 3D dataset (linked in the Appendix).
- **Model Weights**: The trained model weights (`best_unet3d_model.pth`) must be provided alongside the code as reference to check for consistency between different runs. 
- **Device Consistency**: Training was performed on a NVIDIA A100 on a GPU using CUDA, on Google Collabs. The training time was approximately 40 - 50 minutes, however, results may vary slightly if run on a different GPU or CPU.
- **File Pathing**: As specified previously, training was done on the Google Collabs which requires the dataset to be uploaded on Google Drives as a zip file. Prior to running the dataset.py file, a bash script was needed to unzip the file and place it in a temporary directory '/tmp/data_3d' so that the data could be processed faster. In order to recreate this process in a local directory, the file path in dataset.py should be modified to the location of the dataset (line 14): LOCAL_UNZIPPED_BASE_DIR = "/tmp/data_3d". 




