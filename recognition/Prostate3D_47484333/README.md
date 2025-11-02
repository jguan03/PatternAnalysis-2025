# 3D Medical Volume Segmentation using Improved UNet3D

## 1. Algorithm Description and Problem Solved

### Problem Statement

The accurate and efficient segmentation of anatomical structures in three dimensional medical imaging is important for clinical applications. Examples of these applications include diagnostic analysis and radiation therapy planning. This report aims to address the challenges of attempting to segment various different organ types located in the pelvic region, which include: prostate gland, rectum, bone, and the body outline from downsampled data obtained from the downsampled 3D NIfTI volumes. This task is considered to be inherently difficult due to the low constrast nature of MRI/CT scans, which is also worsened by the inherent noise in the medical system and variability in organ shapes from patient to patient. The significant class imbalance of organs is deemed true as the target (the prostate) occupies only a small fraction of the total volume of the scans. The main objective of this project is to produce segmentations with high fidelity, highlighting all of the organs with ground and masks, and to further quantify our successes by achieving a a minimum dice similarity coefficient (DSC) of 0.7 for every segmented structure.  

### Algorithm Overview: Improved 3D UNet Architecture

This project uses a modified 3D UNet architecture to tackle the medical volume segmentation of the prostate gland. The network follows a contracting and expanding structure that processes 3D scan data like from an MRI scan. 

On the downsampling path on the 3D UNet architecture implementation, the system gradually increases the models capacity and reduces the spatial dimensions of the input volume while extracting increasingly more and more complex features. This downsampling processes helps capture entire hierarchical features at multiple scales which enables distinguish between multiple organs. The bottleneck layer of the architecture acts like a junction that efficiently transfers this overall information to aid in the reconstruction. 

During the expansion phase in the UNet3D block, the network reconstructs the segmentation map by progressively upsampling the compressed features back to the original dimensions. The architecture for the UNet3D block utilises skip connections, which are shortcuts between corresponding compression and expansion stages which allow the decoder to recover fine spatial details, such as organ edges and textures that would be lost during the downsampling process. This is crucial for accurately outlining organ boundaries, which is essential for clinical use. 

The improvement in our model comes from integrating residual connections. Often times, deep networks can suffer from vanishing gradients, a problem where gradients which are used for updating the network weights become extremely small when they propagate upwards. These connections act as skip routes, which allow the gradients to flow directly through the network. This overall training process enables us to successfully build and train a more complex and capable model.


