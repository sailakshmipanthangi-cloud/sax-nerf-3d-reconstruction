# SAX-NeRF: Sparse-View 3D CT Reconstruction
#Overview
This project implements **SAX-NeRF**, a neural rendering approach for reconstructing **3D CT volumes from sparse X-ray projections**.  
The model learns volumetric density using a neural radiance field representation and reconstructs 3D medical structures from limited projection data.
This project demonstrates GPU-accelerated training and evaluation using deep learning techniques.

#Features
- Sparse-view CT reconstruction
- Neural Radiance Field (NeRF) based modeling
- GPU accelerated training
- Projection and 3D PSNR evaluation
- Visualization of reconstructed CT slices
- TensorBoard training monitoring
## Project Structure
SAX-NERF-Project
│
├── src
│ ├── dataset
│ ├── encoder
│ ├── trainer.py
│
├── scripts
│ ├── train.py
│
├── visualize_output.py
├── view_reconstruction.py
├── README.md
└── .gitignore
#Created environment:
conda create -n saxnerf python=3.10
conda activate saxnerf
##Installed dependencies:
pip install torch torchvision torchaudio
pip install numpy matplotlib tensorboard tqdm
PYTORCH +CUDA
#Training:
python -m scripts.train
#Monitoring Training
tensorboard --logdir logs
#Results:
proj_psnr : 3.95
psnr_3d   : 23.45
Author
Priya Pranathi
