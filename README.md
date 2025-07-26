# 2d-image-to-3d-model-converter

A web-based application that converts 2D images into 3D models using deep learning and computer vision techniques.

✨ Features

Real-time 2D to 3D conversion using state-of-the-art depth estimation
GPU acceleration with CUDA support for faster processing
Web interface with drag-and-drop functionality
Multiple depth estimation models (MiDaS, DPT) with fallback options
3D mesh generation using Poisson surface reconstruction
Export formats (.ply files compatible with Blender, MeshLab)

Prerequisites

Python 3.8+
CUDA-compatible GPU (optional, but recommended)
Conda environment manager

🏗️ Project Structure

2d-to-3d-converter/
├── app.py                 # Flask web server
├── depth_estimator.py     # Depth estimation logic
├── mesh_generator.py      # 3D mesh creation
├── requirements.txt       # Dependencies
├── README.md             # This file
├── LICENSE               # MIT License
├── .gitignore            # Git ignore rules
├── templates/
│   └── index.html        # Web interface
├── uploads/              # Temporary file storage (auto-created)
└── demo/                 # Demo images and results
    ├── input_example.jpg
    └── output_example.ply

🔧 How It Works

1) Image Upload: User uploads a 2D image through the web interface
2) Depth Estimation: Uses pre-trained models (Intel DPT-Large or MiDaS) to estimate depth
3) Point Cloud Generation: Converts RGB + Depth data into 3D point cloud
4) Mesh Reconstruction: Uses Poisson surface reconstruction to create 3D mesh
5) Export: Generates downloadable .ply file
