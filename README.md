# Review of CGH for 3D Heterogeneous Data

<div align="center">
  <img src="./Overview.jpg" width="100%" alt="Overview of the CGH pipelines for different 3D data representations">
</div>

This repository provides the experimental code and representative results for
a review of computer-generated holography (CGH) methods for heterogeneous 3D
data representations.

The project implements comparable hologram-generation pipelines for six types
of 3D data:

1. Layered RGB-D and Layered Depth Images (LDI)
2. Colored point clouds
3. Orthographic and perspective light fields
4. Triangular meshes
5. Voxel occupancy volumes
6. Neural radiance fields represented by instant-ngp snapshots

Each pipeline covers data preparation, complex RGB hologram generation, and
numerical reconstruction. The repository also includes shared single-view and
multi-view reconstruction tools.

## Overview

```text
Input OBJ mesh
    │
    ├── Layered RGB-D / LDI
    ├── Point cloud
    ├── Light field
    ├── Triangle representation
    ├── Voxel grid
    └── NeRF training views → instant-ngp snapshot
             │
             ▼
    Representation-specific CGH generation
             │
             ▼
    Complex RGB holograms
             │
             ├── Axial reconstruction
             └── Multi-view reconstruction
```

The numerical holograms are saved as complex arrays or as separate amplitude
and phase arrays. PNG files are provided for visualization and representative
result comparison.

## Repository Structure

```text
.
├── dataset/
│   ├── Layer/                              # Layered RGB-D / LDI data
│   ├── LightField/                         # Orthographic and perspective LF data
│   ├── Mesh/                               # Triangle-based mesh data
│   ├── NeRF/                               # NeRF training-view generation
│   │   ├── gen_multiview.py
│   │   └── README.md
│   ├── PointCloud/                         # Colored point-cloud data
│   ├── Voxel/                              # Voxel occupancy data
│   ├── Obj2Layer.py                        # OBJ → RGB-D / LDI
│   ├── Obj2LF_orth.py                      # OBJ → orthographic light field
│   ├── Obj2LF_pers.py                      # OBJ → perspective light field
│   ├── Obj2Mesh.py                         # OBJ → colored Nx12 triangles
│   ├── Obj2PCD.py                          # OBJ → colored point cloud
│   └── Obj2Vox.py                          # OBJ → voxel occupancy grid
├── model/
│   ├── Layer_ComplexHologram.py
│   ├── LightField_Orth_ComplexHologram.py
│   ├── LightField_Pers_ComplexHologram.py
│   ├── Mesh_ComplexHologram.py
│   ├── NeRF_ComplexHologram.py
│   ├── PointCloud_ComplexHologram.py
│   ├── Voxel_ComplexHologram.py
│   ├── Reconstruction_layer.py
│   └── Reconstruction_mutiview.py
├── README/
│   ├── Layer.md
│   ├── LightField.md
│   ├── Mesh.md
│   ├── NeRF.md
│   ├── PointCloud.md
│   ├── Reconstruction.md
│   └── Voxel.md
├── result/                                 # Representative reconstruction results
├── environment.yml                        # Portable Conda environment
└── README.md
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data.git
cd Review-of-CGH-for-3D-Heterogeneous-Data
```

### 2. Download the Example Mesh

The experiments use `BunnyDragon.obj` as the common source scene.

- Download page:
  [Baidu Netdisk](https://pan.baidu.com/s/1v8WpkBr7jnyo2QVmDijYJA?pwd=kam7)
- Extraction code: `kam7`

Place the downloaded file at:

```text
dataset/BunnyDragon.obj
```

You may also use another valid triangular OBJ mesh and replace the input path
in the example commands.

### 3. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate cgh-review
```

The environment includes the dependencies required by the layer, point-cloud,
light-field, mesh, voxel, and reconstruction pipelines.

The NeRF workflow additionally requires
[instant-ngp](https://github.com/NVlabs/instant-ngp), which must be built
separately. See [README/NeRF.md](./README/NeRF.md) for instructions.

### 4. Select a Pipeline

Each representation has a detailed, copy-and-paste tutorial:

| Representation | Data generator | Hologram generator | Guide |
| --- | --- | --- | --- |
| Layered RGB-D / LDI | [Obj2Layer.py](./dataset/Obj2Layer.py) | [Layer_ComplexHologram.py](./model/Layer_ComplexHologram.py) | [Layer guide](./README/Layer.md) |
| Point cloud | [Obj2PCD.py](./dataset/Obj2PCD.py) | [PointCloud_ComplexHologram.py](./model/PointCloud_ComplexHologram.py) | [Point-cloud guide](./README/PointCloud.md) |
| Light field | [Obj2LF_orth.py](./dataset/Obj2LF_orth.py), [Obj2LF_pers.py](./dataset/Obj2LF_pers.py) | [Orthographic CGH](./model/LightField_Orth_ComplexHologram.py), [Perspective CGH](./model/LightField_Pers_ComplexHologram.py) | [Light-field guide](./README/LightField.md) |
| Triangle mesh | [Obj2Mesh.py](./dataset/Obj2Mesh.py) | [Mesh_ComplexHologram.py](./model/Mesh_ComplexHologram.py) | [Mesh guide](./README/Mesh.md) |
| Voxel volume | [Obj2Vox.py](./dataset/Obj2Vox.py) | [Voxel_ComplexHologram.py](./model/Voxel_ComplexHologram.py) | [Voxel guide](./README/Voxel.md) |
| NeRF | [gen_multiview.py](./dataset/NeRF/gen_multiview.py) | [NeRF_ComplexHologram.py](./model/NeRF_ComplexHologram.py) | [NeRF guide](./README/NeRF.md) |

For reconstruction from an existing hologram, see the
[reconstruction guide](./README/Reconstruction.md).

## Experimental Results

The following images are representative numerical reconstructions included in
the repository. Axial examples show different reconstruction distances.
Multi-view examples show the upper-left, center, and lower-right views of the
view grid.

### 1. Layered RGB-D / LDI

Axial reconstruction over approximately 50–53 mm:

<div align="center">
  <img src="./result/Layer/layer_recon_1.png" width="30%" alt="Layer reconstruction at the first depth">
  <img src="./result/Layer/layer_recon_2.png" width="30%" alt="Layer reconstruction at the middle depth">
  <img src="./result/Layer/layer_recon_3.png" width="30%" alt="Layer reconstruction at the final depth">
</div>

Multi-view reconstruction:

<div align="center">
  <img src="./result/Layer/v_00_00.png" width="30%" alt="Layer upper-left view">
  <img src="./result/Layer/v_04_04.png" width="30%" alt="Layer center view">
  <img src="./result/Layer/v_08_08.png" width="30%" alt="Layer lower-right view">
</div>

### 2. Colored Point Cloud

Axial reconstruction over approximately 50–53 mm:

<div align="center">
  <img src="./result/PointCloud/PCD_recon_1_3e7.png" width="30%" alt="Point-cloud reconstruction at the first depth">
  <img src="./result/PointCloud/PCD_recon_2_3e7.png" width="30%" alt="Point-cloud reconstruction at the middle depth">
  <img src="./result/PointCloud/PCD_recon_3_3e7.png" width="30%" alt="Point-cloud reconstruction at the final depth">
</div>

Multi-view reconstruction:

<div align="center">
  <img src="./result/PointCloud/v_00_00.png" width="30%" alt="Point-cloud upper-left view">
  <img src="./result/PointCloud/v_04_04.png" width="30%" alt="Point-cloud center view">
  <img src="./result/PointCloud/v_08_08.png" width="30%" alt="Point-cloud lower-right view">
</div>

### 3. Light Field

Orthographic light-field reconstruction:

<div align="center">
  <img src="./result/LightField/LF_orth_recon_1.png" width="30%" alt="Orthographic LF reconstruction at the first depth">
  <img src="./result/LightField/LF_orth_recon_2.png" width="30%" alt="Orthographic LF reconstruction at the middle depth">
  <img src="./result/LightField/LF_orth_recon_3.png" width="30%" alt="Orthographic LF reconstruction at the final depth">
</div>

Perspective light-field reconstruction:

<div align="center">
  <img src="./result/LightField/LF_pers_recon_1.png" width="30%" alt="Perspective LF reconstruction at the first depth">
  <img src="./result/LightField/LF_pers_recon_2.png" width="30%" alt="Perspective LF reconstruction at the middle depth">
  <img src="./result/LightField/LF_pers_recon_3.png" width="30%" alt="Perspective LF reconstruction at the final depth">
</div>

Perspective light-field multi-view reconstruction:

<div align="center">
  <img src="./result/LightField/v_00_00.png" width="30%" alt="Light-field upper-left view">
  <img src="./result/LightField/v_04_04.png" width="30%" alt="Light-field center view">
  <img src="./result/LightField/v_08_08.png" width="30%" alt="Light-field lower-right view">
</div>

### 4. Triangle Mesh

The following axial examples use the 40× object-scale experiment:

<div align="center">
  <img src="./result/Mesh/Mesh_recon_40_1.png" width="30%" alt="Mesh reconstruction at the first depth">
  <img src="./result/Mesh/Mesh_recon_40_2.png" width="30%" alt="Mesh reconstruction at the middle depth">
  <img src="./result/Mesh/Mesh_recon_40_3.png" width="30%" alt="Mesh reconstruction at the final depth">
</div>

Multi-view reconstruction from the 20× scale experiment:

<div align="center">
  <img src="./result/Mesh/v_02_02.png" width="30%" alt="Mesh upper-left selected view">
  <img src="./result/Mesh/v_04_04.png" width="30%" alt="Mesh center view">
  <img src="./result/Mesh/v_06_06.png" width="30%" alt="Mesh lower-right selected view">
</div>

### 5. Voxel Occupancy Volume

Axial reconstruction over approximately 50–53 mm:

<div align="center">
  <img src="./result/Voxel/Voxel_recon_1.png" width="30%" alt="Voxel reconstruction at the first depth">
  <img src="./result/Voxel/Voxel_recon_2.png" width="30%" alt="Voxel reconstruction at the middle depth">
  <img src="./result/Voxel/Voxel_recon_3.png" width="30%" alt="Voxel reconstruction at the final depth">
</div>

Multi-view reconstruction:

<div align="center">
  <img src="./result/Voxel/v_00_00.png" width="30%" alt="Voxel upper-left view">
  <img src="./result/Voxel/v_04_04.png" width="30%" alt="Voxel center view">
  <img src="./result/Voxel/v_08_08.png" width="30%" alt="Voxel lower-right view">
</div>

### 6. Neural Radiance Field

Axial reconstruction over approximately 50–53 mm:

<div align="center">
  <img src="./result/NeRF/NeRF_recon1.png" width="30%" alt="NeRF reconstruction at the first depth">
  <img src="./result/NeRF/NeRF_recon2.png" width="30%" alt="NeRF reconstruction at the middle depth">
  <img src="./result/NeRF/NeRF_recon3.png" width="30%" alt="NeRF reconstruction at the final depth">
</div>

Multi-view reconstruction:

<div align="center">
  <img src="./result/NeRF/v_02_02.png" width="30%" alt="NeRF upper-left selected view">
  <img src="./result/NeRF/v_04_04.png" width="30%" alt="NeRF center view">
  <img src="./result/NeRF/v_06_06.png" width="30%" alt="NeRF lower-right selected view">
</div>

## Reconstruction Utilities

The repository includes two shared reconstruction tools:

- [Reconstruction_layer.py](./model/Reconstruction_layer.py): reconstructs
  one RGB image for each axial distance.
- [Reconstruction_mutiview.py](./model/Reconstruction_mutiview.py):
  reconstructs a grid of sub-aperture views at each distance.

Both tools load unnormalized hologram amplitude and phase arrays. Detailed
commands and propagation-mode guidance are provided in
[README/Reconstruction.md](./README/Reconstruction.md).

## Notes on Reproducibility

- Numerical depth and optical units differ between pipelines. Follow the
  representation-specific README rather than transferring parameters
  directly between methods.
- PNG hologram files are visualization previews. Use the `.npy` amplitude,
  phase, or complex arrays for numerical reconstruction.
- Light-field and NeRF holograms can require substantial memory because the
  complex-field size is the product of angular and spatial resolution.
- Mesh hologram generation requires an NVIDIA CUDA GPU and CuPy.
- instant-ngp is an external dependency and is not installed by
  `environment.yml`.

## Detailed Documentation

- [Layered-data pipeline](./README/Layer.md)
- [Point-cloud pipeline](./README/PointCloud.md)
- [Light-field pipeline](./README/LightField.md)
- [Triangle-mesh pipeline](./README/Mesh.md)
- [Voxel pipeline](./README/Voxel.md)
- [NeRF pipeline](./README/NeRF.md)
- [Reconstruction utilities](./README/Reconstruction.md)
