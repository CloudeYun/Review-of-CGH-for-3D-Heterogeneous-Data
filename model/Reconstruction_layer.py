# -*- coding: utf-8 -*-
"""
Reconstruction RGB from amp/phase holograms
Supports:
  - ASM (standard)
  - ASM with 2x padding + bandlimit (match Mesh_GPU2 fn_FresnelPropagation_as_gpu)

IMPORTANT:
- 为了完全复现 Mesh_GPU2 的可视化效果，本脚本默认使用 Mesh 风格显示：
  log1p -> subtract percentile(20) -> scale by percentile(99.5) -> clip -> gamma -> u8

python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_ASM2.py \
 --amp_r /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_3bins_50_53mm/Hologram/amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_3bins_50_53mm/Hologram/pha_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_3bins_50_53mm/Hologram/amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_3bins_50_53mm/Hologram/pha_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_3bins_50_53mm/Hologram/amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_3bins_50_53mm/Hologram/pha_B.npy \
  --outdir /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_3bins_50_53mm/ASM_ReconFromHologram_65 \
  --mode asm \
  --wavelength 532e-9 \
  --pitch 4e-6 \
  --zmin 0.050 --zmax 0.053 --step 0.0001 \
  --gamma 0.65 \
  --vis_mode percentile \
  --p_low 1.0 --p_high 99.0 \
  --out_size 2048

python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_ASM2.py \
 --amp_r /workspace/yh/project/CGHReview/result/PointCloud3/PCD_complexRGB_4bins_3e8_50_53mm_2/Hologram/amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/PointCloud3/PCD_complexRGB_4bins_3e8_50_53mm_2/Hologram/phase_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/PointCloud3/PCD_complexRGB_4bins_3e8_50_53mm_2/Hologram/amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/PointCloud3/PCD_complexRGB_4bins_3e8_50_53mm_2/Hologram/phase_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/PointCloud3/PCD_complexRGB_4bins_3e8_50_53mm_2/Hologram/amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/PointCloud3/PCD_complexRGB_4bins_3e8_50_53mm_2/Hologram/phase_B.npy \
  --outdir /workspace/yh/project/CGHReview/result/PointCloud3/PCD_complexRGB_4bins_3e8_50_53mm_2/ASM_ReconFromHologram_65 \
  --mode asm \
  --wavelength 532e-9 \
  --pitch 5e-6 \
  --zmin 0.050 --zmax 0.053 --step 0.0001 \
  --gamma 0.65 \
  --vis_mode percentile \
  --p_low 1.0 --p_high 99.0 \
  --out_size 2048

Run example (Mesh hologram):
python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_ASM2.py \
  --amp_r  /workspace/yh/project/CGHReview/result/Mesh2/BunnyDragon_RGB_FromTxtColor_20k_5_1/Holograms/channel_Red_amp.npy \
  --pha_r  /workspace/yh/project/CGHReview/result/Mesh2/BunnyDragon_RGB_FromTxtColor_20k_5_1/Holograms/channel_Red_phase.npy \
  --amp_g  /workspace/yh/project/CGHReview/result/Mesh2/BunnyDragon_RGB_FromTxtColor_20k_5_1/Holograms/channel_Green_amp.npy \
  --pha_g  /workspace/yh/project/CGHReview/result/Mesh2/BunnyDragon_RGB_FromTxtColor_20k_5_1/Holograms/channel_Green_phase.npy \
  --amp_b  /workspace/yh/project/CGHReview/result/Mesh2/BunnyDragon_RGB_FromTxtColor_20k_5_1/Holograms/channel_Blue_amp.npy \
  --pha_b  /workspace/yh/project/CGHReview/result/Mesh2/BunnyDragon_RGB_FromTxtColor_20k_5_1/Holograms/channel_Blue_phase.npy \
  --outdir /workspace/yh/project/CGHReview/result/Mesh2/BunnyDragon_RGB_FromTxtColor_20k_5_1/ASM_ReconFromHologram \
  --mode asm_pad \
  --wavelength 532e-9 \
  --pitch 8e-6 \
  --zmin 4.5 --zmax 5.5 --step 0.01 \
  --gamma 1 --out_size 2048
  # 如果你保存前做过 flipud(H)，就加 --flipud

  python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_ASM2.py \
 --amp_r /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/pha_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/pha_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/pha_B.npy \
  --outdir /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/ASM_ReconFromHologram_65 \
  --mode asm \
  --wavelength 532e-9 \
  --pitch 4e-6 \
  --zmin 0.050 --zmax 0.055 --step 0.0001 \
  --gamma 0.65 \
  --vis_mode percentile \
  --p_low 1.0 --p_high 99.0 \
  --out_size 2048
"""

import os
# -*- coding: utf-8 -*-
"""
Reconstruction RGB from amp/phase holograms
Supports:
  - ASM (standard)
  - ASM with 2x padding + bandlimit (match Mesh_GPU2 fn_FresnelPropagation_as_gpu)

支持三种显示模式：
1) mesh_style:
   log1p -> subtract percentile(20) -> scale by percentile(99.5) -> clip -> gamma -> u8
2) percentile:
   clip((x - p_low) / (p_high - p_low)) -> gamma -> u8
   与多视重建脚本保持一致
3) minmax:
   全局 min-max -> gamma -> u8

python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_ASM2.py \
 --amp_r /workspace/yh/project/CGHReview/result/Layer/RGBD_complexRGB_3bins_50_53mm/Hologram/amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/Layer/RGBD_complexRGB_3bins_50_53mm/Hologram/pha_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/Layer/RGBD_complexRGB_3bins_50_53mm/Hologram/amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/Layer/RGBD_complexRGB_3bins_50_53mm/Hologram/pha_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/Layer/RGBD_complexRGB_3bins_50_53mm/Hologram/amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/Layer/RGBD_complexRGB_3bins_50_53mm/Hologram/pha_B.npy \
 --outdir /workspace/yh/project/CGHReview/result/Layer/RGBD_complexRGB_3bins_50_53mm/ASM_ReconFromHologram_percentile \
 --mode asm \
 --wavelength 532e-9 \
 --pitch 4e-6 \
 --zmin 0.050 --zmax 0.053 --step 0.0001 \
 --gamma 0.9 \
 --vis_mode percentile \
 --p_low 1.0 --p_high 99.0 \
 --out_size 2048
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGB Reconstruction from Amplitude/Phase Holograms

Author: Hao Yun
Date: 2026-03-25

Description:
    This script reconstructs RGB images from complex holograms represented by
    amplitude and phase `.npy` files. It supports:

        1. Standard ASM propagation
        2. ASM with 2x padding + bandlimit, aligned with Mesh_GPU2（mesh_style）

Visualization modes:

    1. percentile
       clip((x - p_low) / (p_high - p_low)) -> gamma -> u8

    2. minmax
       global min-max -> gamma -> u8

Example:
    Layer:
        python ./model/Reconstruction_layer.py \
        --amp_r /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_3bins_50_53mm/Hologram/amp_R.npy \
        --pha_r /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_3bins_50_53mm/Hologram/pha_R.npy \
        --amp_g /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_3bins_50_53mm/Hologram/amp_G.npy \
        --pha_g /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_3bins_50_53mm/Hologram/pha_G.npy \
        --amp_b /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_3bins_50_53mm/Hologram/amp_B.npy \
        --pha_b /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_3bins_50_53mm/Hologram/pha_B.npy \
        --outdir /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_3bins_50_53mm/ASM_ReconFromHologram_test \
        --mode asm \
        --wavelength 532e-9 \
        --pitch 4e-6 \
        --zmin 0.050 --zmax 0.053 --step 0.0001 \
        --gamma 0.65 \
        --vis_mode percentile \
        --p_low 1.0 --p_high 99.0 \
        --out_size 2048

    PCD:
        python ./model/Reconstruction_layer.py \
        --amp_r /workspace/yh/project/CGHReview/result/PointCloud3/PCD_complexRGB_4bins_3e8_50_53mm_2/Hologram/amp_R.npy \
        --pha_r /workspace/yh/project/CGHReview/result/PointCloud3/PCD_complexRGB_4bins_3e8_50_53mm_2/Hologram/phase_R.npy \
        --amp_g /workspace/yh/project/CGHReview/result/PointCloud3/PCD_complexRGB_4bins_3e8_50_53mm_2/Hologram/amp_G.npy \
        --pha_g /workspace/yh/project/CGHReview/result/PointCloud3/PCD_complexRGB_4bins_3e8_50_53mm_2/Hologram/phase_G.npy \
        --amp_b /workspace/yh/project/CGHReview/result/PointCloud3/PCD_complexRGB_4bins_3e8_50_53mm_2/Hologram/amp_B.npy \
        --pha_b /workspace/yh/project/CGHReview/result/PointCloud3/PCD_complexRGB_4bins_3e8_50_53mm_2/Hologram/phase_B.npy \
        --outdir /workspace/yh/project/CGHReview/result/PointCloud3/PCD_complexRGB_4bins_3e8_50_53mm_2/ASM_ReconFromHologram_test \
        --mode asm \
        --wavelength 532e-9 \
        --pitch 5e-6 \
        --zmin 0.050 --zmax 0.053 --step 0.0001 \
        --gamma 0.65 \
        --vis_mode percentile \
        --p_low 1.0 --p_high 99.0 \
        --out_size 2048

    LF:
        python ./model/Reconstruction_layer.py \
        --amp_r /workspace/yh/project/CGHReview/result/LightField3/pers_RGB_ComplexHologram_40_800/Hologram/hologram_amp_R.npy \
        --pha_r /workspace/yh/project/CGHReview/result/LightField3/pers_RGB_ComplexHologram_40_800/Hologram/hologram_phase_R.npy \
        --amp_g /workspace/yh/project/CGHReview/result/LightField3/pers_RGB_ComplexHologram_40_800/Hologram/hologram_amp_G.npy \
        --pha_g /workspace/yh/project/CGHReview/result/LightField3/pers_RGB_ComplexHologram_40_800/Hologram/hologram_phase_G.npy \
        --amp_b /workspace/yh/project/CGHReview/result/LightField3/pers_RGB_ComplexHologram_40_800/Hologram/hologram_amp_B.npy \
        --pha_b /workspace/yh/project/CGHReview/result/LightField3/pers_RGB_ComplexHologram_40_800/Hologram/hologram_phase_B.npy \
        --outdir /workspace/yh/project/CGHReview/result/LightField3/pers_RGB_ComplexHologram_40_800/ASM_ReconFromHologram_test \
        --mode asm \
        --wavelength 532e-9 \
        --pitch 2e-6 \
        --zmin -0.055 \
        --zmax -0.045 \
        --step 0.0001 \
        --gamma 0.65 \
        --vis_mode percentile \
        --p_low 1.0 --p_high 99.0 \
        --out_size 2048
    
    Mesh:
        python ./model/Reconstruction_layer.py \
        --amp_r  /workspace/yh/project/CGHReview/result/Mesh3_n/BunnyDragon_RGB_FromTxtColor_50k_y0/Holograms/channel_Red_amp.npy \
        --pha_r  /workspace/yh/project/CGHReview/result/Mesh3_n/BunnyDragon_RGB_FromTxtColor_50k_y0/Holograms/channel_Red_phase.npy \
        --amp_g  /workspace/yh/project/CGHReview/result/Mesh3_n/BunnyDragon_RGB_FromTxtColor_50k_y0/Holograms/channel_Green_amp.npy \
        --pha_g  /workspace/yh/project/CGHReview/result/Mesh3_n/BunnyDragon_RGB_FromTxtColor_50k_y0/Holograms/channel_Green_phase.npy \
        --amp_b  /workspace/yh/project/CGHReview/result/Mesh3_n/BunnyDragon_RGB_FromTxtColor_50k_y0/Holograms/channel_Blue_amp.npy \
        --pha_b  /workspace/yh/project/CGHReview/result/Mesh3_n/BunnyDragon_RGB_FromTxtColor_50k_y0/Holograms/channel_Blue_phase.npy \
        --outdir /workspace/yh/project/CGHReview/result/Mesh3_n/BunnyDragon_RGB_FromTxtColor_50k_y0/ASM_ReconFromHologram_test \
        --mode asm_pad \
        --wavelength 532e-9 \
        --pitch 8e-6 \
        --zmin 4.5 --zmax 5.5 --step 0.01 \
        --gamma 1 --out_size 2048

    Voxel:
        python ./model/Reconstruction_layer.py \
        --amp_r /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/amp_R.npy \
        --pha_r /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/pha_R.npy \
        --amp_g /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/amp_G.npy \
        --pha_g /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/pha_G.npy \
        --amp_b /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/amp_B.npy \
        --pha_b /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/pha_B.npy \
        --outdir /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/ASM_ReconFromHologram_test \
        --mode asm \
        --wavelength 532e-9 \
        --pitch 4e-6 \
        --zmin 0.050 --zmax 0.055 --step 0.0001 \
        --gamma 0.65 \
        --vis_mode percentile \
        --p_low 1.0 --p_high 99.0 \
        --out_size 2048
"""

import os
import argparse
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
import cv2


# =========================================================
# I/O: amplitude / phase loading
# =========================================================
def load_phase(path: str) -> np.ndarray:
    """Load a 2D phase array from .npy."""
    phase = np.load(path)
    if phase.ndim != 2:
        raise ValueError(f"phase.npy must be 2D, got {phase.shape}")
    return phase.astype(np.float32)


def load_amp(path: str) -> np.ndarray:
    """Load a 2D amplitude array from .npy."""
    amp = np.load(path)
    if amp.ndim != 2:
        raise ValueError(f"amp.npy must be 2D, got {amp.shape}")
    amp = amp.astype(np.float32)
    return np.clip(amp, 0.0, None)


def load_complex_hologram(amp_path: str, phase_path: str) -> np.ndarray:
    """Reconstruct a complex hologram from amplitude and phase."""
    amp = load_amp(amp_path)
    phase = load_phase(phase_path)
    if amp.shape != phase.shape:
        raise ValueError(f"amp/phase shape mismatch: {amp.shape} vs {phase.shape}")
    return (amp * np.exp(1j * phase)).astype(np.complex64)


# =========================================================
# Resize to square output
# =========================================================
def resize_keep_aspect_to_square(
    img_u8: np.ndarray,
    target: int = 2048,
    pad_value: int = 0
) -> np.ndarray:
    """Resize an RGB image to a square canvas while keeping aspect ratio."""
    height, width, channels = img_u8.shape
    scale = target / max(height, width)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(img_u8, (new_width, new_height), interpolation=cv2.INTER_AREA)

    output = np.full((target, target, 3), pad_value, dtype=np.uint8)
    y0 = (target - new_height) // 2
    x0 = (target - new_width) // 2
    output[y0:y0 + new_height, x0:x0 + new_width, :] = resized
    return output


# =========================================================
# z list parsing
# =========================================================
def parse_z_list(args):
    """Parse z positions either from --z_list or from zmin/zmax/step."""
    if args.z_list.strip():
        return [float(s) for s in args.z_list.split(",") if s.strip()]

    zmin, zmax, step = args.zmin, args.zmax, args.step
    n = int(np.floor((zmax - zmin) / step + 1e-12)) + 1
    return [zmin + i * step for i in range(n)]


# =========================================================
# ASM (standard)
# =========================================================
def asm_propagate(
    u_in: np.ndarray,
    dist_m: float,
    wavelength_m: float,
    pitch_m: float
) -> np.ndarray:
    """Standard Angular Spectrum Method propagation."""
    u_in = np.asarray(u_in, dtype=np.complex64)
    height, width = u_in.shape

    dfx = 1.0 / (width * pitch_m)
    dfy = 1.0 / (height * pitch_m)
    fx = (np.arange(width) - width / 2.0) * dfx
    fy = (np.arange(height) - height / 2.0) * dfy
    FX, FY = np.meshgrid(fx, fy, indexing="xy")

    k = 2.0 * np.pi / wavelength_m
    term = 1.0 - (wavelength_m * FX) ** 2 - (wavelength_m * FY) ** 2

    transfer_function = np.zeros_like(term, dtype=np.complex128)
    mask = term >= 0
    transfer_function[mask] = np.exp(1j * k * dist_m * np.sqrt(term[mask]).astype(np.float64))

    U_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_in)))
    U_z = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U_fft * transfer_function)))
    return U_z.astype(np.complex64)


# =========================================================
# ASM_PAD (strictly aligned to Mesh_GPU2)
# =========================================================
def asm_propagate_pad_bandlimit(
    u_in: np.ndarray,
    dist_m: float,
    wavelength_m: float,
    pitch_m: float
) -> np.ndarray:
    """
    ASM with 2x padding and bandlimit,
    aligned with Mesh_GPU2 fn_FresnelPropagation_as_gpu.
    """
    inp = np.asarray(u_in, dtype=np.complex128)
    Ny, Nx = inp.shape

    Nxx, Nyy = 2 * Nx, 2 * Ny
    inp2 = np.zeros((Nyy, Nxx), dtype=np.complex128)

    start_x = int(round(Nx / 2.0) - 1)
    start_y = int(round(Ny / 2.0) - 1)
    inp2[start_y:start_y + Ny, start_x:start_x + Nx] = inp

    dal = 1.0 / (Nxx * pitch_m)
    dbl = 1.0 / (Nyy * pitch_m)

    al_1d = (np.arange(1, Nxx + 1, dtype=np.float64) - (Nxx / 2.0)) * dal
    bl_1d = (np.arange(1, Nyy + 1, dtype=np.float64) - (Nyy / 2.0)) * dbl
    AL, BL = np.meshgrid(al_1d, bl_1d)

    A = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(inp2)))

    denom = np.sqrt((1.0 / (wavelength_m ** 2) - AL ** 2 - BL ** 2) + 0j)
    prop_kernel = np.exp(1j * 2.0 * np.pi * dist_m * denom)

    sx, sy = 0.0, 0.0
    fla = np.abs(-AL * dist_m / denom + sx)
    flb = np.abs(-BL * dist_m / denom + sy)

    prop_kernel = np.where(fla > 1.0 / (2.0 * dal), 0.0, prop_kernel)
    prop_kernel = np.where(flb > 1.0 / (2.0 * dbl), 0.0, prop_kernel)

    intermediate = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(A * prop_kernel)))
    out = intermediate[start_y:start_y + Ny, start_x:start_x + Nx]
    return out.astype(np.complex64)


# =========================================================
# Visualization
# =========================================================
def mesh_style_rgb_visualize(amp_rgb: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Mesh_GPU2 style visualization:
      log1p -> subtract percentile(20) -> scale by percentile(99.5)
      -> clip -> gamma -> u8
    """
    vis = np.log1p(np.maximum(amp_rgb.astype(np.float32), 0.0))

    bg = np.percentile(vis, 20.0)
    vis = np.maximum(vis - bg, 0.0)

    p_hi = np.percentile(vis, 99.5)
    img = np.clip(vis / (float(p_hi) + 1e-30), 0.0, 1.0)

    if gamma is not None and abs(gamma - 1.0) > 1e-12:
        img = np.power(img, float(gamma))

    img_u8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img_u8


def normalize_rgb_percentile_with_gamma(
    rgb_float: np.ndarray,
    gamma: float = 1.0,
    p_low: float = 1.0,
    p_high: float = 99.0
) -> np.ndarray:
    """
    Percentile-based normalization with gamma correction.
    """
    x = np.asarray(rgb_float, dtype=np.float32)
    x = np.clip(x, 0.0, None)

    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    x = np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)

    if gamma is not None and abs(gamma - 1.0) > 1e-12:
        x = np.power(x, float(gamma))

    return (x * 255.0).clip(0, 255).astype(np.uint8)


def normalize_rgb_global_with_gamma(
    rgb_float: np.ndarray,
    gamma: float = 1.0
) -> np.ndarray:
    """Global min-max normalization with gamma correction."""
    x = np.asarray(rgb_float, dtype=np.float32)
    x = np.clip(x, 0.0, None)

    x = x - x.min()
    maxv = float(x.max())
    if maxv > 1e-8:
        x = x / maxv

    if gamma is not None and abs(gamma - 1.0) > 1e-12:
        x = np.power(np.clip(x, 0.0, 1.0), float(gamma))

    return (x * 255.0).clip(0, 255).astype(np.uint8)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="RGB reconstruction from complex holograms (amp.npy + phase.npy), supporting ASM / ASM_PAD."
    )

    parser.add_argument("--amp_r", required=True)
    parser.add_argument("--pha_r", required=True)
    parser.add_argument("--amp_g", required=True)
    parser.add_argument("--pha_g", required=True)
    parser.add_argument("--amp_b", required=True)
    parser.add_argument("--pha_b", required=True)

    parser.add_argument("--outdir", required=True)

    parser.add_argument("--wavelength", type=float, default=532e-9)
    parser.add_argument("--pitch", type=float, default=2e-6)

    parser.add_argument("--mode", type=str, default="asm", choices=["asm", "asm_pad"])
    parser.add_argument(
        "--flipud",
        action="store_true",
        help="Flip holograms vertically before propagation (undo generation flipud)."
    )

    parser.add_argument("--z_list", type=str, default="")
    parser.add_argument("--zmin", type=float, default=-0.053)
    parser.add_argument("--zmax", type=float, default=-0.050)
    parser.add_argument("--step", type=float, default=1e-4)

    parser.add_argument("--use_intensity", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--out_size", type=int, default=2048)

    parser.add_argument(
        "--vis_mode",
        type=str,
        default="percentile",
        choices=["mesh_style", "percentile", "minmax"],
        help="Visualization mode."
    )
    parser.add_argument(
        "--p_low",
        type=float,
        default=1.0,
        help="Lower percentile for percentile mode."
    )
    parser.add_argument(
        "--p_high",
        type=float,
        default=99.0,
        help="Upper percentile for percentile mode."
    )

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("[INFO] Loading complex holograms ...")
    holo_R = load_complex_hologram(args.amp_r, args.pha_r)
    holo_G = load_complex_hologram(args.amp_g, args.pha_g)
    holo_B = load_complex_hologram(args.amp_b, args.pha_b)

    if args.flipud:
        holo_R = np.flipud(holo_R)
        holo_G = np.flipud(holo_G)
        holo_B = np.flipud(holo_B)
        print("[INFO] Applied flipud() to holograms before propagation.")

    if args.mode == "asm":
        propagate = asm_propagate
        print("[INFO] Mode = ASM (standard)")
    else:
        propagate = asm_propagate_pad_bandlimit
        print("[INFO] Mode = ASM_PAD (2x padding + bandlimit, aligned with Mesh_GPU2)")

    z_list = parse_z_list(args)
    print(f"[INFO] z count = {len(z_list)}")
    print(f"[INFO] vis_mode = {args.vis_mode}")

    for z in tqdm(z_list, desc=f"Recon ({args.mode}) RGB"):
        rec_R = propagate(holo_R, z, args.wavelength, args.pitch)
        rec_G = propagate(holo_G, z, args.wavelength, args.pitch)
        rec_B = propagate(holo_B, z, args.wavelength, args.pitch)

        if args.use_intensity:
            img_R = (np.abs(rec_R) ** 2).astype(np.float32)
            img_G = (np.abs(rec_G) ** 2).astype(np.float32)
            img_B = (np.abs(rec_B) ** 2).astype(np.float32)
        else:
            img_R = np.abs(rec_R).astype(np.float32)
            img_G = np.abs(rec_G).astype(np.float32)
            img_B = np.abs(rec_B).astype(np.float32)

        amp_rgb = np.stack([img_R, img_G, img_B], axis=-1)

        if args.vis_mode == "mesh_style":
            rgb_u8 = mesh_style_rgb_visualize(amp_rgb, gamma=args.gamma)
        elif args.vis_mode == "percentile":
            rgb_u8 = normalize_rgb_percentile_with_gamma(
                amp_rgb,
                gamma=args.gamma,
                p_low=args.p_low,
                p_high=args.p_high
            )
        else:
            rgb_u8 = normalize_rgb_global_with_gamma(amp_rgb, gamma=args.gamma)

        rgb_u8 = resize_keep_aspect_to_square(rgb_u8, target=args.out_size)
        out_path = os.path.join(args.outdir, f"recon_rgb_z_{z:+.6f}m.png")
        imageio.imwrite(out_path, rgb_u8)

    print("[DONE] Output:", args.outdir)


if __name__ == "__main__":
    main()