#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi-view reconstruction from complex RGB holograms
(amp_R/G/B.npy + phase_R/G/B.npy)

Author: Hao Yun
Date: 2026-04-01

Pipeline:
1) load complex hologram U_h^c = amp * exp(j*phase)
2) ASM propagate to depth z -> U_z^c
3) Fourier-plane sub-aperture selection (crop a window around shifted center) -> different views
4) IFFT -> view field -> |U| or |U|^2
5) RGB global normalize + gamma, resize to out_size (pad)

Output structure:
outdir/
  z_+0.060000m/
    v_00_00.png
    v_00_01.png
    ...


python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_mutiview.py \
 --amp_r /workspace/yh/project/CGHReview/result/Layer/LDI_single_complexRGB_20bins_50_53mm/Hologram/amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/Layer/LDI_single_complexRGB_20bins_50_53mm/Hologram/pha_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/Layer/LDI_single_complexRGB_20bins_50_53mm/Hologram/amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/Layer/LDI_single_complexRGB_20bins_50_53mm/Hologram/pha_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/Layer/LDI_single_complexRGB_20bins_50_53mm/Hologram/amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/Layer/LDI_single_complexRGB_20bins_50_53mm/Hologram/pha_B.npy \
  --outdir /workspace/yh/project/CGHReview/result/Layer/LDI_single_complexRGB_20bins_50_53mm/Multiview_ReconFromHologram \
 --wavelength 532e-9 \
 --pitch 4e-6 \
 --zmin 0.05 --zmax 0.052 --step 0.0005 \
 --gamma 0.9 \
 --out_size 2048 \
 --view_grid 9 \
 --aperture_ratio 0.35 \
 --max_shift_ratio 0.8 \
 --save_mosaic

 python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_mutiview.py \
 --amp_r /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_100bins_50_53mm/Hologram/amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_100bins_50_53mm/Hologram/pha_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_100bins_50_53mm/Hologram/amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_100bins_50_53mm/Hologram/pha_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_100bins_50_53mm/Hologram/amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_100bins_50_53mm/Hologram/pha_B.npy \
  --outdir /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_100bins_50_53mm/Multiview_ReconFromHologram_65 \
 --wavelength 532e-9 \
 --pitch 4e-6 \
 --vis_mode percentile \
 --zmin 0.051 --zmax 0.052 --step 0.001 \
 --gamma 0.65 \
 --out_size 2048 \
 --view_grid 9 \
 --aperture_ratio 0.35 \
 --max_shift_ratio 0.8 \
 --save_mosaic

 python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_mutiview.py \
 --amp_r /workspace/yh/project/CGHReview/result/PointCloud3/PCD_complexRGB_100bins_3e8_50_53mm_2/Hologram/amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/PointCloud3/PCD_complexRGB_100bins_3e8_50_53mm_2/Hologram/phase_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/PointCloud3/PCD_complexRGB_100bins_3e8_50_53mm_2/Hologram/amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/PointCloud3/PCD_complexRGB_100bins_3e8_50_53mm_2/Hologram/phase_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/PointCloud3/PCD_complexRGB_100bins_3e8_50_53mm_2/Hologram/amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/PointCloud3/PCD_complexRGB_100bins_3e8_50_53mm_2/Hologram/phase_B.npy \
 --outdir /workspace/yh/project/CGHReview/result/PointCloud3/PCD_complexRGB_100bins_3e8_50_53mm_2/Multiview_ReconFromHologram_65 \
 --wavelength 532e-9 \
 --pitch 5e-6 \
 --zmin 0.051 --zmax 0.052 --step 0.001 \
 --vis_mode percentile \
 --gamma 0.65 \
 --out_size 2048 \
 --view_grid 9 \
 --aperture_ratio 0.35 \
 --max_shift_ratio 0.8 \
 --save_mosaic


python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_mutiview.py \
 --amp_r /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Hologram/hologram_amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Hologram/hologram_phase_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Hologram/hologram_amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Hologram/hologram_phase_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Hologram/hologram_amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Hologram/hologram_phase_B.npy \
 --outdir /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Multiview_ReconFromHologram_1 \
 --wavelength 532e-9 \
 --pitch 2e-6 \
 --zmin -0.052 --zmax -0.050 --step 0.0005 \
 --gamma 1 \
 --out_size 2048 \
 --view_grid 9 \
 --aperture_ratio 0.35 \
 --max_shift_ratio 0.8 \
 --save_mosaic

python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_mutiview.py \
 --amp_r /workspace/yh/project/CGHReview/result/LightField3/pers_RGB_ComplexHologram_40_800/Hologram/hologram_amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/LightField3/pers_RGB_ComplexHologram_40_800/Hologram/hologram_phase_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/LightField3/pers_RGB_ComplexHologram_40_800/Hologram/hologram_amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/LightField3/pers_RGB_ComplexHologram_40_800/Hologram/hologram_phase_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/LightField3/pers_RGB_ComplexHologram_40_800/Hologram/hologram_amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/LightField3/pers_RGB_ComplexHologram_40_800/Hologram/hologram_phase_B.npy \
 --outdir /workspace/yh/project/CGHReview/result/LightField3/pers_RGB_ComplexHologram_40_800/Multiview_ReconFromHologram_0.65 \
 --wavelength 532e-9 \
 --vis_mode percentile \
 --pitch 2e-6 \
 --zmin -0.051 --zmax -0.050 --step 0.001 \
 --gamma 0.65 \
 --out_size 2048 \
 --view_grid 9 \
 --aperture_ratio 0.35 \
 --max_shift_ratio 0.8 \
 --save_mosaic

    python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_mutiview.py \
 --amp_r /workspace/yh/project/CGHReview/result/Mesh3_n/50k_8/Holograms/channel_Red_amp.npy \
 --pha_r /workspace/yh/project/CGHReview/result/Mesh3_n/50k_8/Holograms/channel_Red_phase.npy \
 --amp_g /workspace/yh/project/CGHReview/result/Mesh3_n/50k_8/Holograms/channel_Green_amp.npy \
 --pha_g /workspace/yh/project/CGHReview/result/Mesh3_n/50k_8/Holograms/channel_Green_phase.npy \
 --amp_b /workspace/yh/project/CGHReview/result/Mesh3_n/50k_8/Holograms/channel_Blue_amp.npy \
 --pha_b /workspace/yh/project/CGHReview/result/Mesh3_n/50k_8/Holograms/channel_Blue_phase.npy \
 --outdir /workspace/yh/project/CGHReview/result/Mesh3_n/50k_8/Multiview_ReconFromHologram \
  --wavelength 532e-9 \
  --pitch 8e-6 \
  --mode asm_pad \
  --remove_dc 1 \
  --vis_mode mesh_style \
  --zmin 1.9 --zmax 1.95 --step 0.05 \
  --view_grid 9 --aperture_ratio 0.35 --max_shift_ratio 0.8 \
  --gamma 0.9 --out_size 2048 --save_mosaic


 python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_mutiview.py \
 --amp_r /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/pha_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/pha_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/pha_B.npy \
  --outdir /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Multiview_ReconFromHologram_65 \
 --wavelength 532e-9 \
 --pitch 4e-6 \
 --zmin 0.051 --zmax 0.053 --step 0.001 \
 --vis_mode percentile \
 --gamma 0.65 \
 --out_size 2048 \
 --view_grid 9 \
 --aperture_ratio 0.35 \
 --max_shift_ratio 0.8 \
 --save_mosaic
 
"""

import os
import argparse
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
import cv2


# =========================
# IO: load amplitude / phase
# =========================
def load_phase(path: str) -> np.ndarray:
    phase = np.load(path)
    if phase.ndim != 2:
        raise ValueError(f"phase.npy must be 2D, got {phase.shape}")
    return phase.astype(np.float32)


def load_amp(path: str) -> np.ndarray:
    amp = np.load(path)
    if amp.ndim != 2:
        raise ValueError(f"amp.npy must be 2D, got {amp.shape}")
    amp = amp.astype(np.float32)
    return np.clip(amp, 0.0, None)


def load_complex_hologram(amp_path: str, phase_path: str) -> np.ndarray:
    amp = load_amp(amp_path)
    phase = load_phase(phase_path)
    if amp.shape != phase.shape:
        raise ValueError("amp / phase shape mismatch")
    return (amp * np.exp(1j * phase)).astype(np.complex64)


# =========================
# ASM propagation (standard)
# =========================
def asm_propagate(u_in: np.ndarray, dist_m: float, wavelength_m: float, pitch_m: float) -> np.ndarray:
    """
    Angular Spectrum Method propagation with evanescent wave masking.
    u_in: complex field (H,W)
    dist_m: propagation distance in meters (+ forward)
    """
    u_in = np.asarray(u_in, dtype=np.complex64)
    H, W = u_in.shape

    # fx corresponds to columns(W), fy corresponds to rows(H)
    dfx = 1.0 / (W * pitch_m)
    dfy = 1.0 / (H * pitch_m)
    fx = (np.arange(W) - W / 2.0) * dfx
    fy = (np.arange(H) - H / 2.0) * dfy
    FX, FY = np.meshgrid(fx, fy, indexing="xy")

    k = 2.0 * np.pi / wavelength_m
    term = 1.0 - (wavelength_m * FX) ** 2 - (wavelength_m * FY) ** 2

    Htf = np.zeros_like(term, dtype=np.complex128)
    mask = term >= 0
    Htf[mask] = np.exp(1j * k * dist_m * np.sqrt(term[mask]).astype(np.float64))

    U_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_in)))
    U_z = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U_fft * Htf)))
    return U_z.astype(np.complex64)


# =========================
# ASM_PAD (match Mesh_GPU2 fn_FresnelPropagation_as_gpu)
# =========================
def asm_propagate_pad_bandlimit(u_in: np.ndarray, dist_m: float, wavelength_m: float, pitch_m: float) -> np.ndarray:
    """
    Strictly matches Mesh_GPU2 (2x padding + bandlimit):
      - pad to (2H,2W)
      - propagate in frequency with denom = sqrt(1/lam^2 - fx^2 - fy^2)
      - apply bandlimit via fla/flb threshold
      - crop back to original size using Mesh-style start indices (round(N/2)-1)
    """
    inp = np.asarray(u_in, dtype=np.complex128)
    Ny, Nx = inp.shape

    Nxx, Nyy = 2 * Nx, 2 * Ny
    inp2 = np.zeros((Nyy, Nxx), dtype=np.complex128)

    start_x = Nxx // 2 - Nx // 2
    start_y = Nyy // 2 - Ny // 2
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


# =========================
# Display / Visualization
# =========================
def normalize_rgb_global_with_gamma(rgb_float: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    x = np.asarray(rgb_float, dtype=np.float32)
    x = np.clip(x, 0.0, None)

    x = x - x.min()
    maxv = float(x.max())
    if maxv > 1e-8:
        x = x / maxv

    if gamma is not None and abs(gamma - 1.0) > 1e-12:
        x = np.power(np.clip(x, 0.0, 1.0), float(gamma))

    return (x * 255.0).clip(0, 255).astype(np.uint8)

def normalize_rgb_percentile_with_gamma(
    rgb_float: np.ndarray,
    gamma: float = 1.0,
    p_low: float = 1.0,
    p_high: float = 99.0
) -> np.ndarray:
    x = np.asarray(rgb_float, dtype=np.float32)
    x = np.clip(x, 0.0, None)

    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    x = np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)

    if gamma is not None and abs(gamma - 1.0) > 1e-12:
        x = np.power(x, float(gamma))

    return (x * 255.0).clip(0, 255).astype(np.uint8)


def mesh_style_rgb_visualize(amp_rgb: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Mesh_GPU2 style visualization (joint RGB):
      log1p -> subtract percentile(20) -> clip -> scale by percentile(99.5) -> gamma -> u8
    """
    vis = np.log1p(np.maximum(amp_rgb.astype(np.float32), 0.0))

    bg = np.percentile(vis, 20.0)
    vis = np.maximum(vis - bg, 0.0)

    p_hi = np.percentile(vis, 99.5)
    img = np.clip(vis / (float(p_hi) + 1e-30), 0.0, 1.0)

    if gamma is not None and abs(gamma - 1.0) > 1e-12:
        img = np.power(img, float(gamma))

    return (img * 255.0).clip(0, 255).astype(np.uint8)


def resize_keep_aspect_to_square(img_u8: np.ndarray, target: int = 2048, pad_value: int = 0) -> np.ndarray:
    """
    img_u8: (H,W,3) uint8
    return: (target,target,3) uint8
    """
    h, w, _ = img_u8.shape
    scale = float(target) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img_u8, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out = np.full((target, target, 3), pad_value, dtype=np.uint8)
    y0 = (target - new_h) // 2
    x0 = (target - new_w) // 2
    out[y0:y0 + new_h, x0:x0 + new_w, :] = resized
    return out

def compute_global_percentile_range(rgb_list, p_low=1.0, p_high=99.0):
    vals = []
    for x in rgb_list:
        x = np.asarray(x, np.float32)
        x = np.clip(x, 0.0, None)
        vals.append(x.reshape(-1))
    vals = np.concatenate(vals, axis=0)
    lo = np.percentile(vals, p_low)
    hi = np.percentile(vals, p_high)
    return float(lo), float(hi)


def apply_fixed_range_with_gamma(rgb_float, lo, hi, gamma=1.0):
    x = np.asarray(rgb_float, dtype=np.float32)
    x = np.clip(x, 0.0, None)
    x = np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)

    if gamma is not None and abs(gamma - 1.0) > 1e-12:
        x = np.power(x, float(gamma))

    return (x * 255.0).clip(0, 255).astype(np.uint8)


# =========================
# Parse z list
# =========================
def parse_z_list(args):
    if args.z_list.strip():
        return [float(s) for s in args.z_list.split(",") if s.strip()]

    zmin, zmax, step = args.zmin, args.zmax, args.step
    n = int(np.floor((zmax - zmin) / step + 1e-12)) + 1
    return [zmin + i * step for i in range(n)]


# =========================
# Sub-aperture / multi-view
# =========================
def spectrum_of_field(U: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U))).astype(np.complex64)


def subaperture_view_from_spectrum(F: np.ndarray,
                                   crop: int,
                                   shift_x: int,
                                   shift_y: int,
                                   recenter: bool = True) -> np.ndarray:
    """
    F: centered spectrum of a complex field at a given depth (H,W)
    crop: window size in frequency domain (crop x crop)
    shift_x, shift_y: shift of the subaperture center (in pixels, frequency-plane coords)
    recenter: if True, place selected subaperture back to spectral center before IFFT
    Return: complex field after subaperture selection (H,W)
    """
    H, W = F.shape

    cy = H // 2 + int(shift_y)
    cx = W // 2 + int(shift_x)

    half = crop // 2
    y0 = cy - half
    y1 = y0 + crop
    x0 = cx - half
    x1 = x0 + crop

    # clamp
    y0c = max(0, y0)
    x0c = max(0, x0)
    y1c = min(H, y1)
    x1c = min(W, x1)

    F_win = np.zeros_like(F, dtype=np.complex64)
    if recenter:
        # Recenter selected subaperture to keep perspective shift in amplitude views.
        dy0 = y0c - y0
        dx0 = x0c - x0
        hh = y1c - y0c
        ww = x1c - x0c
        cy0 = (H // 2) - half + dy0
        cx0 = (W // 2) - half + dx0
        cy1 = cy0 + hh
        cx1 = cx0 + ww
        F_win[cy0:cy1, cx0:cx1] = F[y0c:y1c, x0c:x1c]
    else:
        F_win[y0c:y1c, x0c:x1c] = F[y0c:y1c, x0c:x1c]

    U_view = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F_win)))
    return U_view.astype(np.complex64)


def make_view_shifts(grid: int, max_shift: int):
    """
    grid: number of views per dimension, e.g. 9 -> 9x9 views
    max_shift: maximum frequency shift in pixels (center to edge)
    Return: list of (iy, ix, shift_x, shift_y)
    """
    if grid <= 1:
        return [(0, 0, 0, 0)]

    coords = np.linspace(-max_shift, +max_shift, grid, dtype=np.float32)
    shifts = []
    for iy in range(grid):
        for ix in range(grid):
            shifts.append((iy, ix, int(round(coords[ix])), int(round(coords[iy]))))
    return shifts


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Multi-view (light-field style) RGB reconstruction from COMPLEX hologram (amp.npy + phase.npy)."
    )

    # hologram inputs
    parser.add_argument("--amp_r", required=True)
    parser.add_argument("--pha_r", required=True)
    parser.add_argument("--amp_g", required=True)
    parser.add_argument("--pha_g", required=True)
    parser.add_argument("--amp_b", required=True)
    parser.add_argument("--pha_b", required=True)

    parser.add_argument("--outdir", required=True)

    # optics
    parser.add_argument("--wavelength", type=float, default=532e-9)
    parser.add_argument("--pitch", type=float, default=2e-6)

    # propagation mode (NEW)
    parser.add_argument("--mode", type=str, default="asm", choices=["asm", "asm_pad"],
                        help="asm: standard ASM; asm_pad: 2x padding + bandlimit (match Mesh_GPU2).")

    # optional pre-processing (NEW)
    parser.add_argument("--flipud_holo", action="store_true",
                        help="Flip hologram vertically before propagation (undo generation flipud).")
    parser.add_argument("--remove_dc", type=int, default=0, choices=[0, 1],
                        help="If 1, subtract mean from hologram before propagation (reduce zero-order).")

    # z
    parser.add_argument("--z_list", type=str, default="")
    parser.add_argument("--zmin", type=float, default=-0.051)
    parser.add_argument("--zmax", type=float, default=-0.048)
    parser.add_argument("--step", type=float, default=5e-4)

    # image / display
    parser.add_argument("--use_intensity", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--out_size", type=int, default=2048)

    # visualization mode (NEW)
    parser.add_argument("--vis_mode", type=str, default="percentile",
                    choices=["minmax", "percentile", "mesh_style"])
    parser.add_argument("--p_low", type=float, default=1.0)
    parser.add_argument("--p_high", type=float, default=99.0)

    # LF-like multi-view params
    parser.add_argument("--view_grid", type=int, default=9, help="NxN views")
    parser.add_argument("--aperture_ratio", type=float, default=0.35, help="subaperture crop size ratio (0-1)")
    parser.add_argument("--max_shift_ratio", type=float, default=0.18, help="max shift ratio of half-size (0-1)")
    parser.add_argument("--save_mosaic", action="store_true", help="also save one mosaic per z")
    parser.add_argument("--recenter_subaperture", type=int, default=1, choices=[0, 1],
                        help="Recenter selected subaperture before IFFT (recommended).")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("[INFO] Load complex holograms ...")
    holo_R = load_complex_hologram(args.amp_r, args.pha_r)
    holo_G = load_complex_hologram(args.amp_g, args.pha_g)
    holo_B = load_complex_hologram(args.amp_b, args.pha_b)

    H, W = holo_R.shape
    if holo_G.shape != (H, W) or holo_B.shape != (H, W):
        raise ValueError("RGB hologram shapes mismatch")

    if args.flipud_holo:
        holo_R = np.flipud(holo_R)
        holo_G = np.flipud(holo_G)
        holo_B = np.flipud(holo_B)
        print("[INFO] Applied flipud() to holograms before propagation.")

    z_list = parse_z_list(args)
    print(f"[INFO] z count = {len(z_list)}")

    # choose propagation function
    if args.mode == "asm":
        propagate = asm_propagate
        print("[INFO] Propagation mode = ASM (standard)")
    else:
        propagate = asm_propagate_pad_bandlimit
        print("[INFO] Propagation mode = ASM_PAD (2x padding + bandlimit, match Mesh_GPU2)")

    # subaperture sizes
    crop = int(round(min(H, W) * float(args.aperture_ratio)))
    crop = max(8, crop)
    if crop % 2 == 1:
        crop += 1

    max_shift_req = int(round(0.5 * min(H, W) * float(args.max_shift_ratio)))
    max_shift_cap = max(0, (min(H, W) // 2) - (crop // 2) - 1)
    max_shift = min(max_shift_req, max_shift_cap)
    if max_shift < max_shift_req:
        print(f"[WARN] max_shift clipped from {max_shift_req} to {max_shift} to keep full subaperture in-band.")
    shifts = make_view_shifts(args.view_grid, max_shift)

    print(f"[INFO] view_grid={args.view_grid} -> {len(shifts)} views")
    print(f"[INFO] aperture crop={crop}px, max_shift={max_shift}px")
    print(f"[INFO] vis_mode={args.vis_mode}, remove_dc={args.remove_dc}")

    for z in tqdm(z_list, desc="LF Multi-view (RGB)"):
        # optional DC removal (match your Mesh recon habit)
        hR, hG, hB = holo_R, holo_G, holo_B
        if args.remove_dc == 1:
            hR = hR - np.mean(hR)
            hG = hG - np.mean(hG)
            hB = hB - np.mean(hB)

        # 1) focus to depth
        U_R = propagate(hR, z, args.wavelength, args.pitch)
        U_G = propagate(hG, z, args.wavelength, args.pitch)
        U_B = propagate(hB, z, args.wavelength, args.pitch)

        F_R = spectrum_of_field(U_R)
        F_G = spectrum_of_field(U_G)
        F_B = spectrum_of_field(U_B)

        z_dir = os.path.join(args.outdir, f"z_{z:+.6f}m")
        os.makedirs(z_dir, exist_ok=True)

        mosaic = None
        if args.save_mosaic:
            mosaic = np.zeros((args.view_grid * args.out_size, args.view_grid * args.out_size, 3), dtype=np.uint8)

        for (iy, ix, sx, sy) in shifts:
            # 2) subaperture -> view
            vR = subaperture_view_from_spectrum(
                F_R, crop=crop, shift_x=sx, shift_y=sy, recenter=bool(args.recenter_subaperture)
            )
            vG = subaperture_view_from_spectrum(
                F_G, crop=crop, shift_x=sx, shift_y=sy, recenter=bool(args.recenter_subaperture)
            )
            vB = subaperture_view_from_spectrum(
                F_B, crop=crop, shift_x=sx, shift_y=sy, recenter=bool(args.recenter_subaperture)
            )

            # 3) amplitude/intensity
            if args.use_intensity:
                img_R = (np.abs(vR) ** 2).astype(np.float32)
                img_G = (np.abs(vG) ** 2).astype(np.float32)
                img_B = (np.abs(vB) ** 2).astype(np.float32)
            else:
                img_R = np.abs(vR).astype(np.float32)
                img_G = np.abs(vG).astype(np.float32)
                img_B = np.abs(vB).astype(np.float32)

            rgb = np.stack([img_R, img_G, img_B], axis=-1)  # (H,W,3)

            # 4) visualize
            if args.vis_mode == "mesh_style":
                rgb_u8 = mesh_style_rgb_visualize(rgb, gamma=args.gamma)
            elif args.vis_mode == "percentile":
                rgb_u8 = normalize_rgb_percentile_with_gamma(
                    rgb, gamma=args.gamma, p_low=args.p_low, p_high=args.p_high
                )
            else:
                rgb_u8 = normalize_rgb_global_with_gamma(rgb, gamma=args.gamma)
                        # 5) resize/pad
            rgb_u8 = resize_keep_aspect_to_square(rgb_u8, target=args.out_size, pad_value=0)

            out_path = os.path.join(z_dir, f"v_{iy:02d}_{ix:02d}.png")
            imageio.imwrite(out_path, rgb_u8)

            if mosaic is not None:
                y0 = iy * args.out_size
                x0 = ix * args.out_size
                mosaic[y0:y0 + args.out_size, x0:x0 + args.out_size, :] = rgb_u8

        if mosaic is not None:
            imageio.imwrite(os.path.join(z_dir, "mosaic.png"), mosaic)

    print("[DONE] Output:", args.outdir)


if __name__ == "__main__":
    main()

    # #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import os
# import argparse
# import numpy as np
# import imageio.v2 as imageio
# from tqdm import tqdm
# import cv2


# # =========================
# # IO: load amplitude / phase
# # =========================
# def load_phase(path: str) -> np.ndarray:
#     phase = np.load(path)
#     if phase.ndim != 2:
#         raise ValueError(f"phase.npy must be 2D, got {phase.shape}")
#     return phase.astype(np.float32)


# def load_amp(path: str) -> np.ndarray:
#     amp = np.load(path)
#     if amp.ndim != 2:
#         raise ValueError(f"amp.npy must be 2D, got {amp.shape}")
#     amp = amp.astype(np.float32)
#     return np.clip(amp, 0.0, None)


# def load_complex_hologram(amp_path: str, phase_path: str) -> np.ndarray:
#     amp = load_amp(amp_path)
#     phase = load_phase(phase_path)
#     if amp.shape != phase.shape:
#         raise ValueError("amp / phase shape mismatch")
#     return (amp * np.exp(1j * phase)).astype(np.complex64)


# # =========================
# # ASM propagation (standard)
# # =========================
# def asm_propagate(u_in: np.ndarray, dist_m: float, wavelength_m: float, pitch_m: float) -> np.ndarray:
#     u_in = np.asarray(u_in, dtype=np.complex64)
#     H, W = u_in.shape

#     dfx = 1.0 / (W * pitch_m)
#     dfy = 1.0 / (H * pitch_m)
#     fx = (np.arange(W) - W / 2.0) * dfx
#     fy = (np.arange(H) - H / 2.0) * dfy
#     FX, FY = np.meshgrid(fx, fy, indexing="xy")

#     k = 2.0 * np.pi / wavelength_m
#     term = 1.0 - (wavelength_m * FX) ** 2 - (wavelength_m * FY) ** 2

#     Htf = np.zeros_like(term, dtype=np.complex128)
#     mask = term >= 0
#     Htf[mask] = np.exp(1j * k * dist_m * np.sqrt(term[mask]).astype(np.float64))

#     U_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_in)))
#     U_z = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U_fft * Htf)))
#     return U_z.astype(np.complex64)


# # =========================
# # ASM_PAD
# # =========================
# def asm_propagate_pad_bandlimit(u_in: np.ndarray, dist_m: float, wavelength_m: float, pitch_m: float) -> np.ndarray:
#     inp = np.asarray(u_in, dtype=np.complex128)
#     Ny, Nx = inp.shape

#     Nxx, Nyy = 2 * Nx, 2 * Ny
#     inp2 = np.zeros((Nyy, Nxx), dtype=np.complex128)

#     start_x = Nxx // 2 - Nx // 2
#     start_y = Nyy // 2 - Ny // 2
#     inp2[start_y:start_y + Ny, start_x:start_x + Nx] = inp

#     dal = 1.0 / (Nxx * pitch_m)
#     dbl = 1.0 / (Nyy * pitch_m)

#     al_1d = (np.arange(1, Nxx + 1, dtype=np.float64) - (Nxx / 2.0)) * dal
#     bl_1d = (np.arange(1, Nyy + 1, dtype=np.float64) - (Nyy / 2.0)) * dbl
#     AL, BL = np.meshgrid(al_1d, bl_1d)

#     A = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(inp2)))

#     denom = np.sqrt((1.0 / (wavelength_m ** 2) - AL ** 2 - BL ** 2) + 0j)
#     prop_kernel = np.exp(1j * 2.0 * np.pi * dist_m * denom)

#     sx, sy = 0.0, 0.0
#     fla = np.abs(-AL * dist_m / denom + sx)
#     flb = np.abs(-BL * dist_m / denom + sy)

#     prop_kernel = np.where(fla > 1.0 / (2.0 * dal), 0.0, prop_kernel)
#     prop_kernel = np.where(flb > 1.0 / (2.0 * dbl), 0.0, prop_kernel)

#     intermediate = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(A * prop_kernel)))
#     out = intermediate[start_y:start_y + Ny, start_x:start_x + Nx]
#     return out.astype(np.complex64)


# # =========================
# # Display / Visualization
# # =========================
# def normalize_rgb_minmax_with_gamma(rgb_float: np.ndarray, gamma: float = 1.0) -> np.ndarray:
#     x = np.asarray(rgb_float, dtype=np.float32)
#     x = np.clip(x, 0.0, None)

#     x = x - x.min()
#     maxv = float(x.max())
#     if maxv > 1e-8:
#         x = x / maxv

#     if gamma is not None and abs(gamma - 1.0) > 1e-12:
#         x = np.power(np.clip(x, 0.0, 1.0), float(gamma))

#     return (x * 255.0).clip(0, 255).astype(np.uint8)


# def normalize_rgb_percentile_with_gamma(
#     rgb_float: np.ndarray,
#     gamma: float = 1.0,
#     p_low: float = 1.0,
#     p_high: float = 99.0
# ) -> np.ndarray:
#     x = np.asarray(rgb_float, dtype=np.float32)
#     x = np.clip(x, 0.0, None)

#     lo = np.percentile(x, p_low)
#     hi = np.percentile(x, p_high)
#     x = np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)

#     if gamma is not None and abs(gamma - 1.0) > 1e-12:
#         x = np.power(x, float(gamma))

#     return (x * 255.0).clip(0, 255).astype(np.uint8)


# def mesh_style_rgb_visualize(amp_rgb: np.ndarray, gamma: float = 1.0) -> np.ndarray:
#     vis = np.log1p(np.maximum(amp_rgb.astype(np.float32), 0.0))

#     bg = np.percentile(vis, 20.0)
#     vis = np.maximum(vis - bg, 0.0)

#     p_hi = np.percentile(vis, 99.5)
#     img = np.clip(vis / (float(p_hi) + 1e-30), 0.0, 1.0)

#     if gamma is not None and abs(gamma - 1.0) > 1e-12:
#         img = np.power(img, float(gamma))

#     return (img * 255.0).clip(0, 255).astype(np.uint8)


# def compute_global_percentile_range(rgb_list, p_low=1.0, p_high=99.0):
#     vals = []
#     for x in rgb_list:
#         x = np.asarray(x, np.float32)
#         x = np.clip(x, 0.0, None)
#         vals.append(x.reshape(-1))
#     vals = np.concatenate(vals, axis=0)

#     lo = np.percentile(vals, p_low)
#     hi = np.percentile(vals, p_high)
#     return float(lo), float(hi)


# def apply_fixed_range_with_gamma(rgb_float, lo, hi, gamma=1.0):
#     x = np.asarray(rgb_float, dtype=np.float32)
#     x = np.clip(x, 0.0, None)
#     x = np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)

#     if gamma is not None and abs(gamma - 1.0) > 1e-12:
#         x = np.power(x, float(gamma))

#     return (x * 255.0).clip(0, 255).astype(np.uint8)


# def resize_keep_aspect_to_square(img_u8: np.ndarray, target: int = 2048, pad_value: int = 0) -> np.ndarray:
#     h, w, _ = img_u8.shape
#     scale = float(target) / float(max(h, w))
#     new_w = max(1, int(round(w * scale)))
#     new_h = max(1, int(round(h * scale)))

#     resized = cv2.resize(img_u8, (new_w, new_h), interpolation=cv2.INTER_AREA)

#     out = np.full((target, target, 3), pad_value, dtype=np.uint8)
#     y0 = (target - new_h) // 2
#     x0 = (target - new_w) // 2
#     out[y0:y0 + new_h, x0:x0 + new_w, :] = resized
#     return out


# # =========================
# # Parse z list
# # =========================
# def parse_z_list(args):
#     if args.z_list.strip():
#         return [float(s) for s in args.z_list.split(",") if s.strip()]

#     zmin, zmax, step = args.zmin, args.zmax, args.step
#     n = int(np.floor((zmax - zmin) / step + 1e-12)) + 1
#     return [zmin + i * step for i in range(n)]


# # =========================
# # Sub-aperture / multi-view
# # =========================
# def spectrum_of_field(U: np.ndarray) -> np.ndarray:
#     return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U))).astype(np.complex64)


# def subaperture_view_from_spectrum(F: np.ndarray,
#                                    crop: int,
#                                    shift_x: int,
#                                    shift_y: int,
#                                    recenter: bool = True) -> np.ndarray:
#     H, W = F.shape

#     cy = H // 2 + int(shift_y)
#     cx = W // 2 + int(shift_x)

#     half = crop // 2
#     y0 = cy - half
#     y1 = y0 + crop
#     x0 = cx - half
#     x1 = x0 + crop

#     y0c = max(0, y0)
#     x0c = max(0, x0)
#     y1c = min(H, y1)
#     x1c = min(W, x1)

#     F_win = np.zeros_like(F, dtype=np.complex64)
#     if recenter:
#         dy0 = y0c - y0
#         dx0 = x0c - x0
#         hh = y1c - y0c
#         ww = x1c - x0c
#         cy0 = (H // 2) - half + dy0
#         cx0 = (W // 2) - half + dx0
#         cy1 = cy0 + hh
#         cx1 = cx0 + ww
#         F_win[cy0:cy1, cx0:cx1] = F[y0c:y1c, x0c:x1c]
#     else:
#         F_win[y0c:y1c, x0c:x1c] = F[y0c:y1c, x0c:x1c]

#     U_view = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F_win)))
#     return U_view.astype(np.complex64)


# def make_view_shifts(grid: int, max_shift: int):
#     if grid <= 1:
#         return [(0, 0, 0, 0)]

#     coords = np.linspace(-max_shift, +max_shift, grid, dtype=np.float32)
#     shifts = []
#     for iy in range(grid):
#         for ix in range(grid):
#             shifts.append((iy, ix, int(round(coords[ix])), int(round(coords[iy]))))
#     return shifts


# # =========================
# # Main
# # =========================
# def main():
#     parser = argparse.ArgumentParser(
#         description="Multi-view RGB reconstruction from complex hologram with shared normalization per-z."
#     )

#     parser.add_argument("--amp_r", required=True)
#     parser.add_argument("--pha_r", required=True)
#     parser.add_argument("--amp_g", required=True)
#     parser.add_argument("--pha_g", required=True)
#     parser.add_argument("--amp_b", required=True)
#     parser.add_argument("--pha_b", required=True)

#     parser.add_argument("--outdir", required=True)

#     parser.add_argument("--wavelength", type=float, default=532e-9)
#     parser.add_argument("--pitch", type=float, default=2e-6)

#     parser.add_argument("--mode", type=str, default="asm", choices=["asm", "asm_pad"])
#     parser.add_argument("--flipud_holo", action="store_true")
#     parser.add_argument("--remove_dc", type=int, default=0, choices=[0, 1])

#     parser.add_argument("--z_list", type=str, default="")
#     parser.add_argument("--zmin", type=float, default=-0.053)
#     parser.add_argument("--zmax", type=float, default=-0.050)
#     parser.add_argument("--step", type=float, default=1e-4)

#     parser.add_argument("--use_intensity", action="store_true")
#     parser.add_argument("--gamma", type=float, default=1.0)
#     parser.add_argument("--out_size", type=int, default=2048)

#     parser.add_argument("--vis_mode", type=str, default="percentile",
#                         choices=["minmax", "percentile", "mesh_style"])
#     parser.add_argument("--p_low", type=float, default=1.0)
#     parser.add_argument("--p_high", type=float, default=99.0)

#     parser.add_argument("--view_grid", type=int, default=9)
#     parser.add_argument("--aperture_ratio", type=float, default=0.35)
#     parser.add_argument("--max_shift_ratio", type=float, default=0.18)
#     parser.add_argument("--save_mosaic", action="store_true")
#     parser.add_argument("--recenter_subaperture", type=int, default=1, choices=[0, 1])

#     args = parser.parse_args()
#     os.makedirs(args.outdir, exist_ok=True)

#     print("[INFO] Load complex holograms ...")
#     holo_R = load_complex_hologram(args.amp_r, args.pha_r)
#     holo_G = load_complex_hologram(args.amp_g, args.pha_g)
#     holo_B = load_complex_hologram(args.amp_b, args.pha_b)

#     H, W = holo_R.shape
#     if holo_G.shape != (H, W) or holo_B.shape != (H, W):
#         raise ValueError("RGB hologram shapes mismatch")

#     if args.flipud_holo:
#         holo_R = np.flipud(holo_R)
#         holo_G = np.flipud(holo_G)
#         holo_B = np.flipud(holo_B)
#         print("[INFO] Applied flipud() to holograms before propagation.")

#     z_list = parse_z_list(args)
#     print(f"[INFO] z count = {len(z_list)}")

#     if args.mode == "asm":
#         propagate = asm_propagate
#         print("[INFO] Propagation mode = ASM")
#     else:
#         propagate = asm_propagate_pad_bandlimit
#         print("[INFO] Propagation mode = ASM_PAD")

#     crop = int(round(min(H, W) * float(args.aperture_ratio)))
#     crop = max(8, crop)
#     if crop % 2 == 1:
#         crop += 1

#     max_shift_req = int(round(0.5 * min(H, W) * float(args.max_shift_ratio)))
#     max_shift_cap = max(0, (min(H, W) // 2) - (crop // 2) - 1)
#     max_shift = min(max_shift_req, max_shift_cap)
#     if max_shift < max_shift_req:
#         print(f"[WARN] max_shift clipped from {max_shift_req} to {max_shift}")
#     shifts = make_view_shifts(args.view_grid, max_shift)

#     print(f"[INFO] view_grid={args.view_grid} -> {len(shifts)} views")
#     print(f"[INFO] crop={crop}, max_shift={max_shift}, vis_mode={args.vis_mode}")

#     for z in tqdm(z_list, desc="LF Multi-view (RGB)"):
#         hR, hG, hB = holo_R, holo_G, holo_B
#         if args.remove_dc == 1:
#             hR = hR - np.mean(hR)
#             hG = hG - np.mean(hG)
#             hB = hB - np.mean(hB)

#         # 1) focus to depth
#         U_R = propagate(hR, z, args.wavelength, args.pitch)
#         U_G = propagate(hG, z, args.wavelength, args.pitch)
#         U_B = propagate(hB, z, args.wavelength, args.pitch)

#         F_R = spectrum_of_field(U_R)
#         F_G = spectrum_of_field(U_G)
#         F_B = spectrum_of_field(U_B)

#         z_dir = os.path.join(args.outdir, f"z_{z:+.6f}m")
#         os.makedirs(z_dir, exist_ok=True)

#         # 2) first pass: compute all 81 float RGB views
#         view_buffers = []
#         for (iy, ix, sx, sy) in shifts:
#             vR = subaperture_view_from_spectrum(
#                 F_R, crop=crop, shift_x=sx, shift_y=sy, recenter=bool(args.recenter_subaperture)
#             )
#             vG = subaperture_view_from_spectrum(
#                 F_G, crop=crop, shift_x=sx, shift_y=sy, recenter=bool(args.recenter_subaperture)
#             )
#             vB = subaperture_view_from_spectrum(
#                 F_B, crop=crop, shift_x=sx, shift_y=sy, recenter=bool(args.recenter_subaperture)
#             )

#             if args.use_intensity:
#                 img_R = (np.abs(vR) ** 2).astype(np.float32)
#                 img_G = (np.abs(vG) ** 2).astype(np.float32)
#                 img_B = (np.abs(vB) ** 2).astype(np.float32)
#             else:
#                 img_R = np.abs(vR).astype(np.float32)
#                 img_G = np.abs(vG).astype(np.float32)
#                 img_B = np.abs(vB).astype(np.float32)

#             rgb = np.stack([img_R, img_G, img_B], axis=-1)
#             view_buffers.append((iy, ix, rgb))

#         # 3) shared normalization range for all views under the same z
#         fixed_lo, fixed_hi = None, None
#         if args.vis_mode == "percentile":
#             fixed_lo, fixed_hi = compute_global_percentile_range(
#                 [rgb for _, _, rgb in view_buffers],
#                 p_low=args.p_low,
#                 p_high=args.p_high
#             )
#             print(f"[INFO] z={z:+.6f}: shared percentile range = [{fixed_lo:.6e}, {fixed_hi:.6e}]")

#         mosaic = None
#         if args.save_mosaic:
#             mosaic = np.zeros(
#                 (args.view_grid * args.out_size, args.view_grid * args.out_size, 3),
#                 dtype=np.uint8
#             )

#         # 4) second pass: visualize with the SAME mapping
#         for (iy, ix, rgb) in view_buffers:
#             if args.vis_mode == "mesh_style":
#                 rgb_u8 = mesh_style_rgb_visualize(rgb, gamma=args.gamma)
#             elif args.vis_mode == "percentile":
#                 rgb_u8 = apply_fixed_range_with_gamma(
#                     rgb, fixed_lo, fixed_hi, gamma=args.gamma
#                 )
#             else:
#                 rgb_u8 = normalize_rgb_minmax_with_gamma(rgb, gamma=args.gamma)

#             rgb_u8 = resize_keep_aspect_to_square(rgb_u8, target=args.out_size, pad_value=0)

#             out_path = os.path.join(z_dir, f"v_{iy:02d}_{ix:02d}.png")
#             imageio.imwrite(out_path, rgb_u8)

#             if mosaic is not None:
#                 y0 = iy * args.out_size
#                 x0 = ix * args.out_size
#                 mosaic[y0:y0 + args.out_size, x0:x0 + args.out_size, :] = rgb_u8

#         if mosaic is not None:
#             imageio.imwrite(os.path.join(z_dir, "mosaic.png"), mosaic)

#     print("[DONE] Output:", args.outdir)


# if __name__ == "__main__":
#     main()