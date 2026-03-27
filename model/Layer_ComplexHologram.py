#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Complex RGB Holograms from Layered RGB-D Data and Reconstruct Depth Slices

Author: Hao Yun
Date: 2026-03-25

Description:
    This script converts layered RGB-D data (e.g., LDI-style rgb_layer*.png +
    depth_layer*.npy pairs) into sparse depth bins, generates complex RGB
    holograms via angular spectrum propagation, and reconstructs all depth slices.

Pipeline:
    1. Load layered RGB + depth pairs
    2. Discretize depths into K bins
    3. Keep only the front-most valid pixel per bin
    4. Generate complex RGB holograms using shared random phase across channels
    5. Save complex holograms and amplitude/phase maps
    6. Reconstruct all depth slices and save RGB images

Example:
    python ./model/Layer_ComplexHologram.py \
        --data_root ./dataset/Layer/BunnyDragonColor_RGBD \
        --out_root ./result/Layer/RGBD_3bins \
        --target_size 2048 \
        --z_min_mm 50.0 \
        --z_max_mm 53.0 \
        --num_bins 3 \
        --bg_depth_mm 100.0 \
        --wavelength 532e-9 \
        --pixel_pitch 4e-6 \
        --gamma 0.65 \
        --out_size 2048
"""

import os
import re
import glob
import time
import argparse
import numpy as np
import cv2
from tqdm import tqdm


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate complex RGB holograms from layered RGB-D data and reconstruct all depth slices."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Input directory containing rgb_layer*.png and depth_layer*.npy pairs."
    )
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Output root directory."
    )

    parser.add_argument(
        "--target_size",
        type=int,
        default=2048,
        help="Target square size for input RGB/depth resizing."
    )

    parser.add_argument(
        "--z_min_mm",
        type=float,
        default=50.0,
        help="Minimum valid object depth in mm."
    )
    parser.add_argument(
        "--z_max_mm",
        type=float,
        default=53.0,
        help="Maximum valid object depth in mm."
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=1000,
        help="Number of depth bins."
    )

    parser.add_argument(
        "--bg_depth_mm",
        type=float,
        default=100.0,
        help="Background depth value in mm."
    )
    parser.add_argument(
        "--bg_eps_mm",
        type=float,
        default=0.5,
        help="Tolerance used to distinguish background depth."
    )

    parser.add_argument(
        "--intensity_thresh",
        type=float,
        default=1e-4,
        help="Luminance threshold for valid pixels."
    )

    parser.add_argument(
        "--wavelength",
        type=float,
        default=532e-9,
        help="Optical wavelength in meters."
    )
    parser.add_argument(
        "--pixel_pitch",
        type=float,
        default=4e-6,
        help="Pixel pitch in meters."
    )
    parser.add_argument(
        "--holo_z_m",
        type=float,
        default=0.0,
        help="Hologram plane position in meters."
    )

    parser.add_argument(
        "--rng_seed",
        type=int,
        default=2024,
        help="Random seed for shared random phase generation."
    )

    parser.add_argument(
        "--use_intensity",
        action="store_true",
        help="Use |U|^2 instead of |U| for reconstruction display."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.65,
        help="Gamma value for reconstruction display."
    )
    parser.add_argument(
        "--out_size",
        type=int,
        default=2048,
        help="Final saved reconstruction size (square, with padding)."
    )

    parser.add_argument(
        "--amp_mode",
        type=str,
        default="rgb",
        choices=["rgb", "sqrt_rgb"],
        help='Amplitude mode for object field generation: "rgb" or "sqrt_rgb".'
    )

    parser.add_argument(
        "--vis_mode",
        type=str,
        default="percentile",
        choices=["percentile", "minmax"],
        help='Visualization mode for reconstructed RGB: "percentile" or "minmax".'
    )
    parser.add_argument(
        "--p_low",
        type=float,
        default=1.0,
        help="Lower percentile used when --vis_mode percentile."
    )
    parser.add_argument(
        "--p_high",
        type=float,
        default=99.0,
        help="Upper percentile used when --vis_mode percentile."
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# ASM Propagation
# -----------------------------------------------------------------------------
def asm_propagate(u_in: np.ndarray, dist: float, wavelength: float, pitch: float) -> np.ndarray:
    """
    Angular spectrum propagation with evanescent-wave masking.
    """
    u_in = np.asarray(u_in, dtype=np.complex64)
    height, width = u_in.shape

    dfx = 1.0 / (height * pitch)
    dfy = 1.0 / (width * pitch)

    fx = (np.arange(height) - height / 2) * dfx
    fy = (np.arange(width) - width / 2) * dfy
    fx_grid, fy_grid = np.meshgrid(fx, fy, indexing="ij")

    k = 2.0 * np.pi / wavelength
    term = 1.0 - (wavelength * fx_grid) ** 2 - (wavelength * fy_grid) ** 2

    transfer = np.zeros_like(term, dtype=np.complex64)
    valid = term >= 0
    transfer[valid] = np.exp(1j * k * dist * np.sqrt(term[valid]).astype(np.float32))

    u_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_in)))
    u_z = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(u_fft * transfer)))
    return u_z


# -----------------------------------------------------------------------------
# Input Loading
# -----------------------------------------------------------------------------
def load_ldi_pairs(data_root: str):
    """
    Find all valid rgb_layer*.png / depth_layer*.npy pairs.
    """
    rgb_list = glob.glob(os.path.join(data_root, "rgb_layer*.png"))
    pairs = []

    for rgb_path in rgb_list:
        match = re.search(r"rgb_layer(\d+)\.png$", os.path.basename(rgb_path))
        if not match:
            continue
        layer_idx = int(match.group(1))
        depth_path = os.path.join(data_root, f"depth_layer{layer_idx}.npy")
        if os.path.exists(depth_path):
            pairs.append((layer_idx, rgb_path, depth_path))

    pairs.sort(key=lambda x: x[0])

    if not pairs:
        raise RuntimeError(f"No valid rgb_layer*.png / depth_layer*.npy pairs found in: {data_root}")

    return pairs


def read_rgb_depth(rgb_path: str, depth_path: str, target_size: int):
    """
    Load one RGB/depth pair and resize to a square target size.
    """
    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read RGB image: {rgb_path}")

    bgr = cv2.resize(bgr, (target_size, target_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    depth_mm = np.load(depth_path).astype(np.float32)
    if depth_mm.shape[:2] != (target_size, target_size):
        depth_mm = cv2.resize(depth_mm, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

    return rgb, depth_mm


# -----------------------------------------------------------------------------
# Depth Discretization
# -----------------------------------------------------------------------------
def discretize_keep_frontmost_in_bin_rgb(
    pairs,
    target_size: int,
    z_min_mm: float,
    z_max_mm: float,
    num_bins: int,
    bg_depth_mm: float,
    bg_eps_mm: float,
    intensity_thresh: float
):
    """
    Discretize layered RGB-D data into sparse RGB bins.
    For each bin, keep only the front-most valid pixel at each image location.
    """
    edges = np.linspace(z_min_mm, z_max_mm, num_bins + 1, dtype=np.float32)
    centers = 0.5 * (edges[:-1] + edges[1:])

    rgb0, _ = read_rgb_depth(pairs[0][1], pairs[0][2], target_size)
    height, width, _ = rgb0.shape

    rgb_bins = [np.zeros((height, width, 3), dtype=np.float32) for _ in range(num_bins)]
    depth_bins = [np.full((height, width), np.inf, dtype=np.float32) for _ in range(num_bins)]
    mask_bins = [np.zeros((height, width), dtype=np.uint8) for _ in range(num_bins)]
    counts = np.zeros((num_bins,), dtype=np.int64)

    for layer_idx, rgb_path, depth_path in tqdm(
        pairs,
        desc=f"Discretizing layered RGB-D into {num_bins} sparse depth bins"
    ):
        rgb, depth_mm = read_rgb_depth(rgb_path, depth_path, target_size)

        luminance = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
        not_background = np.abs(depth_mm - bg_depth_mm) > bg_eps_mm
        valid = not_background & (luminance > intensity_thresh)

        if not np.any(valid):
            continue

        bin_ids = np.digitize(depth_mm[valid], edges) - 1
        bin_ids = np.clip(bin_ids, 0, num_bins - 1)

        ys, xs = np.where(valid)

        for bin_idx in range(num_bins):
            selected = (bin_ids == bin_idx)
            if not np.any(selected):
                continue

            yy = ys[selected]
            xx = xs[selected]
            d_new = depth_mm[yy, xx]

            closer = d_new < depth_bins[bin_idx][yy, xx]
            if np.any(closer):
                y_keep = yy[closer]
                x_keep = xx[closer]

                depth_bins[bin_idx][y_keep, x_keep] = depth_mm[y_keep, x_keep]
                rgb_bins[bin_idx][y_keep, x_keep, :] = rgb[y_keep, x_keep, :]
                mask_bins[bin_idx][y_keep, x_keep] = 255
                counts[bin_idx] += int(np.sum(closer))

    return edges, centers, rgb_bins, mask_bins, counts


def save_binned_planes_rgb(
    out_dir_bin: str,
    centers: np.ndarray,
    rgb_bins,
    mask_bins,
    counts: np.ndarray
):
    """
    Save sparse RGB planes and masks for each depth bin.
    """
    os.makedirs(out_dir_bin, exist_ok=True)

    for bin_idx in range(len(centers)):
        plane_dir = os.path.join(out_dir_bin, f"plane_{bin_idx:04d}_z{centers[bin_idx]:.3f}mm")
        os.makedirs(plane_dir, exist_ok=True)

        rgb_u8 = (np.clip(rgb_bins[bin_idx], 0.0, 1.0) * 255.0).astype(np.uint8)
        cv2.imwrite(os.path.join(plane_dir, "rgb.png"), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(plane_dir, "mask.png"), mask_bins[bin_idx])

        with open(os.path.join(plane_dir, "depth_mm.txt"), "w", encoding="utf-8") as f:
            f.write(f"{centers[bin_idx]:.6f}\n")

        print(f"[OK] bin={bin_idx:04d}, center={centers[bin_idx]:.3f} mm, kept_pixels={counts[bin_idx]}")


# -----------------------------------------------------------------------------
# Hologram Generation
# -----------------------------------------------------------------------------
def generate_hologram_complex_rgb(
    rgb_bins,
    mask_bins,
    centers_mm: np.ndarray,
    wavelength: float,
    pixel_pitch: float,
    holo_z_m: float,
    rng_seed: int,
    amp_mode: str
):
    """
    Generate complex RGB holograms from sparse RGB depth bins.
    A shared random phase is used across R/G/B channels within the same depth bin.
    """
    rng = np.random.default_rng(rng_seed)
    height, width, _ = rgb_bins[0].shape

    holograms = [
        np.zeros((height, width), dtype=np.complex64),
        np.zeros((height, width), dtype=np.complex64),
        np.zeros((height, width), dtype=np.complex64),
    ]

    for bin_idx in tqdm(
        range(len(centers_mm)),
        desc="Generating complex RGB holograms from depth bins"
    ):
        mask = mask_bins[bin_idx] > 0
        if not np.any(mask):
            continue

        rgb = np.clip(rgb_bins[bin_idx], 0.0, 1.0).astype(np.float32)
        amplitude = np.sqrt(rgb) if amp_mode == "sqrt_rgb" else rgb

        phase = rng.uniform(-np.pi, np.pi, size=(height, width)).astype(np.float32)
        phasor = np.exp(1j * phase).astype(np.complex64)

        z_m = float(centers_mm[bin_idx]) * 1e-3
        dist = float(holo_z_m - z_m)

        for channel in range(3):
            u_obj = np.zeros((height, width), dtype=np.complex64)
            u_obj[mask] = amplitude[..., channel][mask].astype(np.complex64) * phasor[mask]
            holograms[channel] += asm_propagate(u_obj, dist, wavelength, pixel_pitch)

    return holograms[0], holograms[1], holograms[2]


# -----------------------------------------------------------------------------
# Save Helpers
# -----------------------------------------------------------------------------
def save_float_image(x: np.ndarray, path: str):
    """
    Save a float array as an 8-bit grayscale image after min-max normalization.
    """
    x = np.asarray(x, dtype=np.float32)
    x = x - x.min()
    max_val = float(x.max())
    if max_val > 1e-8:
        x = x / max_val
    image = (x * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(path, image)


def save_phase_image(phase: np.ndarray, path: str):
    """
    Save phase in [-pi, pi] as an 8-bit grayscale image.
    """
    phase = np.asarray(phase, dtype=np.float32)
    image = ((phase + np.pi) / (2 * np.pi) * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(path, image)


def normalize_rgb_percentile_with_gamma(
    rgb_float: np.ndarray,
    gamma: float = 1.0,
    p_low: float = 1.0,
    p_high: float = 99.0
) -> np.ndarray:
    """
    Percentile-based normalization:
      x -> percentile clip/normalize -> gamma -> u8
    """
    x = np.asarray(rgb_float, dtype=np.float32)
    x = np.clip(x, 0.0, None)

    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    x = np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)

    if gamma is not None and abs(gamma - 1.0) > 1e-12:
        x = np.power(x, float(gamma))

    return (x * 255.0).clip(0, 255).astype(np.uint8)


def normalize_rgb_global_with_gamma(rgb_float: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Global min-max normalization followed by gamma correction.
    """
    x = np.asarray(rgb_float, dtype=np.float32)
    x = np.clip(x, 0.0, None)

    x = x - x.min()
    max_val = float(x.max())
    if max_val > 1e-8:
        x = x / max_val

    if gamma is not None and abs(gamma - 1.0) > 1e-12:
        x = np.power(np.clip(x, 0.0, 1.0), float(gamma))

    return (x * 255.0).clip(0, 255).astype(np.uint8)


def resize_keep_aspect_to_square_rgb(img_u8: np.ndarray, target: int = 2048, pad_value: int = 0) -> np.ndarray:
    """
    Resize an RGB image to fit inside a square canvas with padding.
    """
    height, width, _ = img_u8.shape
    scale = float(target) / float(max(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(img_u8, (new_width, new_height), interpolation=cv2.INTER_AREA)

    output = np.full((target, target, 3), pad_value, dtype=np.uint8)
    y0 = (target - new_height) // 2
    x0 = (target - new_width) // 2
    output[y0:y0 + new_height, x0:x0 + new_width, :] = resized
    return output


# -----------------------------------------------------------------------------
# Reconstruction
# -----------------------------------------------------------------------------
def reconstruct_slices_rgb(
    holo_r: np.ndarray,
    holo_g: np.ndarray,
    holo_b: np.ndarray,
    centers_mm: np.ndarray,
    out_dir_recon: str,
    wavelength: float,
    pixel_pitch: float,
    holo_z_m: float,
    use_intensity: bool,
    gamma: float,
    out_size: int,
    vis_mode: str,
    p_low: float,
    p_high: float,
):
    """
    Reconstruct and save all depth slices from RGB holograms.
    """
    os.makedirs(out_dir_recon, exist_ok=True)

    for bin_idx in tqdm(range(len(centers_mm)), desc="Reconstructing all depth slices"):
        z_m = float(centers_mm[bin_idx]) * 1e-3
        dist = float(z_m - holo_z_m)

        rec_r = asm_propagate(holo_r, dist, wavelength, pixel_pitch)
        rec_g = asm_propagate(holo_g, dist, wavelength, pixel_pitch)
        rec_b = asm_propagate(holo_b, dist, wavelength, pixel_pitch)

        if use_intensity:
            img_r = (np.abs(rec_r) ** 2).astype(np.float32)
            img_g = (np.abs(rec_g) ** 2).astype(np.float32)
            img_b = (np.abs(rec_b) ** 2).astype(np.float32)
        else:
            img_r = np.abs(rec_r).astype(np.float32)
            img_g = np.abs(rec_g).astype(np.float32)
            img_b = np.abs(rec_b).astype(np.float32)

        rgb = np.stack([img_r, img_g, img_b], axis=-1)

        if vis_mode == "percentile":
            rgb_u8 = normalize_rgb_percentile_with_gamma(
                rgb,
                gamma=gamma,
                p_low=p_low,
                p_high=p_high,
            )
        else:
            rgb_u8 = normalize_rgb_global_with_gamma(rgb, gamma=gamma)

        rgb_u8 = resize_keep_aspect_to_square_rgb(rgb_u8, target=out_size, pad_value=0)

        out_path = os.path.join(
            out_dir_recon,
            f"recon_plane_{bin_idx:04d}_z{centers_mm[bin_idx]:.3f}mm.png"
        )
        cv2.imwrite(out_path, cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.target_size <= 0:
        raise ValueError("--target_size must be positive.")
    if args.num_bins <= 0:
        raise ValueError("--num_bins must be positive.")
    if args.z_max_mm <= args.z_min_mm:
        raise ValueError("--z_max_mm must be greater than --z_min_mm.")
    if args.out_size <= 0:
        raise ValueError("--out_size must be positive.")
    if not (0.0 <= args.p_low < args.p_high <= 100.0):
        raise ValueError("--p_low and --p_high must satisfy 0 <= p_low < p_high <= 100.")

    dir_bin = os.path.join(args.out_root, "Binned")
    dir_holo = os.path.join(args.out_root, "Hologram")
    dir_recon = os.path.join(args.out_root, "Recon")

    os.makedirs(args.out_root, exist_ok=True)
    os.makedirs(dir_bin, exist_ok=True)
    os.makedirs(dir_holo, exist_ok=True)
    os.makedirs(dir_recon, exist_ok=True)

    start_time = time.time()

    pairs = load_ldi_pairs(args.data_root)
    print(f"[INFO] Found {len(pairs)} RGB-D layers: {[p[0] for p in pairs]}")

    edges, centers, rgb_bins, mask_bins, counts = discretize_keep_frontmost_in_bin_rgb(
        pairs=pairs,
        target_size=args.target_size,
        z_min_mm=args.z_min_mm,
        z_max_mm=args.z_max_mm,
        num_bins=args.num_bins,
        bg_depth_mm=args.bg_depth_mm,
        bg_eps_mm=args.bg_eps_mm,
        intensity_thresh=args.intensity_thresh,
    )

    save_binned_planes_rgb(
        out_dir_bin=dir_bin,
        centers=centers,
        rgb_bins=rgb_bins,
        mask_bins=mask_bins,
        counts=counts
    )

    holo_r, holo_g, holo_b = generate_hologram_complex_rgb(
        rgb_bins=rgb_bins,
        mask_bins=mask_bins,
        centers_mm=centers,
        wavelength=args.wavelength,
        pixel_pitch=args.pixel_pitch,
        holo_z_m=args.holo_z_m,
        rng_seed=args.rng_seed,
        amp_mode=args.amp_mode,
    )

    np.save(os.path.join(dir_holo, "holo_R.npy"), holo_r.astype(np.complex64))
    np.save(os.path.join(dir_holo, "holo_G.npy"), holo_g.astype(np.complex64))
    np.save(os.path.join(dir_holo, "holo_B.npy"), holo_b.astype(np.complex64))

    amp_r = np.clip(np.abs(holo_r).astype(np.float32), 0.0, None)
    amp_g = np.clip(np.abs(holo_g).astype(np.float32), 0.0, None)
    amp_b = np.clip(np.abs(holo_b).astype(np.float32), 0.0, None)

    pha_r = np.angle(holo_r).astype(np.float32)
    pha_g = np.angle(holo_g).astype(np.float32)
    pha_b = np.angle(holo_b).astype(np.float32)

    np.save(os.path.join(dir_holo, "amp_R.npy"), amp_r)
    np.save(os.path.join(dir_holo, "amp_G.npy"), amp_g)
    np.save(os.path.join(dir_holo, "amp_B.npy"), amp_b)

    np.save(os.path.join(dir_holo, "pha_R.npy"), pha_r)
    np.save(os.path.join(dir_holo, "pha_G.npy"), pha_g)
    np.save(os.path.join(dir_holo, "pha_B.npy"), pha_b)

    save_float_image(amp_r, os.path.join(dir_holo, "amp_R.png"))
    save_float_image(amp_g, os.path.join(dir_holo, "amp_G.png"))
    save_float_image(amp_b, os.path.join(dir_holo, "amp_B.png"))

    save_phase_image(pha_r, os.path.join(dir_holo, "phase_R.png"))
    save_phase_image(pha_g, os.path.join(dir_holo, "phase_G.png"))
    save_phase_image(pha_b, os.path.join(dir_holo, "phase_B.png"))

    hologram_end_time = time.time()
    print(f"[INFO] Hologram generation time: {hologram_end_time - start_time:.2f} s")

    reconstruct_slices_rgb(
        holo_r=holo_r,
        holo_g=holo_g,
        holo_b=holo_b,
        centers_mm=centers,
        out_dir_recon=dir_recon,
        wavelength=args.wavelength,
        pixel_pitch=args.pixel_pitch,
        holo_z_m=args.holo_z_m,
        use_intensity=args.use_intensity,
        gamma=args.gamma,
        out_size=args.out_size,
        vis_mode=args.vis_mode,
        p_low=args.p_low,
        p_high=args.p_high,
    )

    end_time = time.time()

    print("[DONE]")
    print(f"Binned planes : {dir_bin}")
    print(f"Holograms     : {dir_holo}")
    print(f"Recon slices  : {dir_recon}")
    print(
        f"[INFO] Display settings: "
        f"vis_mode={args.vis_mode}, p_low={args.p_low}, p_high={args.p_high}, "
        f"use_intensity={args.use_intensity}, gamma={args.gamma}, out_size={args.out_size}"
    )
    print(f"[INFO] Total runtime: {end_time - start_time:.2f} s")


if __name__ == "__main__":
    main()