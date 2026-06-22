#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Complex RGB Holograms from a Colored Point Cloud and Reconstruct Depth Bins

Author: Hao Yun
Date: 2026-03-25

Description:
    This script loads a colored PLY point cloud, optionally rotates it, rasterizes
    it into sparse RGB depth bins using front-most-per-pixel selection, generates
    complex RGB holograms with shared random phase, and reconstructs RGB images
    at all depth-bin centers.

Pipeline:
    1. Load colored PLY point cloud
    2. Apply optional Euler rotation
    3. Normalize depth to a target physical range
    4. Rasterize point cloud into sparse RGB bins
    5. Generate complex RGB holograms using shared random phase
    6. Save hologram amplitude/phase
    7. Reconstruct and save all depth-bin images

Example:
    python ./model/PointCloud_ComplexHologram.py \
      --data_path ./dataset/PointCloud/bunnydragon_pointcloud_3e8.ply \
      --out_root ./result/PointCloud/4bins_3e8\
      --target_size 2048 \
      --out_size 2048 \
      --pixel_pitch 5e-6 \
      --wavelength 532e-9 \
      --z_min_mm 50.0 \
      --z_max_mm 53.0 \
      --num_bins 4 \
      --fit_margin 0.90 \
      --rng_seed 2024 \
      --gamma 0.65 \
      --vis_mode percentile \
      --p_low 1.0 \
      --p_high 99.0 \
      --rot_pitch_deg 90 \
      --rotate_around_center \
      --depth_flip

    python ./model/PointCloud_ComplexHologram.py \
      --data_path /workspace/yh/project/CGHReview/dataset/PointCloud3/PointCloud_3000000/bunnydragonRGB_pointcloud_color_3e7.ply\
      --out_root /workspace/yh/project/CGHReview/result_test/PCD\
      --target_size 2048 \
      --out_size 2048 \
      --pixel_pitch 5e-6 \
      --wavelength 532e-9 \
      --z_min_mm 50.0 \
      --z_max_mm 53.0 \
      --num_bins 4 \
      --fit_margin 0.90 \
      --rng_seed 2024 \
      --gamma 0.65 \
      --vis_mode percentile \
      --p_low 1.0 \
      --p_high 99.0 \
      --rot_pitch_deg 90 \
      --rotate_around_center \
      --depth_flip
"""

import os
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
        description="Generate complex RGB holograms from a colored point cloud and reconstruct all depth bins."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the input colored PLY point cloud."
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
        help="Rasterization image size (square)."
    )
    parser.add_argument(
        "--out_size",
        type=int,
        default=2048,
        help="Final saved reconstruction size (square)."
    )

    parser.add_argument(
        "--pixel_pitch",
        type=float,
        default=5e-6,
        help="Pixel pitch in meters."
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=532e-9,
        help="Optical wavelength in meters."
    )

    parser.add_argument(
        "--z_min_mm",
        type=float,
        default=50.0,
        help="Minimum target depth in mm."
    )
    parser.add_argument(
        "--z_max_mm",
        type=float,
        default=80.0,
        help="Maximum target depth in mm."
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=2,
        help="Number of depth bins."
    )

    parser.add_argument(
        "--fit_margin",
        type=float,
        default=0.90,
        help="Margin ratio used when fitting the point cloud into the raster image."
    )

    parser.add_argument(
        "--rng_seed",
        type=int,
        default=2024,
        help="Random seed for shared random phase."
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

    parser.add_argument(
        "--rot_yaw_deg",
        type=float,
        default=0.0,
        help="Yaw rotation angle in degrees."
    )
    parser.add_argument(
        "--rot_pitch_deg",
        type=float,
        default=90.0,
        help="Pitch rotation angle in degrees."
    )
    parser.add_argument(
        "--rot_roll_deg",
        type=float,
        default=0.0,
        help="Roll rotation angle in degrees."
    )
    parser.add_argument(
        "--rotate_around_center",
        action="store_true",
        help="Rotate around the point-cloud center instead of the origin."
    )

    parser.add_argument(
        "--depth_flip",
        action="store_true",
        help="Flip the depth direction after normalization."
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# ASM
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
# Utils
# -----------------------------------------------------------------------------
def ensure_dirs(out_root: str):
    dir_holo = os.path.join(out_root, "Hologram")
    dir_recon = os.path.join(out_root, "Recon")
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(dir_holo, exist_ok=True)
    os.makedirs(dir_recon, exist_ok=True)
    return dir_holo, dir_recon


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


def normalize_rgb_global_with_gamma(rgb_float: np.ndarray, gamma: float) -> np.ndarray:
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


def normalize_rgb_percentile_with_gamma(
    rgb_float: np.ndarray,
    gamma: float,
    p_low: float,
    p_high: float
) -> np.ndarray:
    """
    Percentile-based normalization followed by gamma correction.
    """
    x = np.asarray(rgb_float, dtype=np.float32)
    x = np.clip(x, 0.0, None)

    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    x = np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)

    if gamma is not None and abs(gamma - 1.0) > 1e-12:
        x = np.power(x, float(gamma))

    return (x * 255.0).clip(0, 255).astype(np.uint8)


def resize_keep_aspect_to_square_rgb(
    img_u8: np.ndarray,
    target: int = 2048,
    pad_value: int = 0
) -> np.ndarray:
    """
    Resize an RGB image to fit inside a square canvas with padding.
    """
    height, width = img_u8.shape[:2]
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
# Load colored PLY
# -----------------------------------------------------------------------------
def load_ply_xyzrgb(path: str):
    """
    Load a colored PLY point cloud using Open3D.

    Returns:
        pts: (N, 3) float32
        cols: (N, 3) float32 in [0, 1], or None if unavailable
    """
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    cols = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
    return pts, cols


# -----------------------------------------------------------------------------
# Rotation
# -----------------------------------------------------------------------------
def rotate_points_euler(
    pts: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    around_center: bool = True
) -> np.ndarray:
    """
    Rotate points using the user's original Euler convention:
        R = Rz(yaw) @ Rx(pitch) @ Ry(roll)
    """
    pts = pts.astype(np.float32)

    if around_center:
        center = pts.mean(axis=0, keepdims=True)
        pts = pts - center
    else:
        center = None

    yaw, pitch, roll = map(np.deg2rad, [yaw_deg, pitch_deg, roll_deg])

    rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw),  np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(pitch), -np.sin(pitch)],
        [0.0, np.sin(pitch),  np.cos(pitch)]
    ], dtype=np.float32)

    ry = np.array([
        [ np.cos(roll), 0.0, np.sin(roll)],
        [0.0, 1.0, 0.0],
        [-np.sin(roll), 0.0, np.cos(roll)]
    ], dtype=np.float32)

    pts = pts @ (rz @ rx @ ry).T

    if around_center:
        pts = pts + center

    return pts


# -----------------------------------------------------------------------------
# Point cloud -> RGB bins (front-most per pixel per bin)
# -----------------------------------------------------------------------------
def pointcloud_to_bins_rgb_frontmost(
    pts: np.ndarray,
    cols: np.ndarray,
    target_size: int,
    z_min_mm: float,
    z_max_mm: float,
    num_bins: int,
    fit_margin: float,
    depth_flip: bool
):
    """
    Rasterize a colored point cloud into sparse RGB bins.

    For each depth bin and pixel, only the front-most point is kept.
    """
    height = target_size
    width = target_size

    if cols is None:
        raise RuntimeError(
            "This pipeline expects a colored PLY (pcd.has_colors() == True), but the input point cloud has no colors."
        )

    cols = np.clip(cols.astype(np.float32), 0.0, 1.0)

    z = pts[:, 2].astype(np.float32)
    z01 = (z - float(z.min())) / (float(z.max()) - float(z.min()) + 1e-12)
    z_mm = z01 * (z_max_mm - z_min_mm) + z_min_mm

    if depth_flip:
        z_mm = (z_min_mm + z_max_mm) - z_mm

    x = pts[:, 0].astype(np.float32).copy()
    y = pts[:, 1].astype(np.float32).copy()

    x -= (float(x.min()) + float(x.max())) / 2.0
    y -= (float(y.min()) + float(y.max())) / 2.0

    scale = fit_margin * min(
        (width - 1) / (float(np.ptp(x)) + 1e-12),
        (height - 1) / (float(np.ptp(y)) + 1e-12)
    )

    px = x * scale + width / 2.0
    py = -y * scale + height / 2.0

    edges = np.linspace(z_min_mm, z_max_mm, num_bins + 1, dtype=np.float32)
    centers_mm = 0.5 * (edges[:-1] + edges[1:])

    rgb_bins = [np.zeros((height, width, 3), dtype=np.float32) for _ in range(num_bins)]
    mask_bins = [np.zeros((height, width), dtype=np.uint8) for _ in range(num_bins)]
    depth_bins = [np.full((height, width), np.inf, dtype=np.float32) for _ in range(num_bins)]

    cx = px.astype(np.int32)
    cy = py.astype(np.int32)
    inside = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height)

    cx = cx[inside]
    cy = cy[inside]
    zmm = z_mm[inside]
    color = cols[inside]

    bin_ids = np.digitize(zmm, edges) - 1
    bin_ids = np.clip(bin_ids, 0, num_bins - 1)

    print("[INFO] Rasterizing point cloud with front-most selection per pixel per bin...")
    for bin_idx in range(num_bins):
        sel = (bin_ids == bin_idx)
        if not np.any(sel):
            continue

        xk = cx[sel]
        yk = cy[sel]
        zk = zmm[sel]
        ck = color[sel]

        for i in tqdm(range(len(zk)), desc=f"bin{bin_idx} raster", leave=False, dynamic_ncols=True):
            x0 = xk[i]
            y0 = yk[i]
            z0 = zk[i]

            if z0 < depth_bins[bin_idx][y0, x0]:
                depth_bins[bin_idx][y0, x0] = z0
                rgb_bins[bin_idx][y0, x0, :] = ck[i]
                mask_bins[bin_idx][y0, x0] = 255

    return rgb_bins, mask_bins, centers_mm


# -----------------------------------------------------------------------------
# Hologram generation
# -----------------------------------------------------------------------------
def hologram_complex_rgb_random(
    rgb_bins,
    mask_bins,
    centers_mm: np.ndarray,
    target_size: int,
    rng_seed: int,
    wavelength: float,
    pixel_pitch: float
):
    """
    Generate complex RGB holograms using shared random phase per bin.
    """
    rng = np.random.default_rng(rng_seed)
    height = target_size
    width = target_size

    holograms = [
        np.zeros((height, width), dtype=np.complex64),
        np.zeros((height, width), dtype=np.complex64),
        np.zeros((height, width), dtype=np.complex64),
    ]

    for bin_idx, zmm in enumerate(tqdm(centers_mm, desc="Generating hologram (complex RGB, shared phase)")):
        mask = (mask_bins[bin_idx] > 0)
        if not np.any(mask):
            continue

        phase = rng.uniform(-np.pi, np.pi, size=(height, width)).astype(np.float32)
        phasor = np.exp(1j * phase).astype(np.complex64)

        rgb = np.clip(rgb_bins[bin_idx], 0.0, 1.0).astype(np.float32)
        dist = -float(zmm) * 1e-3  # object at z>0 to hologram at z=0

        for channel in range(3):
            u_obj = np.zeros((height, width), dtype=np.complex64)
            u_obj[mask] = rgb[..., channel][mask].astype(np.complex64) * phasor[mask]
            holograms[channel] += asm_propagate(u_obj, dist, wavelength, pixel_pitch)

    return holograms[0], holograms[1], holograms[2]


# -----------------------------------------------------------------------------
# Reconstruction
# -----------------------------------------------------------------------------
def reconstruct_bins_rgb(
    holo_r: np.ndarray,
    holo_g: np.ndarray,
    holo_b: np.ndarray,
    centers_mm: np.ndarray,
    out_dir_recon: str,
    wavelength: float,
    pixel_pitch: float,
    use_intensity: bool,
    gamma: float,
    out_size: int,
    vis_mode: str,
    p_low: float,
    p_high: float,
):
    """
    Reconstruct RGB images at all depth-bin centers.
    """
    out_dir = os.path.join(out_dir_recon, "bins")
    os.makedirs(out_dir, exist_ok=True)

    for bin_idx, zmm in enumerate(tqdm(centers_mm, desc="Reconstructing depth bins (RGB complex)")):
        dist = float(zmm) * 1e-3

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
                p_high=p_high
            )
        else:
            rgb_u8 = normalize_rgb_global_with_gamma(rgb, gamma=gamma)

        rgb_u8 = resize_keep_aspect_to_square_rgb(rgb_u8, target=out_size, pad_value=0)

        out_path = os.path.join(out_dir, f"bin{bin_idx:02d}_z{zmm:05.1f}mm.png")
        cv2.imwrite(out_path, cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.target_size <= 0:
        raise ValueError("--target_size must be positive.")
    if args.out_size <= 0:
        raise ValueError("--out_size must be positive.")
    if args.pixel_pitch <= 0:
        raise ValueError("--pixel_pitch must be positive.")
    if args.wavelength <= 0:
        raise ValueError("--wavelength must be positive.")
    if args.z_max_mm <= args.z_min_mm:
        raise ValueError("--z_max_mm must be greater than --z_min_mm.")
    if args.num_bins <= 0:
        raise ValueError("--num_bins must be positive.")
    if not (0.0 <= args.p_low < args.p_high <= 100.0):
        raise ValueError("--p_low and --p_high must satisfy 0 <= p_low < p_high <= 100.")

    start_time = time.time()

    dir_holo, dir_recon = ensure_dirs(args.out_root)

    pts, cols = load_ply_xyzrgb(args.data_path)
    print(f"[INFO] PLY points shape: {pts.shape}")
    print(f"[INFO] Has colors      : {cols is not None}")
    if cols is not None:
        print(
            f"[INFO] Color stats     : min={float(cols.min()):.6f}, "
            f"max={float(cols.max()):.6f}, mean={cols.mean(axis=0)}"
        )

    pts = rotate_points_euler(
        pts=pts,
        yaw_deg=args.rot_yaw_deg,
        pitch_deg=args.rot_pitch_deg,
        roll_deg=args.rot_roll_deg,
        around_center=args.rotate_around_center
    )

    rgb_bins, mask_bins, centers_mm = pointcloud_to_bins_rgb_frontmost(
        pts=pts,
        cols=cols,
        target_size=args.target_size,
        z_min_mm=args.z_min_mm,
        z_max_mm=args.z_max_mm,
        num_bins=args.num_bins,
        fit_margin=args.fit_margin,
        depth_flip=args.depth_flip,
    )

    holo_r, holo_g, holo_b = hologram_complex_rgb_random(
        rgb_bins=rgb_bins,
        mask_bins=mask_bins,
        centers_mm=centers_mm,
        target_size=args.target_size,
        rng_seed=args.rng_seed,
        wavelength=args.wavelength,
        pixel_pitch=args.pixel_pitch,
    )

    np.save(os.path.join(dir_holo, "holo_R.npy"), holo_r.astype(np.complex64))
    np.save(os.path.join(dir_holo, "holo_G.npy"), holo_g.astype(np.complex64))
    np.save(os.path.join(dir_holo, "holo_B.npy"), holo_b.astype(np.complex64))

    amp_r = np.abs(holo_r).astype(np.float32)
    amp_g = np.abs(holo_g).astype(np.float32)
    amp_b = np.abs(holo_b).astype(np.float32)

    pha_r = np.angle(holo_r).astype(np.float32)
    pha_g = np.angle(holo_g).astype(np.float32)
    pha_b = np.angle(holo_b).astype(np.float32)

    np.save(os.path.join(dir_holo, "amp_R.npy"), amp_r)
    np.save(os.path.join(dir_holo, "amp_G.npy"), amp_g)
    np.save(os.path.join(dir_holo, "amp_B.npy"), amp_b)

    np.save(os.path.join(dir_holo, "phase_R.npy"), pha_r)
    np.save(os.path.join(dir_holo, "phase_G.npy"), pha_g)
    np.save(os.path.join(dir_holo, "phase_B.npy"), pha_b)

    save_float_image(amp_r, os.path.join(dir_holo, "amp_R.png"))
    save_float_image(amp_g, os.path.join(dir_holo, "amp_G.png"))
    save_float_image(amp_b, os.path.join(dir_holo, "amp_B.png"))

    save_phase_image(pha_r, os.path.join(dir_holo, "phase_R.png"))
    save_phase_image(pha_g, os.path.join(dir_holo, "phase_G.png"))
    save_phase_image(pha_b, os.path.join(dir_holo, "phase_B.png"))

    hologram_end_time = time.time()
    print(f"[INFO] Hologram generation time: {hologram_end_time - start_time:.2f} seconds")

    reconstruct_bins_rgb(
        holo_r=holo_r,
        holo_g=holo_g,
        holo_b=holo_b,
        centers_mm=centers_mm,
        out_dir_recon=dir_recon,
        wavelength=args.wavelength,
        pixel_pitch=args.pixel_pitch,
        use_intensity=args.use_intensity,
        gamma=args.gamma,
        out_size=args.out_size,
        vis_mode=args.vis_mode,
        p_low=args.p_low,
        p_high=args.p_high,
    )

    end_time = time.time()

    print("[DONE]")
    print(f"Hologram output : {dir_holo}")
    print(f"Recon output    : {os.path.join(dir_recon, 'bins')}")
    print(
        f"[INFO] Display settings: vis_mode={args.vis_mode}, "
        f"p_low={args.p_low}, p_high={args.p_high}, "
        f"use_intensity={args.use_intensity}, gamma={args.gamma}"
    )
    print(f"[INFO] Total runtime: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()