#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Complex RGB Holograms from a Perspective Light Field and Reconstruct at Multiple Depths

Author: Hao Yun
Date: 2026-03-25

Description:
    This script loads a perspective RGB light field together with poses.csv,
    refocuses all views to a reference depth D0 by subpixel shifting, converts
    the centered light field into tiled complex fields, propagates them to the
    hologram plane using ASM, saves complex/amplitude/phase holograms, and
    reconstructs RGB images at multiple axial distances.

Pipeline:
    1. Load perspective LF images and poses.csv
    2. Refocus all views to the reference depth D0
    3. Build centered LF in memory
    4. Convert each RGB channel into a tiled complex field
    5. Propagate RS plane -> hologram plane using ASM
    6. Save hologram complex / amplitude / phase
    7. Reconstruct RGB images at multiple z positions

Example:
    python ./model/LightField_Pers_ComplexHologram.py \
      --lf_root /workspace/yh/project/CGHReview/dataset/LightField3/LF_pers_40x40_rgb_fix_800_50_53mm_parallel_3 \
      --output_dir /workspace/yh/project/CGHReview/result_test\
      --fov_deg 7 \
      --d0 -0.050 \
      --pose_unit mm \
      --wavelength 532e-9 \
      --pitch 2e-6 \
      --dist_rs_to_h 0.05 \
      --gamma 0.65 \
      --out_size 2048 \
      --vis_mode percentile \
      --save_holo_png \
      --save_recon_png \
      --p_low 1.0 \
      --p_high 99.0 \
      --z_min -0.049 \
      --z_max -0.048 \
      --z_step 0.001

Hint: The value of --fov_deg must be the same as that in Obj2LF_pers.py
"""

import os
import csv
import glob
import time
import argparse
import numpy as np
import imageio.v2 as imageio
import cv2
from tqdm import tqdm


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate complex RGB holograms from a perspective light field and reconstruct them at multiple depths."
    )

    parser.add_argument(
        "--lf_root",
        type=str,
        required=True,
        help="Root directory of the perspective LF dataset."
    )
    parser.add_argument(
        "--images_subdir",
        type=str,
        default="images",
        help="Subdirectory containing LF images."
    )
    parser.add_argument(
        "--poses_csv",
        type=str,
        default="poses.csv",
        help="CSV file containing pose metadata."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory."
    )

    parser.add_argument(
        "--fov_deg",
        type=float,
        default=10.0,
        help="Field of view used when generating the LF."
    )
    parser.add_argument(
        "--d0",
        type=float,
        default=-0.050,
        help="Reference refocus depth in meters."
    )
    parser.add_argument(
        "--pose_unit",
        type=str,
        default="mm",
        choices=["mm", "m"],
        help="Unit used in poses.csv for dx and dz."
    )

    parser.add_argument(
        "--wavelength",
        type=float,
        default=532e-9,
        help="Optical wavelength in meters."
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=2e-6,
        help="Pixel pitch in meters."
    )
    parser.add_argument(
        "--dist_rs_to_h",
        type=float,
        default=0.05,
        help="Propagation distance from RS plane to hologram plane in meters."
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.65,
        help="Gamma used for reconstruction visualization."
    )
    parser.add_argument(
        "--out_size",
        type=int,
        default=2048,
        help="Output reconstruction image size (square)."
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
        help="Lower percentile used when vis_mode=percentile."
    )
    parser.add_argument(
        "--p_high",
        type=float,
        default=99.0,
        help="Upper percentile used when vis_mode=percentile."
    )

    parser.add_argument(
        "--z_list",
        type=float,
        nargs="*",
        default=None,
        help="Explicit reconstruction z list in meters. If provided, overrides z_min/z_max/z_step."
    )
    parser.add_argument(
        "--z_min",
        type=float,
        default=-0.060,
        help="Minimum reconstruction distance in meters."
    )
    parser.add_argument(
        "--z_max",
        type=float,
        default=-0.040,
        help="Maximum reconstruction distance in meters."
    )
    parser.add_argument(
        "--z_step",
        type=float,
        default=0.001,
        help="Reconstruction z step in meters."
    )

    parser.add_argument(
        "--save_holo_png",
        action="store_true",
        help="Save hologram amplitude/phase PNGs for visualization."
    )
    parser.add_argument(
        "--save_recon_png",
        action="store_true",
        help="Save reconstructed RGB PNGs."
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=2026,
        help="Random seed used in LF-to-complex conversion."
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def myfft2(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))


def asm_propagate(u_in: np.ndarray, dist_m: float, wavelength_m: float, pitch_m: float) -> np.ndarray:
    u_in = np.asarray(u_in, np.complex64)
    height, width = u_in.shape

    dfx = 1.0 / (height * pitch_m)
    dfy = 1.0 / (width * pitch_m)
    fx = (np.arange(height) - height / 2) * dfx
    fy = (np.arange(width) - width / 2) * dfy
    fx_grid, fy_grid = np.meshgrid(fx, fy, indexing="ij")

    k = 2.0 * np.pi / wavelength_m
    term = 1.0 - (wavelength_m * fx_grid) ** 2 - (wavelength_m * fy_grid) ** 2

    transfer = np.zeros_like(term, np.complex64)
    mask = term >= 0
    transfer[mask] = np.exp(1j * k * dist_m * np.sqrt(term[mask]).astype(np.float32))

    u_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_in)))
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(u_fft * transfer)))


def compute_fx_fy_from_fov(width: int, height: int, fov_deg: float):
    fov = np.deg2rad(fov_deg)
    fx = (width * 0.5) / np.tan(fov * 0.5)
    fy = (height * 0.5) / np.tan(fov * 0.5)
    return float(fx), float(fy)


def shift_image_subpixel(img_u8: np.ndarray, sx: float, sy: float) -> np.ndarray:
    height, width = img_u8.shape[:2]
    mat = np.array([[1, 0, sx],
                    [0, 1, sy]], np.float32)
    return cv2.warpAffine(
        img_u8,
        mat,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )


def save_float_vis(x: np.ndarray, path: str, p_low: float, p_high: float):
    x = np.asarray(x, np.float32)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    x = np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)
    imageio.imwrite(path, (x * 255).astype(np.uint8))


def save_phase_vis(phase: np.ndarray, path: str):
    phase = np.asarray(phase, np.float32)
    image = ((phase + np.pi) / (2 * np.pi) * 255.0).clip(0, 255).astype(np.uint8)
    imageio.imwrite(path, image)


def normalize_rgb_percentile_with_gamma(
    rgb: np.ndarray,
    gamma: float,
    p_low: float,
    p_high: float
) -> np.ndarray:
    x = np.asarray(rgb, np.float32)
    x = np.clip(x, 0, None)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    x = np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)

    if gamma is not None and abs(gamma - 1.0) > 1e-12:
        x = np.power(x, gamma)

    return (x * 255).clip(0, 255).astype(np.uint8)


def normalize_rgb_minmax_with_gamma(rgb: np.ndarray, gamma: float) -> np.ndarray:
    x = np.asarray(rgb, np.float32)
    x = np.clip(x, 0, None)
    x = x - x.min()
    max_val = float(x.max())
    if max_val > 1e-8:
        x = x / max_val

    if gamma is not None and abs(gamma - 1.0) > 1e-12:
        x = np.power(np.clip(x, 0, 1), gamma)

    return (x * 255).clip(0, 255).astype(np.uint8)


def resize_keep_aspect_to_square(img_u8: np.ndarray, target: int, pad_value: int = 0) -> np.ndarray:
    height, width = img_u8.shape[:2]
    scale = float(target) / float(max(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(img_u8, (new_width, new_height), interpolation=cv2.INTER_AREA)
    output = np.full((target, target, 3), pad_value, dtype=np.uint8)

    y0 = (target - new_height) // 2
    x0 = (target - new_width) // 2
    output[y0:y0 + new_height, x0:x0 + new_width] = resized
    return output


def build_z_list(args):
    if args.z_list is not None and len(args.z_list) > 0:
        return list(args.z_list)

    if abs(args.z_step) < 1e-12:
        raise ValueError("--z_step cannot be 0.")

    step = args.z_step
    if (args.z_max - args.z_min) * step < 0:
        step = -step

    n = int(np.floor((args.z_max - args.z_min) / step + 1e-12)) + 1
    if n <= 0:
        return []

    return [args.z_min + i * step for i in range(n)]


# -----------------------------------------------------------------------------
# Step 1: load LF + poses
# -----------------------------------------------------------------------------
def load_lf_and_poses(lf_root: str, images_subdir: str, poses_csv: str):
    img_dir = os.path.join(lf_root, images_subdir)
    poses_path = os.path.join(lf_root, poses_csv)

    if not os.path.isdir(img_dir):
        raise RuntimeError(f"Missing images directory: {img_dir}")
    if not os.path.isfile(poses_path):
        raise RuntimeError(f"Missing poses CSV: {poses_path}")

    rows = []
    with open(poses_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "u": int(row["u"]),
                "v": int(row["v"]),
                "file": row["file"],
                "dx": float(row["dx"]),
                "dz": float(row["dz"]),
            })

    if len(rows) == 0:
        raise RuntimeError("poses.csv is empty.")

    u_max = max(r["u"] for r in rows)
    v_max = max(r["v"] for r in rows)
    nu, nv = u_max + 1, v_max + 1

    sample_path = os.path.join(img_dir, rows[0]["file"])
    if not os.path.isfile(sample_path):
        raise RuntimeError(f"Sample image missing: {sample_path}")

    sample = imageio.imread(sample_path)
    if sample.ndim != 3 or sample.shape[2] < 3:
        raise RuntimeError("RGB images are required.")

    height, width = sample.shape[:2]
    pose_map = {(r["u"], r["v"]): r for r in rows}

    return img_dir, pose_map, nu, nv, height, width


# -----------------------------------------------------------------------------
# Step 2: build centered LF in memory
# -----------------------------------------------------------------------------
def build_centered_lf(
    img_dir: str,
    pose_map: dict,
    nu: int,
    nv: int,
    height: int,
    width: int,
    fov_deg: float,
    d0: float,
    pose_unit: str
) -> np.ndarray:
    fx, fy = compute_fx_fy_from_fov(width, height, fov_deg)
    unit_scale = 1e-3 if pose_unit == "mm" else 1.0

    u0, v0 = nu // 2, nv // 2
    if (u0, v0) not in pose_map:
        u0, v0 = sorted(pose_map.keys())[len(pose_map) // 2]

    dx0 = pose_map[(u0, v0)]["dx"] * unit_scale
    dz0 = pose_map[(u0, v0)]["dz"] * unit_scale

    if abs(d0) < 1e-12:
        raise RuntimeError("D0 must be non-zero.")

    lf_centered = np.zeros((nv, nu, height, width, 3), np.float32)

    for v in tqdm(range(nv), desc="Center LF (in-memory)", dynamic_ncols=True):
        for u in range(nu):
            record = pose_map.get((u, v), None)
            if record is None:
                raise RuntimeError(f"Missing pose for (u, v)=({u}, {v})")

            in_path = os.path.join(img_dir, record["file"])
            if not os.path.isfile(in_path):
                raise RuntimeError(f"Missing image: {in_path}")

            img = imageio.imread(in_path)[..., :3].astype(np.uint8)
            if img.shape[0] != height or img.shape[1] != width:
                raise RuntimeError(
                    f"Image size mismatch: {record['file']} got {img.shape[:2]}, expected {(height, width)}"
                )

            dx = record["dx"] * unit_scale - dx0
            dz = record["dz"] * unit_scale - dz0

            sx = -fx * (dx / d0)
            sy = -fy * (dz / d0)

            out_img = shift_image_subpixel(img, sx, sy).astype(np.float32) / 255.0
            lf_centered[v, u] = out_img

    return lf_centered


# -----------------------------------------------------------------------------
# Step 3: LF -> ComplexField
# -----------------------------------------------------------------------------
def convert_lightfield_to_complex_field(lf_4d: np.ndarray, seed: int) -> np.ndarray:
    """
    lf_4d: [Nv, Nu, Nt, Ns]
    return: [Nv*Nt, Nu*Ns]
    """
    rng = np.random.default_rng(seed)
    nv, nu, nt, ns = lf_4d.shape
    out = np.zeros((nv * nt, nu * ns), np.complex64)

    for s in tqdm(range(ns), desc="LF -> ComplexField", dynamic_ncols=True):
        for t in range(nt):
            patch = lf_4d[:, :, t, s].astype(np.float32)
            phase = rng.random((nv, nu), dtype=np.float32) * (2 * np.pi)
            patch_c = patch.astype(np.complex64) * np.exp(1j * phase).astype(np.complex64)
            out[t * nv:(t + 1) * nv, s * nu:(s + 1) * nu] = myfft2(patch_c).astype(np.complex64)

    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.wavelength <= 0:
        raise ValueError("--wavelength must be positive.")
    if args.pitch <= 0:
        raise ValueError("--pitch must be positive.")
    if args.out_size <= 0:
        raise ValueError("--out_size must be positive.")
    if not (0.0 <= args.p_low < args.p_high <= 100.0):
        raise ValueError("--p_low and --p_high must satisfy 0 <= p_low < p_high <= 100.")

    begin = time.time()

    ensure_dir(args.output_dir)
    holo_dir = os.path.join(args.output_dir, "Hologram")
    recon_dir = os.path.join(args.output_dir, "Recon")
    ensure_dir(holo_dir)
    ensure_dir(recon_dir)

    img_dir, pose_map, nu, nv, height, width = load_lf_and_poses(
        lf_root=args.lf_root,
        images_subdir=args.images_subdir,
        poses_csv=args.poses_csv
    )

    print("[INFO] LF root:", args.lf_root)
    print("[INFO] Images directory:", img_dir)
    print("[INFO] NU, NV:", nu, nv, "Image size:", (height, width))
    print("[INFO] FOV_DEG:", args.fov_deg, "D0:", args.d0, "POSE_UNIT:", args.pose_unit)
    print("[INFO] Wavelength:", args.wavelength, "Pitch:", args.pitch, "DIST_RS2H:", args.dist_rs_to_h)

    lf_centered = build_centered_lf(
        img_dir=img_dir,
        pose_map=pose_map,
        nu=nu,
        nv=nv,
        height=height,
        width=width,
        fov_deg=args.fov_deg,
        d0=args.d0,
        pose_unit=args.pose_unit
    )
    print("[INFO] Centered LF built:", lf_centered.shape)

    holograms = []
    for c, name in enumerate(["R", "G", "B"]):
        print(f"[INFO] Build complex field for channel {name} ...")
        complex_field = convert_lightfield_to_complex_field(
            lf_centered[..., c],
            args.random_seed + 1000 * c
        )

        print(f"[INFO] ASM propagate RS -> H for channel {name} ...")
        holo = asm_propagate(complex_field, args.dist_rs_to_h, args.wavelength, args.pitch).astype(np.complex64)
        holograms.append(holo)

        np.save(os.path.join(holo_dir, f"hologram_complex_{name}.npy"), holo.astype(np.complex64))

        amp = np.abs(holo).astype(np.float32)
        phase = np.angle(holo).astype(np.float32)

        np.save(os.path.join(holo_dir, f"hologram_amp_{name}.npy"), amp)
        np.save(os.path.join(holo_dir, f"hologram_phase_{name}.npy"), phase)

        if args.save_holo_png:
            save_float_vis(amp, os.path.join(holo_dir, f"hologram_amp_{name}.png"), args.p_low, args.p_high)
            save_phase_vis(phase, os.path.join(holo_dir, f"hologram_phase_{name}.png"))

    holo_end = time.time()
    print(f"[INFO] Hologram generation time: {holo_end - begin:.2f} seconds")

    z_list = build_z_list(args)
    if len(z_list) == 0:
        raise RuntimeError("Empty z_list. Check --z_list or --z_min/--z_max/--z_step.")

    print("[INFO] Reconstruction z count:", len(z_list), "range:", (z_list[0], z_list[-1]))

    for z in tqdm(z_list, desc="Recon (ASM RGB)", dynamic_ncols=True):
        rec = []
        for c in range(3):
            u = asm_propagate(holograms[c], z, args.wavelength, args.pitch)
            rec.append(np.abs(u).astype(np.float32))

        rgb = np.stack(rec, axis=-1)

        if args.vis_mode == "percentile":
            rgb_u8 = normalize_rgb_percentile_with_gamma(
                rgb,
                args.gamma,
                args.p_low,
                args.p_high
            )
        else:
            rgb_u8 = normalize_rgb_minmax_with_gamma(rgb, args.gamma)

        rgb_u8 = resize_keep_aspect_to_square(rgb_u8, args.out_size, pad_value=0)

        if args.save_recon_png:
            imageio.imwrite(os.path.join(recon_dir, f"recon_z_{z:+.6f}.png"), rgb_u8)

    print("DONE:", args.output_dir)
    print("[INFO] Saved:")
    print("  Hologram/hologram_complex_{R,G,B}.npy")
    print("  Hologram/hologram_amp_{R,G,B}.npy")
    print("  Hologram/hologram_phase_{R,G,B}.npy")
    if args.save_holo_png:
        print("  Hologram/hologram_amp_{R,G,B}.png")
        print("  Hologram/hologram_phase_{R,G,B}.png")
    if args.save_recon_png:
        print("  Recon/recon_z_*.png")

    end = time.time()
    print(f"[INFO] Total runtime: {end - begin:.2f} seconds")


if __name__ == "__main__":
    main()