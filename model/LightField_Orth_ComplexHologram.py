#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Complex RGB Holograms from an Orthographic Light Field and Reconstruct at Multiple Depths

Author: Hao Yun
Date: 2026-03-25

Description:
    This script loads an orthographic RGB light field, converts each RGB channel
    into a tiled complex field using a MATLAB-style FFT-based patch transform,
    propagates the fields to the hologram plane using the Angular Spectrum Method (ASM),
    saves complex/amplitude/phase holograms, and reconstructs RGB images at
    multiple axial distances.

Pipeline:
    1. Load orthographic LF images into LF_rgb [Nv, Nu, Nt, Ns, 3]
    2. Convert each channel to a complex field
    3. Propagate RS plane -> hologram plane using ASM
    4. Save:
         - complex hologram (.npy)
         - amplitude hologram (.npy / .png)
         - phase hologram (.npy / .png)
    5. Reconstruct RGB images at all specified z positions

Example:
    python ./model/LightField_Orth_ComplexHologram.py \
      --input_dir ./dataset/LightField/LF_orth_20x20_400/images \
      --output_dir ./result/LightField/orth_20_400 \
      --nu 20 --nv 20 --ns 400 --nt 400 \
      --dx 2e-6 --dy 2e-6 \
      --wavelength 532e-9 \
      --distance_rs_to_h 50e-3 \
      --random_seed 2026 \
      --vis_mode percentile \
      --p_low 1.0 \
      --p_high 99.0 \
      --gamma 1.0 \
      --z_list  -0.0510 -0.0515 -0.053 -0.054

"""

import os
import glob
import time
import argparse
import numpy as np
import imageio.v2 as imageio
import cv2

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate complex RGB holograms from an orthographic light field and reconstruct them at multiple depths."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing orthographic light field view images."
    )
    parser.add_argument(
        "--file_ext",
        type=str,
        default="png",
        help="Image file extension."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory."
    )

    parser.add_argument(
        "--nu",
        type=int,
        required=True,
        help="Number of angular samples in u."
    )
    parser.add_argument(
        "--nv",
        type=int,
        required=True,
        help="Number of angular samples in v."
    )
    parser.add_argument(
        "--ns",
        type=int,
        required=True,
        help="Width of each LF view image."
    )
    parser.add_argument(
        "--nt",
        type=int,
        required=True,
        help="Height of each LF view image."
    )

    parser.add_argument(
        "--dx",
        type=float,
        default=2e-6,
        help="Sampling pitch along x in meters."
    )
    parser.add_argument(
        "--dy",
        type=float,
        default=2e-6,
        help="Sampling pitch along y in meters."
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=532e-9,
        help="Optical wavelength in meters."
    )
    parser.add_argument(
        "--distance_rs_to_h",
        type=float,
        default=50e-3,
        help="Propagation distance from RS plane to hologram plane in meters."
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=2026,
        help="Random seed used for phase generation."
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma correction for reconstructed RGB visualization."
    )
    parser.add_argument(
        "--vis_mode",
        type=str,
        default="percentile",
        choices=["percentile", "minmax"],
        help='Visualization mode for reconstruction brightness: "percentile" or "minmax".'
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
        nargs="+",
        required=True,
        help="List of reconstruction distances in meters."
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# FFT wrappers (match MATLAB style)
# -----------------------------------------------------------------------------
def myfft2(x: np.ndarray) -> np.ndarray:
    """
    MATLAB-style fftshift(fft2(ifftshift(x))).
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))


def myifft2(x: np.ndarray) -> np.ndarray:
    """
    MATLAB-style fftshift(ifft2(ifftshift(X))).
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))


# -----------------------------------------------------------------------------
# Save helpers
# -----------------------------------------------------------------------------
def save_float_image(x: np.ndarray, path: str):
    """
    Save a float array as an 8-bit grayscale image after min-max normalization.
    This is for visualization only.
    """
    x = np.asarray(x, dtype=np.float32)
    x = x - x.min()
    max_val = float(x.max())
    if max_val > 1e-8:
        x = x / max_val
    image = (x * 255.0).clip(0, 255).astype(np.uint8)
    imageio.imwrite(path, image)


def save_phase_image(phase: np.ndarray, path: str):
    """
    Map phase in [-pi, pi] to [0, 255] and save as an 8-bit grayscale image.
    """
    phase = np.asarray(phase, dtype=np.float32)
    image = ((phase + np.pi) / (2 * np.pi) * 255.0).clip(0, 255).astype(np.uint8)
    imageio.imwrite(path, image)


def normalize_rgb_minmax_with_gamma(rgb_float: np.ndarray, gamma: float = 1.0) -> np.ndarray:
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
    gamma: float = 1.0,
    p_low: float = 1.0,
    p_high: float = 99.0
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


# -----------------------------------------------------------------------------
# Load LF
# -----------------------------------------------------------------------------
def load_orthographic_lf_rgb(
    input_dir: str,
    file_ext: str,
    nu: int,
    nv: int,
    nt: int,
    ns: int
) -> np.ndarray:
    """
    Load orthographic RGB LF images into LF_rgb with shape [Nv, Nu, Nt, Ns, 3].

    The indexing logic intentionally follows the user's original implementation.
    """
    files = sorted(glob.glob(os.path.join(input_dir, f"*.{file_ext}")))
    if len(files) < nu * nv:
        raise RuntimeError(f"Not enough images: need {nu * nv}, got {len(files)} in {input_dir}")

    temp = np.zeros((nt, ns, 3, nv, nu), dtype=np.float32)
    kmax = nu * nv

    iterator = ((idx_u, idx_v) for idx_u in range(1, nu + 1) for idx_v in range(1, nv + 1))
    if HAS_TQDM:
        iterator = tqdm(iterator, total=nu * nv, desc="Load LF (RGB)", dynamic_ncols=True)

    for idx_u, idx_v in iterator:
        k = (idx_v - 1) * nv + idx_u
        if k < 1 or k > kmax:
            raise RuntimeError(f"k out of range: k={k}, idx_u={idx_u}, idx_v={idx_v}")

        img = imageio.imread(files[k - 1]).astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        if img.ndim != 3 or img.shape[2] < 3:
            raise RuntimeError(f"RGB mode expects a color image: {files[k - 1]}")

        img = img[..., :3]

        if img.shape[0] != nt or img.shape[1] != ns:
            raise RuntimeError(
                f"Image shape mismatch: got {img.shape}, expected {(nt, ns, 3)}: {files[k - 1]}"
            )

        temp[:, :, :, idx_v - 1, idx_u - 1] = img

    lf_rgb = np.transpose(temp, (3, 4, 0, 1, 2)).copy()
    return lf_rgb


# -----------------------------------------------------------------------------
# LF -> complex field (per channel)
# -----------------------------------------------------------------------------
def convert_lightfield_to_complex_field(lf: np.ndarray, seed: int = 2026) -> np.ndarray:
    """
    Convert a 4D light field [Nv, Nu, Nt, Ns] into a tiled complex field.

    The patch arrangement follows the user's original logic exactly.
    """
    rng = np.random.default_rng(seed)

    if lf.ndim != 4:
        raise RuntimeError(f"LF must be 4D [Nv, Nu, Nt, Ns], got shape {lf.shape}")

    nv, nu, nt, ns = lf.shape
    complex_field = np.zeros((nv * nt, nu * ns), dtype=np.complex64)

    total = nt * ns
    if HAS_TQDM:
        pbar = tqdm(total=total, desc="LF -> ComplexField", dynamic_ncols=True)
    else:
        pbar = None

    for idx_s in range(1, ns + 1):
        for idx_t in range(1, nt + 1):
            u0 = (idx_s - 1) * nu
            v0 = (idx_t - 1) * nv
            u_range = slice(u0, u0 + nu)
            v_range = slice(v0, v0 + nv)

            patch = lf[:, :, idx_t - 1, idx_s - 1].astype(np.float32)
            phase = rng.random((nv, nu), dtype=np.float32) * (2.0 * np.pi)
            patch_c = patch.astype(np.complex64) * np.exp(1j * phase).astype(np.complex64)

            complex_field[v_range, u_range] = myfft2(patch_c).astype(np.complex64)

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    return complex_field


# -----------------------------------------------------------------------------
# ASM
# -----------------------------------------------------------------------------
def asm_propagate(u_in: np.ndarray, dist_m: float, wavelength_m: float, pitch_m: float) -> np.ndarray:
    """
    Angular Spectrum Method propagation with evanescent-wave masking.
    """
    u_in = np.asarray(u_in, dtype=np.complex64)
    height, width = u_in.shape

    dfx = 1.0 / (height * pitch_m)
    dfy = 1.0 / (width * pitch_m)
    fx = (np.arange(height) - height / 2) * dfx
    fy = (np.arange(width) - width / 2) * dfy
    fx_grid, fy_grid = np.meshgrid(fx, fy, indexing="ij")

    k = 2.0 * np.pi / wavelength_m
    term = 1.0 - (wavelength_m * fx_grid) ** 2 - (wavelength_m * fy_grid) ** 2

    transfer = np.zeros_like(term, dtype=np.complex64)
    mask = term >= 0
    transfer[mask] = np.exp(1j * k * dist_m * np.sqrt(term[mask]).astype(np.float32))

    u_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_in)))
    u_z = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(u_fft * transfer)))
    return u_z


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    if abs(args.dx - args.dy) > 1e-15:
        raise RuntimeError(
            f"ASM here uses a single pitch; require dx == dy. Got dx={args.dx}, dy={args.dy}"
        )
    if args.nu <= 0 or args.nv <= 0 or args.ns <= 0 or args.nt <= 0:
        raise ValueError("nu, nv, ns, and nt must all be positive.")
    if args.wavelength <= 0:
        raise ValueError("--wavelength must be positive.")
    if not (0.0 <= args.p_low < args.p_high <= 100.0):
        raise ValueError("--p_low and --p_high must satisfy 0 <= p_low < p_high <= 100.")

    os.makedirs(args.output_dir, exist_ok=True)
    hologram_dir = os.path.join(args.output_dir, "Hologram")
    recon_dir = os.path.join(args.output_dir, "Recon")
    os.makedirs(hologram_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    begin = time.time()

    print("Loading orthographic RGB light field...")
    lf_rgb = load_orthographic_lf_rgb(
        input_dir=args.input_dir,
        file_ext=args.file_ext,
        nu=args.nu,
        nv=args.nv,
        nt=args.nt,
        ns=args.ns,
    )
    print(f"[INFO] LF_rgb shape: {lf_rgb.shape} (expected [Nv, Nu, Nt, Ns, 3])")

    print("Converting RGB channels to complex fields...")
    complex_fields = []
    channel_iter = range(3)
    if HAS_TQDM:
        channel_iter = tqdm(channel_iter, desc="RGB Channels", dynamic_ncols=True)

    for c in channel_iter:
        complex_fields.append(
            convert_lightfield_to_complex_field(
                lf_rgb[..., c],
                seed=args.random_seed + 1000 * c
            )
        )

    print("Propagating RS plane -> hologram plane (ASM, complex hologram)...")
    holograms = []
    for c in range(3):
        holo_c = asm_propagate(
            complex_fields[c],
            args.distance_rs_to_h,
            args.wavelength,
            args.dx
        )
        holograms.append(holo_c)

    for c, name in enumerate(["R", "G", "B"]):
        holo_c = holograms[c].astype(np.complex64)

        np.save(os.path.join(hologram_dir, f"hologram_complex_{name}.npy"), holo_c)

        amp_c = np.abs(holo_c).astype(np.float32)
        phs_c = np.angle(holo_c).astype(np.float32)
        np.save(os.path.join(hologram_dir, f"hologram_amp_{name}.npy"), amp_c)
        np.save(os.path.join(hologram_dir, f"hologram_phase_{name}.npy"), phs_c)

        save_float_image(amp_c, os.path.join(hologram_dir, f"hologram_amp_{name}.png"))
        save_phase_image(phs_c, os.path.join(hologram_dir, f"hologram_phase_{name}.png"))

    holo_end = time.time()
    print(f"[INFO] Hologram generation time: {holo_end - begin:.2f} seconds")

    print("Reconstructing RGB images from complex holograms...")
    z_iterator = args.z_list
    if HAS_TQDM:
        z_iterator = tqdm(args.z_list, desc="Reconstruction (z, ASM RGB)", dynamic_ncols=True)

    for z in z_iterator:
        rec_rgb = []
        for c in range(3):
            rec_c = asm_propagate(holograms[c], z, args.wavelength, args.dx)
            rec_rgb.append(np.abs(rec_c).astype(np.float32))

        rec_img = np.stack(rec_rgb, axis=-1)

        if args.vis_mode == "percentile":
            rec_u8 = normalize_rgb_percentile_with_gamma(
                rec_img,
                gamma=args.gamma,
                p_low=args.p_low,
                p_high=args.p_high
            )
        else:
            rec_u8 = normalize_rgb_minmax_with_gamma(
                rec_img,
                gamma=args.gamma
            )

        rec_u8 = cv2.resize(rec_u8, (2048, 2048), interpolation=cv2.INTER_AREA)

        out_path = os.path.join(recon_dir, f"recon_rgb_z_{z:+.6f}m.png")
        imageio.imwrite(out_path, rec_u8)

    end = time.time()

    print(f"Done -> {args.output_dir}")
    print("[INFO] Saved NPY files for exact reconstruction:")
    print("  hologram_complex_{R,G,B}.npy  (complex64)")
    print("  hologram_amp_{R,G,B}.npy      (float32, NOT normalized)")
    print("  hologram_phase_{R,G,B}.npy    (float32 radians)")
    print("[INFO] PNG files are for visualization only (amplitude PNG is normalized).")
    print(
        f"[INFO] Reconstruction display settings: "
        f"vis_mode={args.vis_mode}, p_low={args.p_low}, p_high={args.p_high}, gamma={args.gamma}"
    )
    print(f"[INFO] Total runtime: {end - begin:.2f} seconds")


if __name__ == "__main__":
    main()