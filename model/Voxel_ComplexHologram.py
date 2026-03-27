#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Complex RGB Holograms from Rendered Voxel Bins and Reconstruct RGB Slices

Author: Hao Yun
Date: 2026-03-25

Description:
    This script loads a voxel occupancy volume from an NPZ file, renders
    per-bin mesh subsets after optional rotation, converts rendered RGB bins
    into complex RGB holograms using a shared random phase, and reconstructs
    RGB slices at all bin-center depths.

Important note:
    This version is intended to preserve the numerical behavior of the original
    script as closely as possible. The default settings and core computation
    logic are intentionally kept unchanged.

Pipeline:
    1. Load voxel occupancy grid from NPZ
    2. Extract full-surface mesh with marching cubes
    3. Rotate the mesh vertices if needed
    4. Slice the rotated mesh into bins along a selected axis
    5. Render each bin to RGB + mask
    6. Generate complex RGB holograms using shared random phase
    7. Save complex holograms and amplitude/phase maps
    8. Reconstruct RGB slices at all bin-center depths

Example:
    python ./model/Voxel_ComplexHologram.py \
      --npz_path /workspace/yh/project/CGHReview/dataset/Voxel3/BunnyDragon_voxel.npz \
      --out_root ./result/Voxel/100bins\
      --num_bins 100 
      
"""

import os
import time
import argparse
import numpy as np
import cv2
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate complex RGB holograms from rendered voxel bins and reconstruct RGB slices."
    )

    parser.add_argument(
        "--npz_path",
        type=str,
        required=True,
        help="Path to the voxel NPZ file."
    )
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Output root directory."
    )

    # Image size
    parser.add_argument(
        "--target_height",
        type=int,
        default=2048,
        help="Rendered image height."
    )
    parser.add_argument(
        "--target_width",
        type=int,
        default=2048,
        help="Rendered image width."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=256,
        help="Matplotlib rendering DPI."
    )

    # Fixed depth mapping
    parser.add_argument(
        "--z_min_mm",
        type=float,
        default=50.0,
        help="Minimum mapped depth in mm."
    )
    parser.add_argument(
        "--z_max_mm",
        type=float,
        default=53.0,
        help="Maximum mapped depth in mm."
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=100,
        help="Number of depth bins."
    )

    # Bin slicing axis after rotation
    parser.add_argument(
        "--bin_axis_rotated",
        type=str,
        default="y",
        choices=["x", "y", "z"],
        help="Axis used for binning after rotation."
    )
    parser.add_argument(
        "--bin_reverse",
        type=str2bool,
        default=False,
        help="Whether to reverse the rotated bin coordinate before discretization."
    )

    # Marching cubes / render
    parser.add_argument(
        "--iso_sigma",
        type=float,
        default=0.15,
        help="Gaussian smoothing sigma before marching cubes."
    )
    parser.add_argument(
        "--iso_level",
        type=float,
        default=0.33,
        help="Marching cubes isosurface level."
    )
    parser.add_argument(
        "--iso_max_dim",
        type=int,
        default=800,
        help="Maximum dimension used for downsampled marching cubes."
    )
    parser.add_argument(
        "--min_voxels_to_render",
        type=int,
        default=50,
        help="Minimum occupied voxels needed to render."
    )

    parser.add_argument(
        "--view_elev",
        type=float,
        default=25.0,
        help="3D render camera elevation."
    )
    parser.add_argument(
        "--view_azim",
        type=float,
        default=-90.0,
        help="3D render camera azimuth."
    )
    parser.add_argument(
        "--black_bg",
        type=str2bool,
        default=True,
        help="Use black background when rendering. Default matches the original script."
    )
    parser.add_argument(
        "--no_axes",
        type=str2bool,
        default=True,
        help="Hide axes when rendering. Default matches the original script."
    )

    # Rotation
    parser.add_argument(
        "--rot_yaw_deg",
        type=float,
        default=0.0,
        help="Yaw rotation angle around Z."
    )
    parser.add_argument(
        "--rot_pitch_deg",
        type=float,
        default=0.0,
        help="Pitch rotation angle around X."
    )
    parser.add_argument(
        "--rot_roll_deg",
        type=float,
        default=0.0,
        help="Roll rotation angle around Y."
    )
    parser.add_argument(
        "--rotate_around_center",
        type=str2bool,
        default=True,
        help="Rotate around mesh center. Default matches the original script."
    )

    # Mask from rendered RGB
    parser.add_argument(
        "--mask_thresh",
        type=float,
        default=1e-4,
        help="Luminance threshold used to convert rendered RGB to binary mask."
    )

    # Optical parameters
    parser.add_argument(
        "--wavelength_r",
        type=float,
        default=532e-9,
        help="Red wavelength in meters."
    )
    parser.add_argument(
        "--wavelength_g",
        type=float,
        default=532e-9,
        help="Green wavelength in meters."
    )
    parser.add_argument(
        "--wavelength_b",
        type=float,
        default=532e-9,
        help="Blue wavelength in meters."
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

    # Random phase
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=2024,
        help="Random seed for shared random phase."
    )

    # Reconstruction display
    parser.add_argument(
        "--use_intensity",
        type=str2bool,
        default=False,
        help="False: |U|, True: |U|^2. Default matches the original script."
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
        help="Output reconstruction size."
    )
    parser.add_argument(
        "--p_low",
        type=float,
        default=1.0,
        help="Lower percentile used for reconstruction visualization."
    )
    parser.add_argument(
        "--p_high",
        type=float,
        default=99.0,
        help="Upper percentile used for reconstruction visualization."
    )

    parser.add_argument(
        "--norm_p_low",
        type=float,
        default=1.0,
        help="Lower percentile used in normalize_to_u8 (kept for behavior compatibility)."
    )
    parser.add_argument(
        "--norm_p_high",
        type=float,
        default=99.0,
        help="Upper percentile used in normalize_to_u8 (kept for behavior compatibility)."
    )

    parser.add_argument(
        "--fullview_name",
        type=str,
        default="full_volume_view.png",
        help="Filename for the full-volume rendered view."
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# ASM propagation
# -----------------------------------------------------------------------------
def asm_propagate(u_in, dist, wavelength, pitch):
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
    mask = term >= 0
    transfer[mask] = np.exp(1j * k * dist * np.sqrt(term[mask]).astype(np.float32))

    u_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_in)))
    u_z = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(u_fft * transfer)))
    return u_z


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def ensure_dirs(args):
    dir_bin = os.path.join(args.out_root, "Binned")
    dir_holo = os.path.join(args.out_root, "Hologram")
    dir_recon = os.path.join(args.out_root, "Recon")

    for directory in [args.out_root, dir_bin, dir_holo, dir_recon]:
        os.makedirs(directory, exist_ok=True)

    return dir_bin, dir_holo, dir_recon


def normalize_to_u8(img, args):
    x = img.astype(np.float32)
    lo = float(np.percentile(x, args.norm_p_low))
    hi = float(np.percentile(x, args.norm_p_high))
    x = np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)
    return (255.0 * x).astype(np.uint8)


def phase_to_u8(phase):
    return (((phase + np.pi) / (2 * np.pi)) * 255.0).clip(0, 255).astype(np.uint8)


def save_float_image(x: np.ndarray, path: str):
    x = np.asarray(x, dtype=np.float32)
    x = x - x.min()
    if x.max() > 1e-8:
        x = x / x.max()
    image = (x * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(path, image)


def save_phase_image(phase: np.ndarray, path: str):
    phase = np.asarray(phase, dtype=np.float32)
    image = ((phase + np.pi) / (2 * np.pi) * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(path, image)


def normalize_rgb_global_with_gamma(rgb_float: np.ndarray, gamma: float = 1.0):
    """
    Original behavior:
        global min-max normalization + gamma correction
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
):
    """
    Percentile-based normalization + gamma correction.
    """
    x = np.asarray(rgb_float, dtype=np.float32)
    x = np.clip(x, 0.0, None)

    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    x = np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)

    if gamma is not None and abs(gamma - 1.0) > 1e-12:
        x = np.power(x, float(gamma))

    return (x * 255.0).clip(0, 255).astype(np.uint8)


def resize_keep_aspect_to_square_rgb(img_u8: np.ndarray, target: int = 2048, pad_value: int = 0):
    height, width, _ = img_u8.shape
    scale = float(target) / float(max(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(img_u8, (new_width, new_height), interpolation=cv2.INTER_AREA)

    out = np.full((target, target, 3), pad_value, dtype=np.uint8)
    y0 = (target - new_height) // 2
    x0 = (target - new_width) // 2
    out[y0:y0 + new_height, x0:x0 + new_width, :] = resized
    return out


def set_black_background(fig, ax):
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    try:
        ax.xaxis.pane.set_facecolor((0, 0, 0, 0))
        ax.yaxis.pane.set_facecolor((0, 0, 0, 0))
        ax.zaxis.pane.set_facecolor((0, 0, 0, 0))
        ax.xaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.yaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.zaxis.pane.set_edgecolor((0, 0, 0, 0))
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Geometry processing
# -----------------------------------------------------------------------------
def preprocess_volume(vol_zyx: np.ndarray, args):
    nz, ny, nx = vol_zyx.shape
    step = max(1, int(np.ceil(max(nz, ny, nx) / float(args.iso_max_dim))))
    volume = vol_zyx[::step, ::step, ::step].astype(np.float32)

    if args.iso_sigma and args.iso_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            volume = gaussian_filter(volume, sigma=float(args.iso_sigma))
        except Exception as exc:
            print("[WARN] gaussian_filter failed, skip smoothing:", exc)

    return volume, step


def mc_mesh(vol_zyx: np.ndarray, args):
    if int((vol_zyx > 0.5).sum()) < args.min_voxels_to_render:
        return None, None, None

    volume, step = preprocess_volume(vol_zyx, args)
    if np.max(volume) <= 1e-6:
        return None, None, step

    try:
        verts, faces, _, _ = marching_cubes(volume, level=float(args.iso_level))
    except Exception as exc:
        print("[WARN] marching_cubes failed:", exc)
        return None, None, step

    return verts, faces, step


def rotate_points_euler(pts_xyz: np.ndarray, yaw_deg, pitch_deg, roll_deg, around_center=True):
    """
    Keep the original rotation order exactly:
        R = Rz @ Rx @ Ry
    """
    pts = pts_xyz.astype(np.float32, copy=False)

    if around_center:
        center = pts.mean(0, keepdims=True)
        pts = pts - center
    else:
        center = np.zeros((1, 3), np.float32)

    yaw, pitch, roll = map(np.deg2rad, [yaw_deg, pitch_deg, roll_deg])

    rot_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw),  np.cos(yaw), 0],
                      [0, 0, 1]], dtype=np.float32)
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(pitch), -np.sin(pitch)],
                      [0, np.sin(pitch),  np.cos(pitch)]], dtype=np.float32)
    rot_y = np.array([[ np.cos(roll), 0, np.sin(roll)],
                      [0, 1, 0],
                      [-np.sin(roll), 0, np.cos(roll)]], dtype=np.float32)

    pts = pts @ (rot_z @ rot_x @ rot_y).T

    if around_center:
        pts = pts + center
    return pts


def render_mesh_to_png_and_rgb(verts_xyz, faces, out_png, global_xlim, global_ylim, global_zlim, args):
    figsize = (args.target_width / args.dpi, args.target_height / args.dpi)
    fig = plt.figure(figsize=figsize, dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")

    x = verts_xyz[:, 0]
    y = verts_xyz[:, 1]
    z = verts_xyz[:, 2]

    ax.plot_trisurf(x, y, z, triangles=faces, linewidth=0.0, alpha=1.0, shade=True)
    ax.view_init(elev=float(args.view_elev), azim=float(args.view_azim))

    ax.set_xlim(*global_xlim)
    ax.set_ylim(*global_ylim)
    ax.set_zlim(*global_zlim)

    x_span = float(global_xlim[1] - global_xlim[0])
    y_span = float(global_ylim[1] - global_ylim[0])
    z_span = float(global_zlim[1] - global_zlim[0])
    try:
        ax.set_box_aspect((x_span, y_span, z_span))
    except Exception:
        pass

    if args.black_bg:
        set_black_background(fig, ax)

    if args.no_axes:
        ax.set_axis_off()
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    fig.savefig(out_png, pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)

    img_bgr = cv2.imread(out_png, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read back rendered image: {out_png}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img_rgb


# -----------------------------------------------------------------------------
# Voxel -> rendered RGB bins -> RGB_bins / M_bins
# -----------------------------------------------------------------------------
def build_rendered_bins_from_voxel_rgb(args, dir_bin):
    occ = np.load(args.npz_path)["occ"].astype(np.float32)  # (Z, Y, X)
    print("[INFO] occ shape (Z, Y, X) =", occ.shape)

    verts_zyx, faces, step = mc_mesh(occ, args)
    if verts_zyx is None:
        raise RuntimeError("Full volume marching_cubes failed (empty or too sparse).")

    verts_zyx = verts_zyx * float(step)
    verts_xyz = np.stack([verts_zyx[:, 2], verts_zyx[:, 1], verts_zyx[:, 0]], axis=1).astype(np.float32)

    verts_xyz_rot = rotate_points_euler(
        verts_xyz,
        args.rot_yaw_deg,
        args.rot_pitch_deg,
        args.rot_roll_deg,
        args.rotate_around_center
    )

    x = verts_xyz_rot[:, 0]
    y = verts_xyz_rot[:, 1]
    z = verts_xyz_rot[:, 2]
    global_xlim = (float(x.min()), float(x.max()))
    global_ylim = (float(y.min()), float(y.max()))
    global_zlim = (float(z.min()), float(z.max()))

    full_png = os.path.join(dir_bin, args.fullview_name)
    _ = render_mesh_to_png_and_rgb(verts_xyz_rot, faces, full_png, global_xlim, global_ylim, global_zlim, args)

    axis_id = {"x": 0, "y": 1, "z": 2}.get(args.bin_axis_rotated.lower(), None)
    if axis_id is None:
        raise ValueError("bin_axis_rotated must be 'x', 'y', or 'z'.")

    coord = verts_xyz_rot[:, axis_id].copy()
    cmin, cmax = float(coord.min()), float(coord.max())

    if args.bin_reverse:
        coord = -coord
        cmin, cmax = float(coord.min()), float(coord.max())

    edges_mm = np.linspace(args.z_min_mm, args.z_max_mm, args.num_bins + 1, dtype=np.float32)
    centers_mm = 0.5 * (edges_mm[:-1] + edges_mm[1:])
    edges_coord = np.linspace(cmin, cmax, args.num_bins + 1, dtype=np.float32)

    rgb_bins = []
    mask_bins = []
    counts = np.zeros((args.num_bins,), dtype=np.int64)

    with open(os.path.join(dir_bin, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"NPZ={args.npz_path}\n")
        f.write(f"occ_shape={occ.shape}\n")
        f.write(f"ROT(yaw,pitch,roll)=({args.rot_yaw_deg},{args.rot_pitch_deg},{args.rot_roll_deg})\n")
        f.write(f"BIN_AXIS_ROTATED={args.bin_axis_rotated}, BIN_REVERSE={args.bin_reverse}\n")
        f.write(f"edges_mm={' '.join([f'{e:.6f}' for e in edges_mm])}\n")
        f.write(f"centers_mm={' '.join([f'{c:.6f}' for c in centers_mm])}\n")

    f0 = faces[:, 0]
    f1 = faces[:, 1]
    f2 = faces[:, 2]

    for k in range(args.num_bins):
        c0 = float(edges_coord[k])
        c1 = float(edges_coord[k + 1])
        in_bin_v = (coord >= c0) & (coord < c1)

        in_face = in_bin_v[f0] & in_bin_v[f1] & in_bin_v[f2]
        faces_k = faces[in_face]

        out_dir = os.path.join(dir_bin, f"plane_{k:02d}_z{centers_mm[k]:.3f}mm")
        os.makedirs(out_dir, exist_ok=True)

        if len(faces_k) == 0:
            rgb = np.zeros((args.target_height, args.target_width, 3), np.float32)
        else:
            out_png = os.path.join(out_dir, "render.png")
            rgb = render_mesh_to_png_and_rgb(
                verts_xyz_rot,
                faces_k,
                out_png,
                global_xlim,
                global_ylim,
                global_zlim,
                args
            )

        lum = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
        mask = (lum > args.mask_thresh).astype(np.uint8) * 255

        rgb_bins.append(rgb.astype(np.float32))
        mask_bins.append(mask)
        counts[k] = int(np.sum(mask > 0))

        rgb_u8 = (np.clip(rgb, 0, 1) * 255.0).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, "rgb.png"), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_dir, "mask.png"), mask)
        with open(os.path.join(out_dir, "depth_mm.txt"), "w", encoding="utf-8") as f:
            f.write(f"{centers_mm[k]:.6f}\n")

        print(f"[OK] bin{k}: center={centers_mm[k]:.3f}mm, kept={counts[k]}")

    return edges_mm, centers_mm, rgb_bins, mask_bins, counts


# -----------------------------------------------------------------------------
# Generate RGB complex holograms
# -----------------------------------------------------------------------------
def generate_hologram_complex_rgb(rgb_bins, mask_bins, centers_mm, args):
    """
    Output:
        holo_r, holo_g, holo_b

    Keep the original behavior exactly:
        each bin shares one random phase map phi(H, W) across RGB channels.
    """
    rng = np.random.default_rng(args.rng_seed)
    height, width, _ = rgb_bins[0].shape

    u_holo = [
        np.zeros((height, width), dtype=np.complex64),
        np.zeros((height, width), dtype=np.complex64),
        np.zeros((height, width), dtype=np.complex64),
    ]

    wavelengths = [args.wavelength_r, args.wavelength_g, args.wavelength_b]

    for k in tqdm(range(args.num_bins), desc="Generate COMPLEX RGB hologram (rendered bins, shared random phase)"):
        mk = (mask_bins[k] > 0)
        if not np.any(mk):
            continue

        rgb = np.clip(rgb_bins[k], 0.0, 1.0).astype(np.float32)

        amplitude = rgb

        phi = rng.uniform(-np.pi, np.pi, size=(height, width)).astype(np.float32)
        phasor = np.exp(1j * phi).astype(np.complex64)

        z_m = float(centers_mm[k]) * 1e-3
        dist = float(args.holo_z_m - z_m)

        for c in range(3):
            u_obj = np.zeros((height, width), dtype=np.complex64)
            u_obj[mk] = amplitude[..., c][mk].astype(np.complex64) * phasor[mk]
            u_holo[c] += asm_propagate(u_obj, dist, wavelengths[c], args.pixel_pitch)

    return u_holo[0], u_holo[1], u_holo[2]


# -----------------------------------------------------------------------------
# Reconstruct RGB slices
# -----------------------------------------------------------------------------
def reconstruct_slices_rgb(holo_r, holo_g, holo_b, centers_mm, args, dir_recon):
    os.makedirs(dir_recon, exist_ok=True)

    wavelengths = [args.wavelength_r, args.wavelength_g, args.wavelength_b]

    for k in tqdm(range(args.num_bins), desc="Reconstruct K slices (RGB, complex hologram)"):
        z_m = float(centers_mm[k]) * 1e-3
        dist = float(z_m - args.holo_z_m)

        rec_r = asm_propagate(holo_r, dist, wavelengths[0], args.pixel_pitch)
        rec_g = asm_propagate(holo_g, dist, wavelengths[1], args.pixel_pitch)
        rec_b = asm_propagate(holo_b, dist, wavelengths[2], args.pixel_pitch)

        if args.use_intensity:
            img_r = (np.abs(rec_r) ** 2).astype(np.float32)
            img_g = (np.abs(rec_g) ** 2).astype(np.float32)
            img_b = (np.abs(rec_b) ** 2).astype(np.float32)
        else:
            img_r = np.abs(rec_r).astype(np.float32)
            img_g = np.abs(rec_g).astype(np.float32)
            img_b = np.abs(rec_b).astype(np.float32)

        rgb = np.stack([img_r, img_g, img_b], axis=-1)

        rgb_u8 = normalize_rgb_percentile_with_gamma(
            rgb,
            gamma=args.gamma,
            p_low=args.p_low,
            p_high=args.p_high
        )
        rgb_u8 = resize_keep_aspect_to_square_rgb(rgb_u8, target=args.out_size, pad_value=0)

        out_path = os.path.join(dir_recon, f"Recon_plane{k:02d}_z{centers_mm[k]:.3f}mm.png")
        cv2.imwrite(out_path, cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.target_height <= 0 or args.target_width <= 0:
        raise ValueError("--target_height and --target_width must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    if args.z_max_mm <= args.z_min_mm:
        raise ValueError("--z_max_mm must be greater than --z_min_mm.")
    if args.num_bins <= 0:
        raise ValueError("--num_bins must be positive.")
    if args.pixel_pitch <= 0:
        raise ValueError("--pixel_pitch must be positive.")
    if args.wavelength_r <= 0 or args.wavelength_g <= 0 or args.wavelength_b <= 0:
        raise ValueError("All wavelengths must be positive.")
    if args.out_size <= 0:
        raise ValueError("--out_size must be positive.")
    if not (0.0 <= args.p_low < args.p_high <= 100.0):
        raise ValueError("--p_low and --p_high must satisfy 0 <= p_low < p_high <= 100.")

    begin = time.time()
    dir_bin, dir_holo, dir_recon = ensure_dirs(args)

    edges, centers, rgb_bins, mask_bins, counts = build_rendered_bins_from_voxel_rgb(args, dir_bin)

    holo_r, holo_g, holo_b = generate_hologram_complex_rgb(rgb_bins, mask_bins, centers, args)

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

    np.save(os.path.join(dir_holo, "pha_R.npy"), pha_r)
    np.save(os.path.join(dir_holo, "pha_G.npy"), pha_g)
    np.save(os.path.join(dir_holo, "pha_B.npy"), pha_b)

    save_float_image(amp_r, os.path.join(dir_holo, "amp_R.png"))
    save_float_image(amp_g, os.path.join(dir_holo, "amp_G.png"))
    save_float_image(amp_b, os.path.join(dir_holo, "amp_B.png"))

    save_phase_image(pha_r, os.path.join(dir_holo, "phase_R.png"))
    save_phase_image(pha_g, os.path.join(dir_holo, "phase_G.png"))
    save_phase_image(pha_b, os.path.join(dir_holo, "phase_B.png"))

    holo_end = time.time()
    print(f"[INFO] Hologram generation time: {holo_end - begin:.2f} seconds")

    reconstruct_slices_rgb(holo_r, holo_g, holo_b, centers, args, dir_recon)

    print("[DONE]")
    print("Binned planes:", dir_bin)
    print("Hologram:", dir_holo)
    print("Recon:", dir_recon)
    print(
        f"[INFO] Display: USE_INTENSITY={args.use_intensity}, "
        f"GAMMA={args.gamma}, P_LOW={args.p_low}, P_HIGH={args.p_high}, OUT_SIZE={args.out_size}"
    )


if __name__ == "__main__":
    main()