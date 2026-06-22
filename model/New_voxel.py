#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Complex RGB Holograms from Voxel Volume Integral and Reconstruct RGB Slices

Author: Hao Yun
Date: 2026-04-13

Description:
    This script directly uses voxel occupancy volume to generate holograms
    through a volume-integral-style discretization.

Core idea:
    1. Load voxel occupancy volume from NPZ
    2. Rotate the volume in 3D
    3. Choose one rotated axis as depth axis
    4. For each depth bin, integrate voxel density inside the slab:
           A_k(u, v) = integral rho(u, v, z) dz
       implemented as summation over voxels in the bin
    5. Use the resulting 2D amplitude layer as the object field of that bin
    6. Apply shared random phase per bin and propagate to hologram plane
    7. Reconstruct RGB slices at all bin-center depths

Important note:
    - This is a direct voxel-volume-based method.
    - It no longer uses marching cubes or surface rendering.
    - Since the current NPZ is assumed to contain only occupancy "occ",
      the generated RGB hologram uses the same amplitude for R/G/B channels.
      If you later have color voxel data, the code can be extended.

Example:
    python /workspace/yh/project/CGHReviewCode/model/New_voxel.py \
      --npz_path /workspace/yh/project/CGHReview/dataset/Voxel3/BunnyDragon_voxel.npz \
      --out_root /workspace/yh/project/CGHReview/result/volume_integral3 \
      --num_bins 4
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Complex RGB Holograms from Voxel Volume Integral and Reconstruct RGB Slices

Author: Hao Yun
Date: 2026-04-13

Description:
    This script directly uses voxel occupancy volume to generate holograms
    through a volume-integral-style discretization, while preserving the
    original object orientation logic:
        1) object Euler rotation
        2) binning along rotated object axis
        3) camera view projection using view_elev / view_azim
        4) volume integration along camera optical axis

Important note:
    - This version no longer uses marching cubes or surface rendering.
    - It keeps the original object rotation parameters.
    - It also restores the original "render view" effect by explicitly
      introducing a camera-coordinate transform before projection.
    - Since the NPZ is assumed to contain only occupancy "occ", the generated
      RGB hologram uses the same amplitude for R/G/B channels.
      If color voxel data is available later, this script can be extended.

Example:
    python ./model/Voxel_ComplexHologram_VolumeIntegral.py \
      --npz_path /workspace/yh/project/CGHReview/dataset/Voxel3/BunnyDragon_voxel.npz \
      --out_root ./result/Voxel/100bins_volume_integral \
      --num_bins 100
"""

import os
import time
import argparse
import numpy as np
import cv2
from tqdm import tqdm

from scipy.ndimage import gaussian_filter, affine_transform


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
        description="Generate complex RGB holograms from voxel volume integral and reconstruct RGB slices."
    )

    parser.add_argument(
        "--npz_path",
        type=str,
        required=True,
        help="Path to the voxel NPZ file. Must contain key 'occ'."
    )
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Output root directory."
    )

    # Output image size
    parser.add_argument(
        "--target_height",
        type=int,
        default=2048,
        help="Target hologram/reconstruction image height."
    )
    parser.add_argument(
        "--target_width",
        type=int,
        default=2048,
        help="Target hologram/reconstruction image width."
    )

    # Fixed physical depth mapping
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

    # Bin slicing axis after object rotation
    parser.add_argument(
        "--bin_axis_rotated",
        type=str,
        default="y",
        choices=["x", "y", "z"],
        help="Axis used for binning after object rotation."
    )
    parser.add_argument(
        "--bin_reverse",
        type=str2bool,
        default=False,
        help="Whether to reverse the rotated bin coordinate before discretization."
    )

    # Volume preprocessing
    parser.add_argument(
        "--iso_sigma",
        type=float,
        default=0.15,
        help="Gaussian smoothing sigma applied directly to the voxel volume."
    )
    parser.add_argument(
        "--min_voxels_to_render",
        type=int,
        default=50,
        help="Minimum occupied voxels needed to proceed."
    )

    # These are now truly used to restore the original viewing direction
    parser.add_argument(
        "--view_elev",
        type=float,
        default=25.0,
        help="Camera elevation angle, consistent with the original matplotlib render."
    )
    parser.add_argument(
        "--view_azim",
        type=float,
        default=-90.0,
        help="Camera azimuth angle, consistent with the original matplotlib render."
    )

    # Optional image coordinate correction
    parser.add_argument(
        "--camera_flip_ud",
        type=str2bool,
        default=True,
        help="Flip the projected image vertically after camera projection."
    )
    parser.add_argument(
        "--camera_flip_lr",
        type=str2bool,
        default=False,
        help="Flip the projected image horizontally after camera projection."
    )

    # Kept for compatibility with old CLI
    parser.add_argument(
        "--dpi",
        type=int,
        default=256,
        help="Kept for compatibility; not used in volume-integral mode."
    )
    parser.add_argument(
        "--black_bg",
        type=str2bool,
        default=True,
        help="Kept for compatibility; not used in volume-integral mode."
    )
    parser.add_argument(
        "--no_axes",
        type=str2bool,
        default=True,
        help="Kept for compatibility; not used in volume-integral mode."
    )

    # Rotation (same as original logic)
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
        help="Rotate around volume center."
    )

    # Mask threshold
    parser.add_argument(
        "--mask_thresh",
        type=float,
        default=1e-4,
        help="Threshold used on normalized slab-integral amplitude to generate mask."
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
        help="False: |U|, True: |U|^2."
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
        help="Lower percentile used in normalize_to_u8 (kept for compatibility)."
    )
    parser.add_argument(
        "--norm_p_high",
        type=float,
        default=99.0,
        help="Upper percentile used in normalize_to_u8 (kept for compatibility)."
    )

    parser.add_argument(
        "--fullview_name",
        type=str,
        default="full_volume_view.png",
        help="Filename for the full-volume projected preview."
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


def normalize_rgb_percentile_with_gamma(
    rgb_float: np.ndarray,
    gamma: float = 1.0,
    p_low: float = 1.0,
    p_high: float = 99.0
):
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

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img_u8, (new_width, new_height), interpolation=interp)

    out = np.full((target, target, 3), pad_value, dtype=np.uint8)
    y0 = (target - new_height) // 2
    x0 = (target - new_width) // 2
    out[y0:y0 + new_height, x0:x0 + new_width, :] = resized
    return out


def resize_keep_aspect_to_canvas_gray(img_f32: np.ndarray, target_h: int, target_w: int, pad_value: float = 0.0):
    height, width = img_f32.shape
    scale = min(float(target_h) / float(height), float(target_w) / float(width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img_f32.astype(np.float32), (new_width, new_height), interpolation=interp)

    out = np.full((target_h, target_w), pad_value, dtype=np.float32)
    y0 = (target_h - new_height) // 2
    x0 = (target_w - new_width) // 2
    out[y0:y0 + new_height, x0:x0 + new_width] = resized
    return out


def save_gray_preview(path: str, img_f32: np.ndarray):
    x = np.asarray(img_f32, dtype=np.float32)
    x = x - x.min()
    if x.max() > 1e-8:
        x = x / x.max()
    u8 = (x * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(path, u8)


# -----------------------------------------------------------------------------
# Geometry / camera transforms
# -----------------------------------------------------------------------------
AXIS2ID = {"x": 0, "y": 1, "z": 2}


def rotation_matrix_object(yaw_deg, pitch_deg, roll_deg):
    """
    Keep the original rotation order exactly:
        R = Rz @ Rx @ Ry
    """
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

    return (rot_z @ rot_x @ rot_y).astype(np.float32)


def camera_matrix_from_view(elev_deg, azim_deg):
    """
    Build a camera-coordinate rotation matrix from matplotlib-like elev/azim.

    Output axes:
        x_cam: right
        y_cam: up
        z_cam: forward (from camera to scene)
    """
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)

    # Camera position on a unit sphere, looking toward the origin
    cam_pos = np.array([
        np.cos(elev) * np.cos(azim),
        np.cos(elev) * np.sin(azim),
        np.sin(elev)
    ], dtype=np.float32)

    forward = -cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-12)

    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(forward, up_world))) > 0.999:
        up_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    right = np.cross(forward, up_world)
    right = right / (np.linalg.norm(right) + 1e-12)

    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-12)

    # p_cam = R_cam @ p_world
    r_cam = np.stack([right, up, forward], axis=0).astype(np.float32)
    return r_cam


def build_affine_spec(shape_in_xyz, forward_matrix):
    """
    Build affine-transform spec for scipy.ndimage.affine_transform.

    forward_matrix maps:
        centered_input_coord -> centered_output_coord
    """
    shape_in = np.array(shape_in_xyz, dtype=np.float32)
    center_in = (shape_in - 1.0) / 2.0

    nx, ny, nz = shape_in_xyz
    corners = np.array([
        [0,      0,      0],
        [0,      0,      nz - 1],
        [0,      ny - 1, 0],
        [0,      ny - 1, nz - 1],
        [nx - 1, 0,      0],
        [nx - 1, 0,      nz - 1],
        [nx - 1, ny - 1, 0],
        [nx - 1, ny - 1, nz - 1],
    ], dtype=np.float32)

    corners_centered = corners - center_in[None, :]
    corners_out = (forward_matrix @ corners_centered.T).T

    half_extent = np.max(np.abs(corners_out), axis=0)
    shape_out = np.ceil(2.0 * half_extent + 1.0).astype(np.int32)
    shape_out = np.maximum(shape_out, 1)

    center_out = (shape_out.astype(np.float32) - 1.0) / 2.0

    matrix_inv = np.linalg.inv(forward_matrix).astype(np.float32)
    offset = center_in - matrix_inv @ center_out

    return {
        "matrix": matrix_inv,
        "offset": offset.astype(np.float32),
        "output_shape": tuple(shape_out.tolist())
    }


def apply_affine_spec(vol_xyz: np.ndarray, spec):
    vol_out = affine_transform(
        input=vol_xyz,
        matrix=spec["matrix"],
        offset=spec["offset"],
        output_shape=spec["output_shape"],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False
    )
    vol_out = np.clip(vol_out.astype(np.float32), 0.0, None)
    return vol_out


def project_camera_volume_to_hw(vol_cam_xyz: np.ndarray, flip_ud: bool = True, flip_lr: bool = False):
    """
    Camera coordinates are assumed to be:
        X -> right
        Y -> up
        Z -> forward(depth)

    Projection:
        integrate along Z, then map to image H/W.
    """
    amp_xy = np.sum(vol_cam_xyz, axis=2).astype(np.float32)   # (X, Y)
    amp_hw = np.transpose(amp_xy, (1, 0)).astype(np.float32)  # (Y, X)

    if flip_ud:
        amp_hw = np.flipud(amp_hw)
    if flip_lr:
        amp_hw = np.fliplr(amp_hw)

    return amp_hw


# -----------------------------------------------------------------------------
# Volume preparation
# -----------------------------------------------------------------------------
def load_and_prepare_volume(args):
    data = np.load(args.npz_path)
    if "occ" not in data:
        raise KeyError(f"NPZ file must contain key 'occ'. Found keys: {list(data.keys())}")

    occ_zyx = np.asarray(data["occ"], dtype=np.float32)
    if occ_zyx.ndim != 3:
        raise ValueError(f"'occ' must be a 3D array, but got shape {occ_zyx.shape}")

    occ_zyx = np.clip(occ_zyx, 0.0, 1.0)

    if float(np.sum(occ_zyx > 0.0)) < args.min_voxels_to_render:
        raise RuntimeError("Voxel volume is too sparse or empty.")

    if args.iso_sigma and args.iso_sigma > 0:
        occ_zyx = gaussian_filter(occ_zyx, sigma=float(args.iso_sigma))
        occ_zyx = np.clip(occ_zyx, 0.0, None)

    # Convert from (Z, Y, X) to (X, Y, Z)
    vol_xyz = np.transpose(occ_zyx, (2, 1, 0)).astype(np.float32)
    return vol_xyz


# -----------------------------------------------------------------------------
# Voxel volume integral -> binned 2D amplitudes
# -----------------------------------------------------------------------------
def build_volume_integral_bins(args, dir_bin):
    vol_xyz = load_and_prepare_volume(args)
    print("[INFO] input volume shape (X, Y, Z) =", vol_xyz.shape)

    # Step 1: object rotation (same role as original rotate_points_euler)
    r_obj = rotation_matrix_object(
        args.rot_yaw_deg,
        args.rot_pitch_deg,
        args.rot_roll_deg
    )
    spec_obj = build_affine_spec(vol_xyz.shape, r_obj)
    vol_obj = apply_affine_spec(vol_xyz, spec_obj)
    print("[INFO] object-rotated volume shape (X, Y, Z) =", vol_obj.shape)

    # Step 2: binning axis is defined in the rotated object coordinate system
    axis_id = AXIS2ID[args.bin_axis_rotated.lower()]
    axis_size = vol_obj.shape[axis_id]

    edges_mm = np.linspace(args.z_min_mm, args.z_max_mm, args.num_bins + 1, dtype=np.float32)
    centers_mm = 0.5 * (edges_mm[:-1] + edges_mm[1:])
    edges_idx_float = np.linspace(0.0, float(axis_size), args.num_bins + 1)

    # Step 3: restore original render view using camera transform
    r_cam = camera_matrix_from_view(args.view_elev, args.view_azim)
    spec_cam = build_affine_spec(vol_obj.shape, r_cam)

    # Full-view preview
    vol_cam_full = apply_affine_spec(vol_obj, spec_cam)
    full_amp_native = project_camera_volume_to_hw(
        vol_cam_full,
        flip_ud=args.camera_flip_ud,
        flip_lr=args.camera_flip_lr
    )
    full_amp_canvas = resize_keep_aspect_to_canvas_gray(
        full_amp_native,
        args.target_height,
        args.target_width,
        pad_value=0.0
    )
    save_gray_preview(os.path.join(dir_bin, args.fullview_name), full_amp_canvas)

    global_max = max(float(full_amp_native.max()), 1e-8)

    with open(os.path.join(dir_bin, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"NPZ={args.npz_path}\n")
        f.write(f"INPUT_VOLUME_XYZ_SHAPE={vol_xyz.shape}\n")
        f.write(f"OBJECT_ROTATED_VOLUME_XYZ_SHAPE={vol_obj.shape}\n")
        f.write(f"ROT(yaw,pitch,roll)=({args.rot_yaw_deg},{args.rot_pitch_deg},{args.rot_roll_deg})\n")
        f.write(f"BIN_AXIS_ROTATED={args.bin_axis_rotated}, BIN_REVERSE={args.bin_reverse}\n")
        f.write(f"VIEW_ELEV={args.view_elev}, VIEW_AZIM={args.view_azim}\n")
        f.write(f"CAMERA_FLIP_UD={args.camera_flip_ud}, CAMERA_FLIP_LR={args.camera_flip_lr}\n")
        f.write(f"edges_mm={' '.join([f'{e:.6f}' for e in edges_mm])}\n")
        f.write(f"centers_mm={' '.join([f'{c:.6f}' for c in centers_mm])}\n")
        f.write(f"edges_idx_float={' '.join([f'{e:.6f}' for e in edges_idx_float])}\n")

    rgb_bins = []
    mask_bins = []
    counts = np.zeros((args.num_bins,), dtype=np.int64)

    for k in range(args.num_bins):
        # Reverse only affects which side maps to low/high physical depth,
        # while geometry orientation itself remains unchanged.
        seg = (args.num_bins - 1 - k) if args.bin_reverse else k

        i0 = int(np.floor(edges_idx_float[seg]))
        i1 = int(np.floor(edges_idx_float[seg + 1]))
        if seg == args.num_bins - 1:
            i1 = axis_size

        i0 = max(0, min(i0, axis_size))
        i1 = max(0, min(i1, axis_size))

        slab_full = np.zeros_like(vol_obj, dtype=np.float32)

        if i1 > i0:
            slicer = [slice(None), slice(None), slice(None)]
            slicer[axis_id] = slice(i0, i1)
            slab_full[tuple(slicer)] = vol_obj[tuple(slicer)]

        # Project this slab with the SAME camera transform as the full object
        slab_cam = apply_affine_spec(slab_full, spec_cam)
        amp_native = project_camera_volume_to_hw(
            slab_cam,
            flip_ud=args.camera_flip_ud,
            flip_lr=args.camera_flip_lr
        )

        amp_norm_native = np.clip(amp_native / global_max, 0.0, 1.0)
        amp_canvas = resize_keep_aspect_to_canvas_gray(
            amp_norm_native,
            args.target_height,
            args.target_width,
            pad_value=0.0
        )
        amp_canvas = np.clip(amp_canvas, 0.0, 1.0).astype(np.float32)

        rgb = np.repeat(amp_canvas[..., None], 3, axis=-1)
        mask = (amp_canvas > args.mask_thresh).astype(np.uint8) * 255

        rgb_bins.append(rgb)
        mask_bins.append(mask)
        counts[k] = int(np.sum(mask > 0))

        out_dir = os.path.join(dir_bin, f"plane_{k:02d}_z{centers_mm[k]:.3f}mm")
        os.makedirs(out_dir, exist_ok=True)

        rgb_u8 = (amp_canvas * 255.0).clip(0, 255).astype(np.uint8)
        rgb_u8_3 = np.repeat(rgb_u8[..., None], 3, axis=-1)

        cv2.imwrite(os.path.join(out_dir, "rgb.png"), cv2.cvtColor(rgb_u8_3, cv2.COLOR_RGB2BGR))
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
    Each bin is a slab-integrated amplitude layer.
    Shared random phase is still used for each bin across RGB channels.
    """
    rng = np.random.default_rng(args.rng_seed)
    height, width, _ = rgb_bins[0].shape

    u_holo = [
        np.zeros((height, width), dtype=np.complex64),
        np.zeros((height, width), dtype=np.complex64),
        np.zeros((height, width), dtype=np.complex64),
    ]

    wavelengths = [args.wavelength_r, args.wavelength_g, args.wavelength_b]

    for k in tqdm(range(args.num_bins), desc="Generate COMPLEX RGB hologram (volume integral bins)"):
        mk = (mask_bins[k] > 0)
        if not np.any(mk):
            continue

        rgb = np.clip(rgb_bins[k], 0.0, 1.0).astype(np.float32)

        phi = rng.uniform(-np.pi, np.pi, size=(height, width)).astype(np.float32)
        phasor = np.exp(1j * phi).astype(np.complex64)

        z_m = float(centers_mm[k]) * 1e-3
        dist = float(args.holo_z_m - z_m)

        for c in range(3):
            u_obj = np.zeros((height, width), dtype=np.complex64)
            u_obj[mk] = rgb[..., c][mk].astype(np.complex64) * phasor[mk]
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

    edges, centers, rgb_bins, mask_bins, counts = build_volume_integral_bins(args, dir_bin)

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