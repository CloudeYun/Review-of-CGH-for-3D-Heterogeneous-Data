#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Complex RGB Holograms Directly from an instant-ngp NeRF Snapshot

Author: Hao Yun
Date: 2026-06-22

Description:
    This script combines NeRF light-field rendering and perspective
    light-field hologram generation into one end-to-end pipeline. It loads a
    trained instant-ngp snapshot, renders a regular perspective light field,
    saves the intermediate light-field dataset, generates RGB complex
    holograms, and reconstructs images at multiple axial distances.

Pipeline:
    1. Load an instant-ngp snapshot and a reference transforms.json
    2. Render a regular camera array with parallel optical axes
    3. Save intermediate light-field images and camera metadata
    4. Refocus the light field to a reference depth
    5. Convert each RGB channel into a tiled complex field
    6. Propagate the complex fields to the hologram plane using ASM
    7. Save complex, amplitude, and phase holograms
    8. Reconstruct RGB images at multiple propagation distances

Output structure:
    output_dir/
        LightField/
            images/
            poses.csv
            intrinsics.csv
            reference_pose.json
            time.csv
        Hologram/
            hologram_complex_{R,G,B}.npy
            hologram_amp_{R,G,B}.npy
            hologram_phase_{R,G,B}.npy
        Recon/
            recon_z_*.png

Example:
    PYTHONPATH=/path/to/instant-ngp/build:$PYTHONPATH \
    python ./model/NeRF_ComplexHologram.py \
        --snapshot ./dataset/NeRF/model.ingp \
        --ref_transforms ./dataset/NeRF/multiview/transforms.json \
        --output_dir ./result/NeRF\
        --nu 40 \
        --nv 40 \
        --lf_width 800 \
        --lf_height 800 \
        --total_baseline_x 0.20 \
        --total_baseline_z 0.20 \
        --scene_unit_in_m 1.0 \
        --allow_large_field \
        --d0 -0.050 \
        --wavelength 532e-9 \
        --pitch 2e-6 \
        --dist_rs_to_h 0.05 \
        --z_min -0.052 \
        --z_max -0.048 \
        --z_step 0.001 \
        --save_holo_png \
        --save_recon_png
"""

import argparse
import csv
import importlib
import json
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


CHANNEL_NAMES = ("R", "G", "B")


# -----------------------------------------------------------------------------
# Command-Line Interface
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a perspective light field from an instant-ngp snapshot "
            "and generate complex RGB holograms."
        )
    )

    # instant-ngp inputs
    parser.add_argument(
        "--snapshot",
        required=True,
        help="Path to a trained instant-ngp snapshot (.ingp or .msgpack).",
    )
    parser.add_argument(
        "--ref_transforms",
        required=True,
        help="Path to the NeRF training transforms.json.",
    )
    parser.add_argument(
        "--instant_ngp_build",
        default="",
        help=(
            "Path to the instant-ngp build directory containing pyngp. "
            "Not required when pyngp is already available on PYTHONPATH."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root directory for light-field, hologram, and reconstruction outputs.",
    )
    parser.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="Reference frame used as the center light-field camera.",
    )

    # Light-field rendering
    parser.add_argument(
        "--lf_width",
        type=int,
        default=400,
        help="Width of each rendered light-field image.",
    )
    parser.add_argument(
        "--lf_height",
        type=int,
        default=400,
        help="Height of each rendered light-field image.",
    )
    parser.add_argument(
        "--fov_deg",
        type=float,
        default=-1.0,
        help=(
            "Horizontal field of view in degrees. A negative value reads "
            "camera_angle_x from the reference transforms file."
        ),
    )
    parser.add_argument("--nu", type=int, default=40, help="Camera samples along u.")
    parser.add_argument("--nv", type=int, default=40, help="Camera samples along v.")
    parser.add_argument(
        "--total_baseline_x",
        type=float,
        default=0.20,
        help="Total horizontal camera baseline in NeRF scene units.",
    )
    parser.add_argument(
        "--total_baseline_z",
        type=float,
        default=0.20,
        help="Total vertical camera baseline in NeRF scene units.",
    )
    parser.add_argument(
        "--scene_unit_in_m",
        type=float,
        default=1.0,
        help="Number of meters represented by one NeRF scene unit.",
    )
    parser.add_argument(
        "--offset_mode",
        choices=["world_xz", "camera_ru"],
        default="world_xz",
        help="Camera-array offset coordinate system.",
    )
    parser.add_argument(
        "--look_at_dist",
        type=float,
        default=200.0,
        help="Virtual look-at distance used to keep all optical axes parallel.",
    )
    parser.add_argument(
        "--spp",
        type=int,
        default=8,
        help="instant-ngp samples per pixel.",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=0.0,
        help="Exposure compensation in stops.",
    )
    parser.add_argument(
        "--render_near_distance",
        type=float,
        default=-1.0,
        help="Optional instant-ngp near distance; negative keeps the snapshot default.",
    )
    parser.add_argument(
        "--background_rgba",
        type=float,
        nargs=4,
        default=[0.0, 0.0, 0.0, 0.0],
        metavar=("R", "G", "B", "A"),
        help="Rendering background color in [0, 1].",
    )
    parser.add_argument(
        "--linear_to_srgb",
        action="store_true",
        help="Convert rendered linear RGB values to sRGB before saving.",
    )
    parser.add_argument(
        "--save_rgba_npz",
        action="store_true",
        help="Save raw instant-ngp RGBA arrays next to the PNG images.",
    )

    # Light-field refocusing and hologram generation
    parser.add_argument(
        "--d0",
        type=float,
        default=-0.050,
        help="Reference light-field refocus depth in meters.",
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=532e-9,
        help="Optical wavelength in meters.",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=2e-6,
        help="Hologram pixel pitch in meters.",
    )
    parser.add_argument(
        "--dist_rs_to_h",
        type=float,
        default=0.05,
        help="Propagation distance from the ray-sampling plane to the hologram plane.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=2026,
        help="Random seed used in light-field-to-complex conversion.",
    )

    # Reconstruction and visualization
    parser.add_argument(
        "--z_list",
        type=float,
        nargs="*",
        default=None,
        help="Explicit reconstruction distances in meters.",
    )
    parser.add_argument("--z_min", type=float, default=-0.060)
    parser.add_argument("--z_max", type=float, default=-0.040)
    parser.add_argument("--z_step", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.65)
    parser.add_argument("--out_size", type=int, default=2048)
    parser.add_argument(
        "--vis_mode",
        choices=["percentile", "minmax"],
        default="percentile",
    )
    parser.add_argument("--p_low", type=float, default=1.0)
    parser.add_argument("--p_high", type=float, default=99.0)
    parser.add_argument(
        "--save_holo_png",
        action="store_true",
        help="Save hologram amplitude and phase visualizations.",
    )
    parser.add_argument(
        "--save_recon_png",
        action="store_true",
        help="Save reconstructed RGB images.",
    )
    parser.add_argument(
        "--allow_large_field",
        action="store_true",
        help=(
            "Allow a tiled complex field larger than the conservative memory "
            "threshold. Use only after checking available RAM."
        ),
    )

    args = parser.parse_args()
    validate_args(parser, args)
    return args


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    """Validate all command-line parameters before loading instant-ngp."""
    if not os.path.isfile(args.snapshot):
        parser.error(f"Snapshot not found: {args.snapshot}")
    if not os.path.isfile(args.ref_transforms):
        parser.error(f"Reference transforms file not found: {args.ref_transforms}")
    if args.instant_ngp_build and not os.path.isdir(args.instant_ngp_build):
        parser.error(
            f"instant-ngp build directory not found: {args.instant_ngp_build}"
        )
    if args.frame_idx < 0:
        parser.error("--frame_idx cannot be negative.")
    if args.lf_width <= 0 or args.lf_height <= 0:
        parser.error("--lf_width and --lf_height must be positive.")
    if args.nu <= 0 or args.nv <= 0:
        parser.error("--nu and --nv must be positive.")
    if args.total_baseline_x < 0 or args.total_baseline_z < 0:
        parser.error("Camera baselines cannot be negative.")
    if args.scene_unit_in_m <= 0:
        parser.error("--scene_unit_in_m must be positive.")
    if args.look_at_dist <= 0:
        parser.error("--look_at_dist must be positive.")
    if args.spp <= 0:
        parser.error("--spp must be positive.")
    if len(args.background_rgba) != 4 or any(
        value < 0.0 or value > 1.0 for value in args.background_rgba
    ):
        parser.error("--background_rgba values must lie in [0, 1].")
    if abs(args.d0) < 1e-12:
        parser.error("--d0 must be non-zero.")
    if args.wavelength <= 0 or args.pitch <= 0:
        parser.error("--wavelength and --pitch must be positive.")
    if args.out_size <= 0:
        parser.error("--out_size must be positive.")
    if not 0.0 <= args.p_low < args.p_high <= 100.0:
        parser.error("--p_low and --p_high must satisfy 0 <= p_low < p_high <= 100.")

    field_height = args.nv * args.lf_height
    field_width = args.nu * args.lf_width
    field_bytes = field_height * field_width * np.dtype(np.complex64).itemsize
    field_gib = field_bytes / (1024.0 ** 3)
    if field_gib > 4.0 and not args.allow_large_field:
        parser.error(
            "The requested tiled complex field is "
            f"{field_height}x{field_width} ({field_gib:.2f} GiB per "
            "complex64 channel before FFT workspaces). Reduce --nu/--nv or "
            "--lf_width/--lf_height, or pass --allow_large_field after "
            "confirming sufficient memory."
        )


# -----------------------------------------------------------------------------
# General Utilities
# -----------------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def import_pyngp(build_dir: str):
    """Import pyngp from PYTHONPATH or an explicitly supplied build directory."""
    if build_dir:
        absolute_build_dir = os.path.abspath(build_dir)
        if absolute_build_dir not in sys.path:
            sys.path.insert(0, absolute_build_dir)

    try:
        return importlib.import_module("pyngp")
    except ImportError as exc:
        raise ImportError(
            "Unable to import pyngp. Build instant-ngp and either set "
            "PYTHONPATH=/path/to/instant-ngp/build or pass "
            "--instant_ngp_build /path/to/instant-ngp/build."
        ) from exc


def load_reference_transforms(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not metadata.get("frames"):
        raise RuntimeError(f"No frames were found in: {path}")
    return metadata


def offsets_force_center_zero(
    total_baseline: float,
    num_samples: int,
) -> np.ndarray:
    """Create camera offsets with the center sample located exactly at zero."""
    if num_samples == 1:
        return np.zeros(1, dtype=np.float32)
    step = total_baseline / float(num_samples)
    return (
        np.arange(num_samples, dtype=np.float32) - float(num_samples // 2)
    ) * step


def get_look_at_matrix(
    eye: Sequence[float],
    target: Sequence[float],
    up: Sequence[float],
) -> np.ndarray:
    """Construct an OpenGL camera-to-world look-at matrix."""
    eye_array = np.asarray(eye, dtype=np.float32)
    target_array = np.asarray(target, dtype=np.float32)
    up_array = np.asarray(up, dtype=np.float32)

    forward = target_array - eye_array
    forward /= max(float(np.linalg.norm(forward)), 1e-8)
    right = np.cross(forward, up_array)
    right /= max(float(np.linalg.norm(right)), 1e-8)
    corrected_up = np.cross(right, forward)
    corrected_up /= max(float(np.linalg.norm(corrected_up)), 1e-8)

    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 0] = right
    matrix[:3, 1] = corrected_up
    matrix[:3, 2] = -forward
    matrix[:3, 3] = eye_array
    return matrix


def decompose_pose(
    pose: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a camera-to-world matrix into eye, right, up, and forward."""
    eye = pose[:3, 3].astype(np.float32)
    right = pose[:3, 0].astype(np.float32)
    up = pose[:3, 1].astype(np.float32)
    forward = -pose[:3, 2].astype(np.float32)

    right /= max(float(np.linalg.norm(right)), 1e-8)
    up /= max(float(np.linalg.norm(up)), 1e-8)
    forward /= max(float(np.linalg.norm(forward)), 1e-8)
    return eye, right, up, forward


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return np.where(
        values <= 0.0031308,
        12.92 * values,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )


def rgba_to_rgb_u8(
    rgba: np.ndarray,
    exposure: float,
    convert_to_srgb: bool,
) -> np.ndarray:
    """Composite an instant-ngp RGBA render on black and convert it to RGB."""
    rgba = np.asarray(rgba, dtype=np.float32)
    rgb = rgba[..., :3] * rgba[..., 3:4]
    if abs(exposure) > 1e-12:
        rgb *= 2.0 ** exposure
    rgb = np.clip(rgb, 0.0, 1.0)
    if convert_to_srgb:
        rgb = linear_to_srgb(rgb)
    return (rgb * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Stage 1: instant-ngp Snapshot -> Perspective Light Field
# -----------------------------------------------------------------------------
def resolve_fov_deg(args: argparse.Namespace, metadata: dict) -> float:
    """Resolve the horizontal rendering FOV from CLI or transforms.json."""
    if args.fov_deg > 0:
        return float(args.fov_deg)
    if "camera_angle_x" not in metadata:
        raise RuntimeError(
            "--fov_deg is negative, but ref_transforms has no camera_angle_x."
        )
    return float(np.rad2deg(float(metadata["camera_angle_x"])))


def save_lightfield_metadata(
    lightfield_dir: str,
    args: argparse.Namespace,
    fov_deg: float,
    ref_frame: dict,
    ref_pose: np.ndarray,
    base_eye: np.ndarray,
    base_right: np.ndarray,
    base_up: np.ndarray,
    base_forward: np.ndarray,
) -> None:
    """Save light-field intrinsics and reference-camera metadata."""
    intrinsics_path = os.path.join(lightfield_dir, "intrinsics.csv")
    with open(intrinsics_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        rows = [
            ("snapshot", args.snapshot),
            ("ref_transforms", args.ref_transforms),
            ("frame_idx", args.frame_idx),
            ("img_width", args.lf_width),
            ("img_height", args.lf_height),
            ("fov_deg", fov_deg),
            ("nu", args.nu),
            ("nv", args.nv),
            ("total_baseline_x_scene", args.total_baseline_x),
            ("total_baseline_z_scene", args.total_baseline_z),
            ("scene_unit_in_m", args.scene_unit_in_m),
            ("offset_mode", args.offset_mode),
            ("look_at_dist", args.look_at_dist),
            ("spp", args.spp),
            ("exposure", args.exposure),
            ("background_rgba", tuple(args.background_rgba)),
        ]
        writer.writerows(rows)

    reference_path = os.path.join(lightfield_dir, "reference_pose.json")
    with open(reference_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "frame_idx": args.frame_idx,
                "file_path": ref_frame.get("file_path", ""),
                "reference_pose": ref_pose.tolist(),
                "base_eye": base_eye.tolist(),
                "base_right": base_right.tolist(),
                "base_up": base_up.tolist(),
                "base_forward": base_forward.tolist(),
            },
            file,
            indent=4,
        )


def render_lightfield(
    args: argparse.Namespace,
    lightfield_dir: str,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], dict], float]:
    """
    Render and save the perspective light field.

    Returns:
        lightfield: Float RGB array with shape [Nv, Nu, H, W, 3].
        pose_map: Camera offsets in meters, indexed by (u, v).
        fov_deg: Effective horizontal field of view.
    """
    ngp = import_pyngp(args.instant_ngp_build)
    metadata = load_reference_transforms(args.ref_transforms)
    if args.frame_idx >= len(metadata["frames"]):
        raise IndexError(
            f"frame_idx={args.frame_idx} exceeds "
            f"{len(metadata['frames']) - 1}."
        )

    ref_frame = metadata["frames"][args.frame_idx]
    ref_pose = np.asarray(ref_frame.get("transform_matrix"), dtype=np.float32)
    if ref_pose.shape != (4, 4):
        raise RuntimeError(
            f"Reference transform must be 4x4, got {ref_pose.shape}."
        )

    base_eye, base_right, base_up, base_forward = decompose_pose(ref_pose)
    fov_deg = resolve_fov_deg(args, metadata)

    image_dir = os.path.join(lightfield_dir, "images")
    ensure_dir(image_dir)
    save_lightfield_metadata(
        lightfield_dir,
        args,
        fov_deg,
        ref_frame,
        ref_pose,
        base_eye,
        base_right,
        base_up,
        base_forward,
    )

    print(f"[INFO] Loading instant-ngp snapshot: {args.snapshot}")
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.load_snapshot(args.snapshot)
    testbed.shall_train = False
    testbed.dynamic_res = False
    testbed.background_color = np.asarray(
        args.background_rgba,
        dtype=np.float32,
    )
    testbed.fov_axis = 0
    testbed.fov = fov_deg
    testbed.autofocus = False
    testbed.aperture_size = 0.0
    if args.render_near_distance >= 0:
        testbed.render_near_distance = args.render_near_distance

    offsets_x = offsets_force_center_zero(args.total_baseline_x, args.nu)
    offsets_z = offsets_force_center_zero(args.total_baseline_z, args.nv)
    lightfield = np.zeros(
        (args.nv, args.nu, args.lf_height, args.lf_width, 3),
        dtype=np.float32,
    )
    pose_map: Dict[Tuple[int, int], dict] = {}
    poses_path = os.path.join(lightfield_dir, "poses.csv")
    start_time = time.time()

    with open(poses_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "u",
                "v",
                "file",
                "dx",
                "dz",
                "dx_mm",
                "dz_mm",
                "dx_scene",
                "dz_scene",
                "eye_x",
                "eye_y",
                "eye_z",
                "center_x",
                "center_y",
                "center_z",
                "dir_x",
                "dir_y",
                "dir_z",
                "elapsed_s",
                "eta_s",
            ]
        )

        progress = tqdm(
            total=args.nu * args.nv,
            desc="Render NeRF light field",
            dynamic_ncols=True,
        )
        rendered_count = 0
        try:
            for u in range(args.nu):
                for v in range(args.nv):
                    dx_scene = float(offsets_x[u])
                    dz_scene = float(offsets_z[v])
                    if args.offset_mode == "world_xz":
                        offset = np.array(
                            [dx_scene, 0.0, dz_scene],
                            dtype=np.float32,
                        )
                    else:
                        offset = (
                            base_right * dx_scene + base_up * dz_scene
                        )

                    eye = base_eye + offset
                    center = eye + base_forward * args.look_at_dist
                    pose = get_look_at_matrix(eye, center, base_up)
                    testbed.set_nerf_camera_matrix(
                        np.matrix(pose, dtype=np.float32)[:-1, :]
                    )
                    testbed.reset_accumulation(
                        due_to_camera_movement=True,
                        immediate_redraw=True,
                    )

                    rgba = np.asarray(
                        testbed.render(
                            args.lf_width,
                            args.lf_height,
                            args.spp,
                            True,
                        ),
                        dtype=np.float32,
                    )
                    rgb_u8 = rgba_to_rgb_u8(
                        rgba,
                        exposure=args.exposure,
                        convert_to_srgb=args.linear_to_srgb,
                    )
                    lightfield[v, u] = rgb_u8.astype(np.float32) / 255.0

                    filename = f"u{u:02d}_v{v:02d}.png"
                    imageio.imwrite(
                        os.path.join(image_dir, filename),
                        rgb_u8,
                    )
                    if args.save_rgba_npz:
                        np.savez_compressed(
                            os.path.join(
                                image_dir,
                                f"u{u:02d}_v{v:02d}.npz",
                            ),
                            rgba=rgba,
                        )

                    dx_m = dx_scene * args.scene_unit_in_m
                    dz_m = dz_scene * args.scene_unit_in_m
                    pose_map[(u, v)] = {
                        "u": u,
                        "v": v,
                        "file": filename,
                        "dx": dx_m,
                        "dz": dz_m,
                    }

                    rendered_count += 1
                    elapsed = time.time() - start_time
                    eta = elapsed / rendered_count * (
                        args.nu * args.nv - rendered_count
                    )
                    writer.writerow(
                        [
                            u,
                            v,
                            filename,
                            dx_m,
                            dz_m,
                            dx_m * 1000.0,
                            dz_m * 1000.0,
                            dx_scene,
                            dz_scene,
                            *[float(value) for value in eye],
                            *[float(value) for value in center],
                            *[float(value) for value in base_forward],
                            elapsed,
                            eta,
                        ]
                    )
                    progress.update(1)
                    progress.set_postfix(eta=f"{eta / 60.0:.1f}m")
        finally:
            progress.close()

    with open(
        os.path.join(lightfield_dir, "time.csv"),
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "timestamp",
                "nu",
                "nv",
                "width",
                "height",
                "spp",
                "runtime_s",
            ]
        )
        writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                args.nu,
                args.nv,
                args.lf_width,
                args.lf_height,
                args.spp,
                time.time() - start_time,
            ]
        )

    return lightfield, pose_map, fov_deg


# -----------------------------------------------------------------------------
# Stage 2: Perspective Light Field -> Complex Holograms
# -----------------------------------------------------------------------------
def compute_focal_lengths(
    width: int,
    height: int,
    fov_deg: float,
) -> Tuple[float, float]:
    fov_rad = np.deg2rad(fov_deg)
    fx = width * 0.5 / np.tan(fov_rad * 0.5)
    fy = height * 0.5 / np.tan(fov_rad * 0.5)
    return float(fx), float(fy)


def shift_image_subpixel(
    image: np.ndarray,
    shift_x: float,
    shift_y: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = np.array(
        [[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def refocus_lightfield(
    lightfield: np.ndarray,
    pose_map: Dict[Tuple[int, int], dict],
    fov_deg: float,
    d0: float,
) -> np.ndarray:
    """Refocus all views in place to avoid duplicating the full light field."""
    nv, nu, height, width, _ = lightfield.shape
    focal_x, focal_y = compute_focal_lengths(width, height, fov_deg)

    center_key = (nu // 2, nv // 2)
    if center_key not in pose_map:
        center_key = sorted(pose_map)[len(pose_map) // 2]
    center_dx = pose_map[center_key]["dx"]
    center_dz = pose_map[center_key]["dz"]

    for v in tqdm(range(nv), desc="Refocus light field", dynamic_ncols=True):
        for u in range(nu):
            pose = pose_map[(u, v)]
            delta_x = pose["dx"] - center_dx
            delta_z = pose["dz"] - center_dz
            shift_x = -focal_x * delta_x / d0
            shift_y = -focal_y * delta_z / d0
            lightfield[v, u] = shift_image_subpixel(
                lightfield[v, u],
                shift_x,
                shift_y,
            )
    return lightfield


def centered_fft2(values: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(values)))


def lightfield_channel_to_complex(
    lightfield_channel: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Convert one [Nv, Nu, H, W] channel into a tiled complex field."""
    rng = np.random.default_rng(seed)
    nv, nu, height, width = lightfield_channel.shape
    output = np.zeros((nv * height, nu * width), dtype=np.complex64)

    for x in tqdm(
        range(width),
        desc="Light field to complex field",
        dynamic_ncols=True,
    ):
        for y in range(height):
            angular_patch = lightfield_channel[:, :, y, x]
            phase = rng.random((nv, nu), dtype=np.float32) * (2.0 * np.pi)
            complex_patch = angular_patch * np.exp(1j * phase)
            output[
                y * nv:(y + 1) * nv,
                x * nu:(x + 1) * nu,
            ] = centered_fft2(complex_patch).astype(np.complex64)
    return output


def asm_propagate(
    input_field: np.ndarray,
    distance_m: float,
    wavelength_m: float,
    pitch_m: float,
) -> np.ndarray:
    """Propagate a complex field using the angular spectrum method."""
    input_field = np.asarray(input_field, dtype=np.complex64)
    height, width = input_field.shape

    frequency_y = (
        np.arange(height, dtype=np.float32) - height / 2.0
    ) / (height * pitch_m)
    frequency_x = (
        np.arange(width, dtype=np.float32) - width / 2.0
    ) / (width * pitch_m)
    grid_y, grid_x = np.meshgrid(
        frequency_y,
        frequency_x,
        indexing="ij",
    )

    wave_number = 2.0 * np.pi / wavelength_m
    propagating_term = (
        1.0
        - (wavelength_m * grid_y) ** 2
        - (wavelength_m * grid_x) ** 2
    )
    transfer = np.zeros_like(propagating_term, dtype=np.complex64)
    mask = propagating_term >= 0.0
    transfer[mask] = np.exp(
        1j
        * wave_number
        * distance_m
        * np.sqrt(propagating_term[mask])
    ).astype(np.complex64)

    spectrum = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(input_field))
    )
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(spectrum * transfer))
    ).astype(np.complex64)


def save_float_visualization(
    values: np.ndarray,
    path: str,
    p_low: float,
    p_high: float,
) -> None:
    values = np.asarray(values, dtype=np.float32)
    low = np.percentile(values, p_low)
    high = np.percentile(values, p_high)
    normalized = np.clip(
        (values - low) / (high - low + 1e-12),
        0.0,
        1.0,
    )
    imageio.imwrite(path, (normalized * 255.0).astype(np.uint8))


def save_phase_visualization(phase: np.ndarray, path: str) -> None:
    normalized = (
        (np.asarray(phase, dtype=np.float32) + np.pi)
        / (2.0 * np.pi)
    )
    imageio.imwrite(
        path,
        (normalized * 255.0).clip(0, 255).astype(np.uint8),
    )


def generate_holograms(
    centered_lightfield: np.ndarray,
    args: argparse.Namespace,
    hologram_dir: str,
) -> List[np.ndarray]:
    """Generate and save one complex hologram for each RGB channel."""
    holograms: List[np.ndarray] = []
    for channel_index, channel_name in enumerate(CHANNEL_NAMES):
        print(f"[INFO] Generating {channel_name} complex field.")
        complex_field = lightfield_channel_to_complex(
            centered_lightfield[..., channel_index],
            args.random_seed + 1000 * channel_index,
        )
        hologram = asm_propagate(
            complex_field,
            args.dist_rs_to_h,
            args.wavelength,
            args.pitch,
        )
        holograms.append(hologram)

        amplitude = np.abs(hologram).astype(np.float32)
        phase = np.angle(hologram).astype(np.float32)
        np.save(
            os.path.join(
                hologram_dir,
                f"hologram_complex_{channel_name}.npy",
            ),
            hologram,
        )
        np.save(
            os.path.join(
                hologram_dir,
                f"hologram_amp_{channel_name}.npy",
            ),
            amplitude,
        )
        np.save(
            os.path.join(
                hologram_dir,
                f"hologram_phase_{channel_name}.npy",
            ),
            phase,
        )

        if args.save_holo_png:
            save_float_visualization(
                amplitude,
                os.path.join(
                    hologram_dir,
                    f"hologram_amp_{channel_name}.png",
                ),
                args.p_low,
                args.p_high,
            )
            save_phase_visualization(
                phase,
                os.path.join(
                    hologram_dir,
                    f"hologram_phase_{channel_name}.png",
                ),
            )
    return holograms


# -----------------------------------------------------------------------------
# Stage 3: Hologram Reconstruction
# -----------------------------------------------------------------------------
def build_z_list(args: argparse.Namespace) -> List[float]:
    if args.z_list:
        return list(args.z_list)
    if abs(args.z_step) < 1e-12:
        raise ValueError("--z_step cannot be zero.")

    step = args.z_step
    if (args.z_max - args.z_min) * step < 0:
        step = -step
    count = int(
        np.floor((args.z_max - args.z_min) / step + 1e-12)
    ) + 1
    return [args.z_min + index * step for index in range(max(count, 0))]


def normalize_rgb(
    rgb: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    values = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, None)
    if args.vis_mode == "percentile":
        low = np.percentile(values, args.p_low)
        high = np.percentile(values, args.p_high)
        values = np.clip(
            (values - low) / (high - low + 1e-12),
            0.0,
            1.0,
        )
    else:
        values -= values.min()
        maximum = float(values.max())
        if maximum > 1e-8:
            values /= maximum

    if abs(args.gamma - 1.0) > 1e-12:
        values = np.power(values, args.gamma)
    return (values * 255.0).clip(0, 255).astype(np.uint8)


def resize_to_square(
    image: np.ndarray,
    output_size: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    scale = output_size / float(max(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation,
    )
    output = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    x0 = (output_size - new_width) // 2
    y0 = (output_size - new_height) // 2
    output[y0:y0 + new_height, x0:x0 + new_width] = resized
    return output


def reconstruct_holograms(
    holograms: Sequence[np.ndarray],
    args: argparse.Namespace,
    reconstruction_dir: str,
) -> None:
    z_values = build_z_list(args)
    if not z_values:
        raise RuntimeError("The reconstruction distance list is empty.")

    for distance in tqdm(
        z_values,
        desc="Reconstruct RGB hologram",
        dynamic_ncols=True,
    ):
        channels = [
            np.abs(
                asm_propagate(
                    hologram,
                    distance,
                    args.wavelength,
                    args.pitch,
                )
            ).astype(np.float32)
            for hologram in holograms
        ]
        rgb = normalize_rgb(np.stack(channels, axis=-1), args)
        rgb = resize_to_square(rgb, args.out_size)
        if args.save_recon_png:
            imageio.imwrite(
                os.path.join(
                    reconstruction_dir,
                    f"recon_z_{distance:+.6f}.png",
                ),
                rgb,
            )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def save_run_configuration(
    args: argparse.Namespace,
    output_dir: str,
    effective_fov_deg: float,
) -> None:
    configuration = vars(args).copy()
    configuration["effective_fov_deg"] = effective_fov_deg
    with open(
        os.path.join(output_dir, "run_config.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(configuration, file, indent=4)


def main() -> None:
    args = parse_args()
    start_time = time.time()

    lightfield_dir = os.path.join(args.output_dir, "LightField")
    hologram_dir = os.path.join(args.output_dir, "Hologram")
    reconstruction_dir = os.path.join(args.output_dir, "Recon")
    for directory in (
        args.output_dir,
        lightfield_dir,
        hologram_dir,
        reconstruction_dir,
    ):
        ensure_dir(directory)

    lightfield, pose_map, effective_fov_deg = render_lightfield(
        args,
        lightfield_dir,
    )
    save_run_configuration(
        args,
        args.output_dir,
        effective_fov_deg,
    )

    print("[INFO] Refocusing the rendered light field.")
    centered_lightfield = refocus_lightfield(
        lightfield,
        pose_map,
        effective_fov_deg,
        args.d0,
    )

    print("[INFO] Generating complex RGB holograms.")
    holograms = generate_holograms(
        centered_lightfield,
        args,
        hologram_dir,
    )

    print("[INFO] Reconstructing holograms.")
    reconstruct_holograms(
        holograms,
        args,
        reconstruction_dir,
    )

    print(f"[DONE] Results saved to: {args.output_dir}")
    print(f"[INFO] Light field: {lightfield_dir}")
    print(f"[INFO] Holograms: {hologram_dir}")
    if args.save_recon_png:
        print(f"[INFO] Reconstructions: {reconstruction_dir}")
    print(f"[INFO] Total runtime: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
