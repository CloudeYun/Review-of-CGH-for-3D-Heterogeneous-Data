#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Multi-View Images and Camera Poses from a 3D Mesh

Author: Hao Yun
Date: 2026-06-22

Description:
    This script renders a set of RGB images from a 3D mesh and exports the
    corresponding camera-to-world matrices in a NeRF-compatible
    ``transforms.json`` file.

Pipeline:
    1. Load all valid geometries from an OBJ or other Trimesh-supported file
    2. Normalize the complete scene while preserving relative geometry poses
    3. Optionally mirror one sub-mesh around its local center
    4. Sample camera positions over the upper hemisphere
    5. Render, crop, and save each RGB view
    6. Save camera metadata in NeRF format

Example:
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    python ./dataset/NeRF/gen_multiview.py \
        --mesh_path ./dataset/BunnyDragonRGB.obj \
        --output_dir ./dataset/NeRF/multiview \
        --num_views 120 \
        --resolution 6144 \
        --crop_margin 2048 \
        --mirror_mesh_index auto \
        --seed 0

Notes:
    - Headless rendering uses EGL by default.
    - Set ``EGL_DEVICE_ID`` before execution to select a different EGL device.
    - ``--mirror_mesh_index auto`` selects the most compact sub-mesh.
    - Use ``--mirror_mesh_index none`` to disable mirroring.
"""

import argparse
import json
import math
import os
from typing import List, Optional, Sequence

# These variables must be configured before importing pyrender/PyOpenGL.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_DEVICE_ID", "0")
os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/mesa_shader_cache")

import numpy as np
import pyrender
import trimesh
from PIL import Image


PALETTE_RGBA = [
    [0.62, 0.58, 0.50, 1.0],
    [0.82, 0.82, 0.82, 1.0],
    [0.71, 0.53, 0.39, 1.0],
    [0.10, 0.10, 0.90, 1.0],
]


# -----------------------------------------------------------------------------
# Command-Line Interface
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render multi-view RGB images and NeRF-compatible camera poses "
            "from a 3D mesh."
        )
    )

    parser.add_argument(
        "--mesh_path",
        required=True,
        help="Path to an OBJ or another mesh format supported by Trimesh.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory in which rendered images and transforms.json are saved.",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=120,
        help="Number of rendered camera views.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=6144,
        help="Square rendering resolution before cropping.",
    )
    parser.add_argument(
        "--crop_margin",
        type=int,
        default=2048,
        help="Number of pixels cropped from each image border.",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=0.8,
        help="Maximum normalized scene extent.",
    )
    parser.add_argument(
        "--camera_radius",
        type=float,
        default=3.0,
        help="Distance between each camera and the scene origin.",
    )
    parser.add_argument(
        "--camera_fov_deg",
        type=float,
        default=60.0,
        help="Camera vertical field of view in degrees.",
    )
    parser.add_argument(
        "--max_polar_angle_deg",
        type=float,
        default=float(np.rad2deg(np.pi / 2.2)),
        help="Maximum polar angle used for upper-hemisphere sampling.",
    )
    parser.add_argument(
        "--mirror_mesh_index",
        default="auto",
        help=(
            "Sub-mesh to mirror along its local Y axis: an integer index, "
            "'auto' for the most compact mesh, or 'none' to disable mirroring."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for camera sampling.",
    )
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=30,
        help="Print progress after this many rendered views.",
    )

    args = parser.parse_args()

    if args.num_views <= 0:
        parser.error("--num_views must be positive.")
    if args.resolution <= 0:
        parser.error("--resolution must be positive.")
    if args.crop_margin < 0:
        parser.error("--crop_margin cannot be negative.")
    if 2 * args.crop_margin >= args.resolution:
        parser.error("--crop_margin must be smaller than half of --resolution.")
    if args.scale_factor <= 0:
        parser.error("--scale_factor must be positive.")
    if args.camera_radius <= 0:
        parser.error("--camera_radius must be positive.")
    if not (0.0 < args.camera_fov_deg < 180.0):
        parser.error("--camera_fov_deg must be between 0 and 180 degrees.")
    if not (0.0 < args.max_polar_angle_deg <= 180.0):
        parser.error("--max_polar_angle_deg must be in the interval (0, 180].")
    if args.progress_interval <= 0:
        parser.error("--progress_interval must be positive.")

    return args


# -----------------------------------------------------------------------------
# Data Utilities
# -----------------------------------------------------------------------------
def clean_json_data(value):
    """Replace non-finite values before strict JSON serialization."""
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else 0.0
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, np.ndarray):
        return clean_json_data(value.tolist())
    if isinstance(value, (list, tuple)):
        return [clean_json_data(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_json_data(item) for key, item in value.items()}
    if value is None:
        return 0.0
    return value


def get_look_at_matrix(
    eye: Sequence[float],
    target: Sequence[float],
    up: Sequence[float],
) -> np.ndarray:
    """Construct an OpenGL-style camera-to-world look-at matrix."""
    eye_array = np.asarray(eye, dtype=np.float32)
    target_array = np.asarray(target, dtype=np.float32)
    up_array = np.asarray(up, dtype=np.float32)

    forward = target_array - eye_array
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-8:
        raise ValueError("Camera eye and target positions must be different.")
    forward /= forward_norm

    right = np.cross(forward, up_array)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-8:
        raise ValueError("Camera up vector cannot be parallel to the view direction.")
    right /= right_norm

    corrected_up = np.cross(right, forward)
    corrected_up /= max(float(np.linalg.norm(corrected_up)), 1e-8)

    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 0] = right
    matrix[:3, 1] = corrected_up
    matrix[:3, 2] = -forward
    matrix[:3, 3] = eye_array
    return matrix


# -----------------------------------------------------------------------------
# Mesh Processing
# -----------------------------------------------------------------------------
def load_scene_geometries(mesh_path: str) -> List[trimesh.Trimesh]:
    """Load all valid mesh geometries while preserving their scene transforms."""
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    loaded = trimesh.load(mesh_path, force=None)
    if isinstance(loaded, trimesh.Scene):
        geometries = [
            geometry.copy()
            for geometry in loaded.dump()
            if isinstance(geometry, trimesh.Trimesh)
            and len(geometry.vertices) > 0
            and len(geometry.faces) > 0
        ]
    elif isinstance(loaded, trimesh.Trimesh):
        geometries = [loaded.copy()]
    else:
        raise TypeError(f"Unsupported mesh type: {type(loaded)}")

    if not geometries:
        raise RuntimeError("No valid mesh geometries were found in the input file.")
    return geometries


def print_mesh_info(geometries: Sequence[trimesh.Trimesh]) -> None:
    """Print geometric statistics for each sub-mesh."""
    print("\n[INFO] Sub-mesh statistics")
    for index, geometry in enumerate(geometries):
        vertices = np.asarray(geometry.vertices)
        minimum = vertices.min(axis=0)
        maximum = vertices.max(axis=0)
        center = vertices.mean(axis=0)
        extent = maximum - minimum
        bbox_volume = float(np.prod(extent))
        compact_score = float(np.linalg.norm(extent))
        print(
            f"  [{index}] "
            f"center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}), "
            f"extent=({extent[0]:.4f}, {extent[1]:.4f}, {extent[2]:.4f}), "
            f"bbox_volume={bbox_volume:.6f}, "
            f"compact_score={compact_score:.6f}, "
            f"vertices={len(vertices)}"
        )
    print()


def normalize_scene(
    geometries: Sequence[trimesh.Trimesh],
    scale_factor: float,
) -> None:
    """Center and uniformly scale all geometries as one complete scene."""
    scene_minimum = np.min(
        [np.asarray(geometry.vertices).min(axis=0) for geometry in geometries],
        axis=0,
    )
    scene_maximum = np.max(
        [np.asarray(geometry.vertices).max(axis=0) for geometry in geometries],
        axis=0,
    )

    center = 0.5 * (scene_minimum + scene_maximum)
    maximum_extent = float(np.max(scene_maximum - scene_minimum))
    if maximum_extent <= 1e-8:
        raise ValueError("The input scene has a zero-sized bounding box.")

    for geometry in geometries:
        geometry.vertices = (
            (np.asarray(geometry.vertices) - center)
            / maximum_extent
            * scale_factor
        )


def guess_most_compact_mesh_index(
    geometries: Sequence[trimesh.Trimesh],
) -> int:
    """Select the sub-mesh with the smallest compactness heuristic."""
    scores = []
    for index, geometry in enumerate(geometries):
        vertices = np.asarray(geometry.vertices)
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        score = float(np.prod(extent)) + 0.1 * float(np.linalg.norm(extent))
        scores.append((score, index))
    return min(scores)[1]


def resolve_mirror_mesh_index(
    value: str,
    geometries: Sequence[trimesh.Trimesh],
) -> Optional[int]:
    """Resolve the mirror selection string to a sub-mesh index."""
    normalized = str(value).strip().lower()
    if normalized in {"none", "off", "disable", "-1"}:
        return None
    if normalized == "auto":
        return guess_most_compact_mesh_index(geometries)

    try:
        index = int(normalized)
    except ValueError as exc:
        raise ValueError(
            "--mirror_mesh_index must be an integer, 'auto', or 'none'."
        ) from exc

    if not 0 <= index < len(geometries):
        raise ValueError(
            f"Mirror mesh index {index} is invalid for "
            f"{len(geometries)} geometries."
        )
    return index


def mirror_mesh_along_local_y(mesh: trimesh.Trimesh) -> None:
    """Mirror a mesh along its local Y axis around its bounding-box center."""
    center = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)

    translate_to_origin = np.eye(4, dtype=np.float64)
    translate_to_origin[:3, 3] = -center

    mirror_y = np.eye(4, dtype=np.float64)
    mirror_y[1, 1] = -1.0

    translate_back = np.eye(4, dtype=np.float64)
    translate_back[:3, 3] = center

    mesh.apply_transform(translate_back @ mirror_y @ translate_to_origin)

    # Reflection reverses triangle winding, so restore outward-facing normals.
    mesh.invert()


# -----------------------------------------------------------------------------
# Scene and Camera Construction
# -----------------------------------------------------------------------------
def build_render_scene(
    geometries: Sequence[trimesh.Trimesh],
    target: np.ndarray,
) -> pyrender.Scene:
    """Create a Pyrender scene with materials and directional lighting."""
    scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 0.0],
        ambient_light=[0.25, 0.25, 0.25],
    )

    for index, geometry in enumerate(geometries):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.05,
            roughnessFactor=0.9,
            baseColorFactor=tuple(PALETTE_RGBA[index % len(PALETTE_RGBA)]),
        )
        scene.add(pyrender.Mesh.from_trimesh(geometry, material=material))

    light_specs = [
        ([2.0, 2.0, 3.0], 2.5),
        ([-2.0, 1.0, 2.0], 1.0),
        ([0.0, -2.0, -2.0], 0.5),
    ]
    light_up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    for position, intensity in light_specs:
        scene.add(
            pyrender.DirectionalLight(
                color=[1.0, 1.0, 1.0],
                intensity=intensity,
            ),
            pose=get_look_at_matrix(position, target, light_up),
        )

    return scene


def sample_camera_pose(
    view_index: int,
    radius: float,
    max_polar_angle_rad: float,
    rng: np.random.Generator,
    target: np.ndarray,
) -> np.ndarray:
    """Sample one camera pose while retaining a deterministic first view."""
    if view_index == 0:
        eye = np.array([0.0, radius, radius * 0.03], dtype=np.float32)
        up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return get_look_at_matrix(eye, target, up)

    theta = rng.uniform(0.0, max_polar_angle_rad)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    eye = np.array(
        [
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta),
        ],
        dtype=np.float32,
    )

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    view_direction = target - eye
    view_direction /= max(float(np.linalg.norm(view_direction)), 1e-8)
    if abs(float(np.dot(view_direction, up))) > 0.99:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    return get_look_at_matrix(eye, target, up)


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------
def render_multiview_dataset(args: argparse.Namespace) -> None:
    """Render all views and save images plus NeRF camera metadata."""
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Loading mesh: {args.mesh_path}")
    geometries = load_scene_geometries(args.mesh_path)
    print_mesh_info(geometries)
    normalize_scene(geometries, args.scale_factor)

    mirror_index = resolve_mirror_mesh_index(
        args.mirror_mesh_index,
        geometries,
    )
    if mirror_index is not None:
        print(f"[INFO] Mirroring sub-mesh {mirror_index} along its local Y axis.")
        mirror_mesh_along_local_y(geometries[mirror_index])
    else:
        print("[INFO] Sub-mesh mirroring is disabled.")

    target = np.zeros(3, dtype=np.float32)
    scene = build_render_scene(geometries, target)
    camera_fov_rad = float(np.deg2rad(args.camera_fov_deg))
    camera = pyrender.PerspectiveCamera(
        yfov=camera_fov_rad,
        aspectRatio=1.0,
    )
    renderer = pyrender.OffscreenRenderer(
        viewport_width=args.resolution,
        viewport_height=args.resolution,
    )
    rng = np.random.default_rng(args.seed)

    transforms = {
        "camera_angle_x": camera_fov_rad,
        "aabb_scale": 4,
        "frames": [],
    }

    print(f"[INFO] Rendering {args.num_views} views.")
    try:
        for view_index in range(args.num_views):
            pose = sample_camera_pose(
                view_index=view_index,
                radius=args.camera_radius,
                max_polar_angle_rad=float(
                    np.deg2rad(args.max_polar_angle_deg)
                ),
                rng=rng,
                target=target,
            )

            camera_node = scene.add(camera, pose=pose)
            try:
                color, _ = renderer.render(scene)
            finally:
                scene.remove_node(camera_node)

            if args.crop_margin > 0:
                color = color[
                    args.crop_margin:-args.crop_margin,
                    args.crop_margin:-args.crop_margin,
                ]

            filename = f"r_{view_index:03d}.png"
            output_path = os.path.join(args.output_dir, filename)
            Image.fromarray(color).convert("RGB").save(output_path)

            transforms["frames"].append(
                {
                    "file_path": filename,
                    "transform_matrix": pose.tolist(),
                }
            )

            completed = view_index + 1
            if (
                completed == 1
                or completed % args.progress_interval == 0
                or completed == args.num_views
            ):
                print(f"[INFO] Rendered {completed}/{args.num_views} views.")
    finally:
        renderer.delete()

    transforms_path = os.path.join(args.output_dir, "transforms.json")
    with open(transforms_path, "w", encoding="utf-8") as file:
        json.dump(
            clean_json_data(transforms),
            file,
            indent=4,
            allow_nan=False,
        )

    print(f"[DONE] Images and camera poses saved to: {args.output_dir}")


def main() -> None:
    args = parse_args()
    render_multiview_dataset(args)


if __name__ == "__main__":
    main()
