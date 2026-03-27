#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orthographic Light Field Rendering from a Colored Mesh

Author: Hao Yun
Date: 2026-03-23

Description:
    This script renders an orthographic light field from a colored triangle mesh
    using Open3D ray casting. The mesh is first uniformly scaled so that its
    thickness along the viewing direction matches a target physical depth range.
    Then, for each angular view, orthographic rays are cast from a reference
    sampling (RS) plane toward the scene to generate a set of multi-view RGB images.

Main features:
    - Orthographic multi-view rendering
    - Material-based triangle colors
    - Double-sided mesh expansion
    - Automatic RS plane placement
    - Optional RS size auto-scaling with aspect-ratio matching
    - Optional 180-degree rotation and image mirroring correction
    - MATLAB-compatible view ordering
    - Export of view images and CSV metadata

Example:
    python ./dataset/Obj2LF_orth.py \
        --input_obj ./dataset/BunnyDragon.obj \
        --output_dir ./dataset/LightField/LF_orth_40x40_800 \
        --nu 40 \
        --nv 40 \
        --img_width 800 \
        --img_height 800 \
        --target_z_min 50.0 \
        --target_z_max 70.0 \
        --eye_y 50.0 \
        --total_baseline_x 4.0 \
        --total_baseline_z 4.0 \
        --auto_rs_plane \
        --auto_rs_size \
        --force_rs_aspect_match \
        --rotate_180 \
        --mirror_lr_image \
        --matlab_compat_order
"""

import os
import csv
import time
import argparse
import numpy as np
import imageio.v2 as imageio
import open3d as o3d
import open3d.core as o3c

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False


INVALID_ID = np.uint32(4294967295)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Render a perspective light field with parallel optical axes."
    )

    parser.add_argument(
        "--input_obj",
        type=str,
        required=True,
        help="Path to the input OBJ mesh."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for rendered images and metadata."
    )

    parser.add_argument(
        "--img_width",
        type=int,
        default=400,
        help="Width of each rendered image."
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=400,
        help="Height of each rendered image."
    )
    parser.add_argument(
        "--fov_deg",
        type=float,
        default=10.0,
        help="Perspective camera field of view in degrees."
    )

    parser.add_argument(
        "--nu",
        type=int,
        default=40,
        help="Number of angular samples along the u direction."
    )
    parser.add_argument(
        "--nv",
        type=int,
        default=40,
        help="Number of angular samples along the v direction."
    )

    parser.add_argument(
        "--total_baseline_x",
        type=float,
        default=4.0,
        help="Total baseline span along X in mm."
    )
    parser.add_argument(
        "--total_baseline_z",
        type=float,
        default=4.0,
        help="Total baseline span along Z in mm."
    )

    parser.add_argument(
        "--bg_color",
        type=int,
        nargs=3,
        default=[0, 0, 0],
        metavar=("R", "G", "B"),
        help="Background RGB color in [0, 255]."
    )

    parser.add_argument(
        "--target_z_min",
        type=float,
        default=45.0,
        help="Near target depth for mesh scaling in mm."
    )
    parser.add_argument(
        "--target_z_max",
        type=float,
        default=50.0,
        help="Far target depth for mesh scaling in mm."
    )

    parser.add_argument(
        "--eye_pos",
        type=float,
        nargs=3,
        default=[0.0, 50.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Base camera position."
    )
    parser.add_argument(
        "--center_pos",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Base look-at center."
    )
    parser.add_argument(
        "--up_dir",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        metavar=("X", "Y", "Z"),
        help="Camera up direction."
    )

    parser.add_argument(
        "--temp_obj_y",
        type=float,
        default=0.0,
        help="Temporary Y translation applied to the mesh after scaling."
    )

    parser.add_argument(
        "--shade_min",
        type=float,
        default=0.3,
        help="Minimum shading coefficient."
    )
    parser.add_argument(
        "--shade_scale",
        type=float,
        default=0.7,
        help="Shading scale coefficient."
    )

    parser.add_argument(
        "--look_at_dist",
        type=float,
        default=200.0,
        help="Look-at distance used to define the center for parallel optical axes."
    )

    return parser.parse_args()


def ensure_dir(path: str):
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def offsets_force_center_zero(total_baseline: float, num_samples: int) -> np.ndarray:
    """
    Generate evenly spaced offsets centered around zero.

    Note:
        The step size is defined as total_baseline / num_samples,
        intentionally matching the user's original implementation
        for strict dataset alignment.
    """
    if num_samples == 1:
        return np.array([0.0], dtype=np.float32)
    step = total_baseline / float(num_samples)
    center_idx = num_samples // 2
    return (np.arange(num_samples, dtype=np.float32) - float(center_idx)) * step


def load_mesh_scaled(
    path: str,
    target_z_min: float,
    target_z_max: float,
    temp_obj_y: float
) -> o3d.geometry.TriangleMesh:
    """
    Load, center, and uniformly scale a mesh so that its thickness along Y
    matches the target depth range.
    """
    print(f"Loading mesh: {path}")
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load OBJ mesh: {path}")

    bbox = mesh.get_axis_aligned_bounding_box()
    mesh.translate(-bbox.get_center())

    target_thickness = target_z_max - target_z_min
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    current_thickness = extent[1]

    scale_factor = target_thickness / (current_thickness + 1e-8)
    mesh.scale(scale_factor, center=(0.0, 0.0, 0.0))
    mesh.translate([0.0, temp_obj_y, 0.0])
    mesh.compute_vertex_normals()

    print("--- Mesh Scaling Information ---")
    print(f"Current thickness along Y: {current_thickness:.6f}")
    print(f"Target thickness along Y : {target_thickness:.6f}")
    print(f"Uniform scale factor     : {scale_factor:.6f}")

    return mesh


def get_triangle_colors_from_materials(
    mesh: o3d.geometry.TriangleMesh
) -> np.ndarray:
    """
    Assign per-triangle colors based on material IDs.

    If material IDs are unavailable, use a default gray color.
    """
    default_color = np.array([0.8, 0.8, 0.8], dtype=np.float32)
    num_triangles = len(mesh.triangles)
    triangle_colors = np.zeros((num_triangles, 3), dtype=np.float32)

    try:
        material_ids = np.asarray(mesh.triangle_material_ids)
    except Exception:
        material_ids = None

    if material_ids is None or material_ids.shape[0] != num_triangles:
        triangle_colors[:] = default_color
        return triangle_colors

    base_colors = np.array([
        [0.71, 0.53, 0.39],
        [0.82, 0.82, 0.82],
        [0.62, 0.58, 0.50],
        [0.10, 0.10, 0.90],
    ], dtype=np.float32)

    for idx, material_id in enumerate(np.unique(material_ids)):
        triangle_colors[material_ids == material_id] = base_colors[idx % len(base_colors)]

    triangle_colors[triangle_colors.sum(axis=1) == 0] = default_color
    return triangle_colors


def expand_to_double_sided(
    mesh_single: o3d.geometry.TriangleMesh,
    triangle_colors_single: np.ndarray
):
    """
    Make the mesh double-sided by duplicating triangles with reversed winding.
    """
    triangles = np.asarray(mesh_single.triangles)
    triangles_flipped = triangles[:, [0, 2, 1]]
    vertices = np.asarray(mesh_single.vertices)

    triangles_new = np.vstack([triangles, triangles_flipped])

    mesh_double = o3d.geometry.TriangleMesh()
    mesh_double.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_double.triangles = o3d.utility.Vector3iVector(triangles_new)
    mesh_double.compute_vertex_normals()

    triangle_colors_double = np.vstack([triangle_colors_single, triangle_colors_single])
    return mesh_double, triangle_colors_double


def build_scene(mesh_double: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    """Build a ray-casting scene from a legacy Open3D mesh."""
    mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh_double)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_tensor)
    return scene


def render_rgb(
    scene: o3d.t.geometry.RaycastingScene,
    triangle_colors: np.ndarray,
    eye: np.ndarray,
    center: np.ndarray,
    up: np.ndarray,
    img_width: int,
    img_height: int,
    fov_deg: float,
    bg_color,
    shade_min: float,
    shade_scale: float,
    out_png: str,
) -> float:
    """
    Render one perspective RGB image and return the hit ratio.
    """
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=fov_deg,
        center=center.tolist(),
        eye=eye.tolist(),
        up=up.tolist(),
        width_px=img_width,
        height_px=img_height,
    )

    rays_np = rays.numpy()
    ray_origins = rays_np[..., 0:3]
    ray_dirs = rays_np[..., 3:6]
    ray_dirs /= (np.linalg.norm(ray_dirs, axis=-1, keepdims=True) + 1e-8)

    rays_tensor = o3c.Tensor(
        np.concatenate([ray_origins, ray_dirs], axis=-1),
        dtype=o3c.Dtype.Float32
    )
    ans = scene.cast_rays(rays_tensor)

    primitive_ids = ans["primitive_ids"].numpy().reshape(img_height, img_width)
    primitive_normals = ans["primitive_normals"].numpy().reshape(img_height, img_width, 3)

    hit_mask = (primitive_ids != INVALID_ID)
    hit_ratio = float(hit_mask.mean())

    rgb = np.zeros((img_height, img_width, 3), dtype=np.float32)
    rgb[~hit_mask] = np.array(bg_color, dtype=np.float32) / 255.0

    if np.any(hit_mask):
        base_colors = triangle_colors[primitive_ids[hit_mask]]
        normals = primitive_normals[hit_mask]
        view_dirs = -ray_dirs[hit_mask]
        view_dirs /= (np.linalg.norm(view_dirs, axis=-1, keepdims=True) + 1e-8)

        shading = shade_min + shade_scale * np.clip(
            np.abs(np.sum(normals * view_dirs, axis=-1)),
            0.0,
            1.0
        )
        rgb[hit_mask] = base_colors * shading[:, None]

    image = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    imageio.imwrite(out_png, image)
    return hit_ratio


def main():
    args = parse_args()

    ensure_dir(args.output_dir)
    image_dir = os.path.join(args.output_dir, "images")
    ensure_dir(image_dir)

    start_time = time.time()

    mesh_single = load_mesh_scaled(
        path=args.input_obj,
        target_z_min=args.target_z_min,
        target_z_max=args.target_z_max,
        temp_obj_y=args.temp_obj_y,
    )
    triangle_colors_single = get_triangle_colors_from_materials(mesh_single)
    mesh_double, triangle_colors_double = expand_to_double_sided(
        mesh_single, triangle_colors_single
    )
    scene = build_scene(mesh_double)

    base_eye = np.array(args.eye_pos, dtype=np.float32)
    base_center = np.array(args.center_pos, dtype=np.float32)
    up = np.array(args.up_dir, dtype=np.float32)

    # Fixed optical axis direction for all cameras
    dir0 = base_center - base_eye
    dir0 = dir0 / (np.linalg.norm(dir0) + 1e-8)

    offsets_x = offsets_force_center_zero(args.total_baseline_x, args.nu)
    offsets_z = offsets_force_center_zero(args.total_baseline_z, args.nv)

    intrinsics_csv = os.path.join(args.output_dir, "intrinsics.csv")
    poses_csv = os.path.join(args.output_dir, "poses.csv")
    time_csv = os.path.join(args.output_dir, "time.csv")

    with open(intrinsics_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input_obj", args.input_obj])
        writer.writerow(["img_width", args.img_width])
        writer.writerow(["img_height", args.img_height])
        writer.writerow(["fov_deg", args.fov_deg])
        writer.writerow(["nu", args.nu])
        writer.writerow(["nv", args.nv])
        writer.writerow(["total_baseline_x_mm", args.total_baseline_x])
        writer.writerow(["total_baseline_z_mm", args.total_baseline_z])
        writer.writerow(["base_eye", base_eye.tolist()])
        writer.writerow(["base_center", base_center.tolist()])
        writer.writerow(["up_dir", up.tolist()])
        writer.writerow(["dir0_unit", dir0.tolist()])
        writer.writerow(["look_at_dist", args.look_at_dist])
        writer.writerow(["temp_obj_y", args.temp_obj_y])
        writer.writerow(["bg_color", str(tuple(args.bg_color))])
        writer.writerow(["shade_min", args.shade_min])
        writer.writerow(["shade_scale", args.shade_scale])

    total_views = args.nu * args.nv
    rendered_count = 0

    if HAS_TQDM:
        pbar = tqdm(
            total=total_views,
            desc="Rendering Parallel LF",
            dynamic_ncols=True,
            mininterval=0.2
        )
    else:
        pbar = None

    with open(poses_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "u", "v", "file", "dx_mm", "dz_mm",
            "eye_x", "eye_y", "eye_z",
            "center_x", "center_y", "center_z",
            "hit_ratio", "elapsed_s", "eta_s"
        ])

        for u in range(args.nu):
            for v in range(args.nv):
                dx = float(offsets_x[u])
                dz = float(offsets_z[v])

                eye = base_eye + np.array([dx, 0.0, dz], dtype=np.float32)
                center = eye + dir0 * float(args.look_at_dist)

                filename = f"u{u:02d}_v{v:02d}.png"
                out_png = os.path.join(image_dir, filename)

                hit_ratio = render_rgb(
                    scene=scene,
                    triangle_colors=triangle_colors_double,
                    eye=eye,
                    center=center,
                    up=up,
                    img_width=args.img_width,
                    img_height=args.img_height,
                    fov_deg=args.fov_deg,
                    bg_color=args.bg_color,
                    shade_min=args.shade_min,
                    shade_scale=args.shade_scale,
                    out_png=out_png,
                )

                rendered_count += 1
                elapsed = time.time() - start_time
                average_time = elapsed / max(rendered_count, 1)
                eta = average_time * (total_views - rendered_count)

                writer.writerow([
                    u, v, filename, dx, dz,
                    float(eye[0]), float(eye[1]), float(eye[2]),
                    float(center[0]), float(center[1]), float(center[2]),
                    hit_ratio, elapsed, eta
                ])

                if pbar is not None:
                    pbar.set_postfix(hit=f"{hit_ratio:.3f}", eta=f"{eta / 60:.1f}m")
                    pbar.update(1)

    if pbar is not None:
        pbar.close()

    end_time = time.time()
    with open(time_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            args.nu,
            args.nv,
            args.img_width,
            args.img_height,
            end_time - start_time
        ])

    print("Done.")
    print(f"Images     : {image_dir}")
    print(f"poses.csv  : {poses_csv}")
    print(f"intrinsics : {intrinsics_csv}")
    print(f"time.csv   : {time_csv}")
    print(f"Running time: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    main()

