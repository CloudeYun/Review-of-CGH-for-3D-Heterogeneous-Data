#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDI Dataset Generation from a Colored Mesh via Multi-Layer Ray Casting

Author: Hao Yun
Date: 2026-03-23

Description:
    This script generates layered RGB images and depth maps from a 3D mesh
    using Open3D ray casting. The mesh is uniformly scaled so that its
    thickness along the viewing direction matches a target physical depth
    range, then rendered into multiple LDI layers.

Main outputs:
    - rgb_layer{idx}.png
    - depth_layer{idx}.npy
    - depth_layer{idx}.png

Example:
    python ./dataset/Obj2Layer.py \
        --input_obj ./dataset/BunnyDragon.obj \
        --output_dir ./dataset/Layer/BunnyDragonLDI \
        --img_width 2048 \
        --img_height 2048 \
        --fov_deg 1.0 \
        --ldi_layers 1 \
        --target_z_min 50.0 \
        --target_z_max 53.0 \
        --bg_depth 100.0

You can change ldi_layers to express different types of multilayers data:
    -- 1 : RGBD
    -- >1 : LDI
"""

import os
import argparse
import numpy as np
import imageio.v2 as imageio
import open3d as o3d
import open3d.core as o3c


INVALID_ID = np.uint32(4294967295)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate LDI RGB/depth layers from a colored mesh."
    )

    # Required arguments
    parser.add_argument(
        "--input_obj",
        type=str,
        required=True,
        help="Path to the input OBJ mesh file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for saving generated RGB and depth layers."
    )

    # Image and rendering settings
    parser.add_argument(
        "--img_width",
        type=int,
        default=2048,
        help="Output image width in pixels."
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=2048,
        help="Output image height in pixels."
    )
    parser.add_argument(
        "--fov_deg",
        type=float,
        default=1.0,
        help="Camera field of view in degrees."
    )
    parser.add_argument(
        "--ldi_layers",
        type=int,
        default=10,
        help="Number of LDI layers to generate."
    )
    parser.add_argument(
        "--eps_shift",
        type=float,
        default=1e-4,
        help="Small ray origin offset to avoid self-intersection between layers."
    )

    # Physical depth settings
    parser.add_argument(
        "--target_z_min",
        type=float,
        default=50.0,
        help="Nearest target depth in mm."
    )
    parser.add_argument(
        "--target_z_max",
        type=float,
        default=53.0,
        help="Farthest target depth in mm."
    )
    parser.add_argument(
        "--bg_depth",
        type=float,
        default=100.0,
        help="Background depth value in mm."
    )

    # Camera settings
    parser.add_argument(
        "--eye_pos",
        type=float,
        nargs=3,
        default=[0.0, 1.5, 0.0],
        metavar=("X", "Y", "Z"),
        help="Camera position."
    )
    parser.add_argument(
        "--center_pos",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Camera look-at center."
    )
    parser.add_argument(
        "--up_dir",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        metavar=("X", "Y", "Z"),
        help="Camera up direction."
    )

    # Temporary mesh placement
    parser.add_argument(
        "--temp_obj_y",
        type=float,
        default=-200.0,
        help="Temporary Y translation applied to the mesh before rendering."
    )

    # Background color
    parser.add_argument(
        "--bg_color",
        type=int,
        nargs=3,
        default=[0, 0, 0],
        metavar=("R", "G", "B"),
        help="Background RGB color in [0, 255]."
    )

    return parser.parse_args()


def load_mesh_scaled(
    path: str,
    target_z_min: float,
    target_z_max: float,
    temp_obj_y: float
) -> o3d.geometry.TriangleMesh:
    """
    Load a mesh, center it at the origin, uniformly scale it so that its
    thickness along the Y axis matches the target depth range, and then
    translate it to a temporary location for ray casting.
    """
    print(f"Loading mesh from: {path}")
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load OBJ mesh: {path}")

    # Center the mesh at the origin
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    mesh.translate(-center)

    # Match the thickness along Y to the target physical range
    target_thickness = float(target_z_max - target_z_min)

    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()  # [dx, dy, dz]
    current_thickness = float(extent[1])

    scale_factor = target_thickness / (current_thickness + 1e-8)

    print("\n--- Mesh Scaling Information ---")
    print(f"Original thickness along Y: {current_thickness:.6f}")
    print(f"Target physical thickness : {target_thickness:.6f} mm")
    print(f"Uniform scale factor      : {scale_factor:.6f}")

    mesh.scale(scale_factor, center=(0.0, 0.0, 0.0))

    # Translate mesh to a temporary position in front of the camera
    mesh.translate([0.0, float(temp_obj_y), 0.0])
    mesh.compute_vertex_normals()
    return mesh


def get_triangle_colors_from_materials(
    mesh: o3d.geometry.TriangleMesh
) -> np.ndarray:
    """
    Assign per-triangle base colors according to material IDs.

    If triangle_material_ids are unavailable, a default gray color is used.
    """
    default_color = np.array([0.3, 0.3, 0.3], dtype=np.float32)
    num_triangles = len(mesh.triangles)
    tri_colors = np.zeros((num_triangles, 3), dtype=np.float32)

    try:
        material_ids = np.asarray(mesh.triangle_material_ids)
    except Exception:
        material_ids = None

    if material_ids is None or material_ids.shape[0] != num_triangles:
        tri_colors[:] = default_color
        return tri_colors

    unique_materials = np.unique(material_ids)

    base_colors = np.array([
        [0.71, 0.53, 0.39],
        [0.82, 0.82, 0.82],
        [0.62, 0.58, 0.50],
        [0.10, 0.10, 0.90],
    ], dtype=np.float32)

    for i, material_id in enumerate(unique_materials):
        tri_colors[material_ids == material_id] = base_colors[i % len(base_colors)]

    unset_mask = (tri_colors.sum(axis=1) == 0)
    tri_colors[unset_mask] = default_color
    return tri_colors


def expand_to_double_sided(
    mesh_single: o3d.geometry.TriangleMesh,
    tri_color_single: np.ndarray
):
    """
    Convert a single-sided mesh into a double-sided mesh by duplicating
    triangles with reversed winding order.
    """
    triangles = np.asarray(mesh_single.triangles)
    triangles_flipped = triangles[:, [0, 2, 1]]
    vertices = np.asarray(mesh_single.vertices)

    triangles_new = np.vstack([triangles, triangles_flipped])

    mesh_double = o3d.geometry.TriangleMesh()
    mesh_double.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_double.triangles = o3d.utility.Vector3iVector(triangles_new)
    mesh_double.compute_vertex_normals()

    tri_color_double = np.vstack([tri_color_single, tri_color_single])
    return mesh_double, tri_color_double


def build_scene(mesh_double: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    """Build an Open3D ray casting scene from a legacy triangle mesh."""
    mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh_double)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_tensor)
    return scene


def save_rgb_with_shading(
    prim_ids: np.ndarray,
    prim_normals: np.ndarray,
    ray_dirs: np.ndarray,
    tri_colors: np.ndarray,
    out_png: str,
    bg_color
):
    """
    Save an RGB image with simple view-dependent Lambert-like shading.
    """
    height, width = prim_ids.shape
    hit_mask = (prim_ids != INVALID_ID)

    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[~hit_mask] = np.array(bg_color, dtype=np.float32) / 255.0

    if np.any(hit_mask):
        base_color = tri_colors[prim_ids[hit_mask].astype(np.int64)]
        normals = prim_normals[hit_mask].astype(np.float32)
        view_dirs = (-ray_dirs[hit_mask]).astype(np.float32)
        view_dirs /= (np.linalg.norm(view_dirs, axis=-1, keepdims=True) + 1e-8)

        cosine = np.abs(np.sum(normals * view_dirs, axis=-1))
        cosine = np.clip(cosine, 0.0, 1.0)
        shading = 0.3 + 0.7 * cosine
        rgb[hit_mask] = base_color * shading[:, None]

    rgb_u8 = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    imageio.imwrite(out_png, rgb_u8)


def save_shifted_depth(
    raw_depth_layer: np.ndarray,
    shift_val: float,
    layer_idx: int,
    out_dir: str,
    target_z_min: float,
    bg_depth: float
):
    """
    Save shifted depth as:
        1) .npy file in float32 (mm)
        2) .png visualization where near=dark and far=bright
    """
    height, width = raw_depth_layer.shape
    valid_mask = np.isfinite(raw_depth_layer)

    final_depth = np.full((height, width), float(bg_depth), dtype=np.float32)
    if np.any(valid_mask):
        final_depth[valid_mask] = raw_depth_layer[valid_mask].astype(np.float32) + float(shift_val)

    npy_path = os.path.join(out_dir, f"depth_layer{layer_idx}.npy")
    np.save(npy_path, final_depth)

    vis_min = float(target_z_min)
    vis_max = float(bg_depth)

    depth_vis = np.ones((height, width), dtype=np.float32)
    if np.any(valid_mask):
        depth_norm = (final_depth[valid_mask] - vis_min) / (vis_max - vis_min + 1e-8)
        depth_vis[valid_mask] = np.clip(depth_norm, 0.0, 1.0)

    depth_u8 = (depth_vis * 255.0).clip(0, 255).astype(np.uint8)
    png_path = os.path.join(out_dir, f"depth_layer{layer_idx}.png")
    imageio.imwrite(png_path, depth_u8)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== LDI Dataset Generation ===")
    print(f"Input mesh : {args.input_obj}")
    print(f"Output dir : {args.output_dir}")
    print(f"Image size : {args.img_width} x {args.img_height}")
    print(f"LDI layers : {args.ldi_layers}")
    print(f"Depth range: [{args.target_z_min}, {args.target_z_max}] mm")

    # Step 1: Load, scale, and translate the mesh
    mesh_single = load_mesh_scaled(
        path=args.input_obj,
        target_z_min=args.target_z_min,
        target_z_max=args.target_z_max,
        temp_obj_y=args.temp_obj_y,
    )

    tri_colors_single = get_triangle_colors_from_materials(mesh_single)
    mesh_double, tri_colors_double = expand_to_double_sided(mesh_single, tri_colors_single)
    scene = build_scene(mesh_double)

    # Step 2: Create camera rays
    eye = np.array(args.eye_pos, dtype=np.float32)
    center = np.array(args.center_pos, dtype=np.float32)
    up = np.array(args.up_dir, dtype=np.float32)

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=float(args.fov_deg),
        center=center.tolist(),
        eye=eye.tolist(),
        up=up.tolist(),
        width_px=int(args.img_width),
        height_px=int(args.img_height),
    )

    rays_np = rays.numpy()
    ray_origins = rays_np[..., 0:3].astype(np.float32)
    ray_dirs = rays_np[..., 3:6].astype(np.float32)
    ray_dirs /= (np.linalg.norm(ray_dirs, axis=-1, keepdims=True) + 1e-8)

    # Step 3: Multi-layer ray casting
    print(f"\nPhase 1: Ray Casting {args.ldi_layers} Layers...")
    raw_depth_layers = []

    current_origins = ray_origins.copy()
    t_prev = np.full((args.img_height, args.img_width), np.nan, dtype=np.float32)
    hits_prev = np.zeros((args.img_height, args.img_width, 3), dtype=np.float32)

    for layer_idx in range(args.ldi_layers):
        if layer_idx > 0:
            valid_prev = np.isfinite(t_prev)
            if np.any(valid_prev):
                current_origins[valid_prev] = (
                    hits_prev[valid_prev] + float(args.eps_shift) * ray_dirs[valid_prev]
                )

        rays_tensor = o3c.Tensor(
            np.concatenate([current_origins, ray_dirs], axis=-1),
            dtype=o3c.Dtype.Float32
        )
        ans = scene.cast_rays(rays_tensor)

        t_hit = ans["t_hit"].numpy().reshape(args.img_height, args.img_width).astype(np.float32)
        prim_ids = ans["primitive_ids"].numpy().reshape(args.img_height, args.img_width).astype(np.uint32)
        prim_normals = ans["primitive_normals"].numpy().reshape(args.img_height, args.img_width, 3).astype(np.float32)

        valid = np.isfinite(t_hit) & (prim_ids != INVALID_ID)
        hits = current_origins + ray_dirs * t_hit[..., None]

        depth_abs = np.full((args.img_height, args.img_width), np.nan, dtype=np.float32)
        if np.any(valid):
            diff = hits - eye.reshape(1, 1, 3)
            dist = np.linalg.norm(diff, axis=-1)
            depth_abs[valid] = dist[valid]

        raw_depth_layers.append(depth_abs)

        rgb_path = os.path.join(args.output_dir, f"rgb_layer{layer_idx + 1}.png")
        save_rgb_with_shading(
            prim_ids=prim_ids,
            prim_normals=prim_normals,
            ray_dirs=ray_dirs,
            tri_colors=tri_colors_double,
            out_png=rgb_path,
            bg_color=args.bg_color,
        )

        t_prev = t_hit
        hits_prev = hits

    # Step 4: Shift depths into the target physical range
    print("\nPhase 2: Applying Global Depth Shift...")
    all_depths = np.array(raw_depth_layers, dtype=np.float32)

    if np.all(np.isnan(all_depths)):
        print("Error: no valid intersections found. Please check camera settings or mesh position.")
        return

    raw_min = float(np.nanmin(all_depths))
    raw_max = float(np.nanmax(all_depths))

    shift_val = float(args.target_z_min - raw_min)
    new_min = raw_min + shift_val
    new_max = raw_max + shift_val

    print(f"Original rendered depth range: [{raw_min:.6f}, {raw_max:.6f}] mm")
    print(f"Applied depth shift         : {shift_val:+.6f} mm")
    print(f"Final shifted depth range   : [{new_min:.6f}, {new_max:.6f}] mm")
    print(f"Background depth            : {args.bg_depth:.6f} mm")

    for layer_idx, raw_layer in enumerate(raw_depth_layers, start=1):
        save_shifted_depth(
            raw_depth_layer=raw_layer,
            shift_val=shift_val,
            layer_idx=layer_idx,
            out_dir=args.output_dir,
            target_z_min=args.target_z_min,
            bg_depth=args.bg_depth,
        )

    print(f"\nDone. Results have been saved to: {args.output_dir}")


if __name__ == "__main__":
    main()