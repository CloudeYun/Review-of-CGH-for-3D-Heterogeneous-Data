#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colored Point Cloud Sampling from a Mesh

Author: Hao Yun
Date: 2026-03-23

Description:
    This script samples a colored point cloud from a triangle mesh according
    to triangle area distribution. Each sampled point inherits the color of
    its source triangle, optionally modulated by a view-dependent shading term
    for consistency with LDI-style rendering.

Main features:
    - Area-weighted triangle sampling
    - Uniform barycentric sampling inside triangles
    - Material-based triangle color assignment
    - Optional view-dependent shading
    - Export to PLY point cloud format

Example:
    python ./dataset/Obj2PCD.py \
        --input_obj ./dataset/BunnyDragon.obj \
        --output_ply ./dataset/PointCloud/bunnydragon_pointcloud_3e8.ply \
        --num_points 300000000 \
        --seed 2024 \
        --use_shading \
        --eye_pos 0.0 1.5 0.0
"""

import os
import argparse
import numpy as np
import open3d as o3d


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sample a colored point cloud from a triangle mesh."
    )

    parser.add_argument(
        "--input_obj",
        type=str,
        required=True,
        help="Path to the input OBJ mesh."
    )
    parser.add_argument(
        "--output_ply",
        type=str,
        required=True,
        help="Path to the output PLY point cloud."
    )
    parser.add_argument(
        "--num_points",
        type=int,
        required=True,
        help="Number of points to sample from the mesh."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed for reproducible sampling."
    )

    parser.add_argument(
        "--use_shading",
        action="store_true",
        help="Enable view-dependent shading for sampled point colors."
    )
    parser.add_argument(
        "--shading_ambient",
        type=float,
        default=0.30,
        help="Ambient shading coefficient."
    )
    parser.add_argument(
        "--shading_diffuse",
        type=float,
        default=0.70,
        help="Diffuse shading coefficient."
    )
    parser.add_argument(
        "--eye_pos",
        type=float,
        nargs=3,
        default=[0.0, 1.5, 0.0],
        metavar=("X", "Y", "Z"),
        help="Camera position used for view-dependent shading."
    )

    parser.add_argument(
        "--center_mesh",
        action="store_true",
        help="Center the mesh at the origin before sampling."
    )
    parser.add_argument(
        "--write_ascii",
        action="store_true",
        help="Write the output PLY in ASCII format."
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Write the output PLY in compressed format."
    )

    return parser.parse_args()


def get_triangle_colors_from_materials(
    mesh: o3d.geometry.TriangleMesh
) -> np.ndarray:
    """
    Assign per-triangle colors based on material IDs.

    If triangle material IDs are unavailable, a default gray color is used.

    Args:
        mesh: Input triangle mesh.

    Returns:
        A float32 NumPy array of shape (num_triangles, 3), with values in [0, 1].
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

    unique_materials = np.unique(material_ids)
    base_colors = np.array([
        [0.71, 0.53, 0.39],
        [0.82, 0.82, 0.82],
        [0.62, 0.58, 0.50],
        [0.10, 0.10, 0.90],
    ], dtype=np.float32)

    for idx, material_id in enumerate(unique_materials):
        triangle_colors[material_ids == material_id] = base_colors[idx % len(base_colors)]

    unset_mask = (triangle_colors.sum(axis=1) == 0)
    triangle_colors[unset_mask] = default_color
    return triangle_colors


def sample_points_colored_from_mesh(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int,
    seed: int,
    use_shading: bool,
    shading_ambient: float,
    shading_diffuse: float,
    eye_pos: np.ndarray,
):
    """
    Sample colored points from a triangle mesh.

    The sampling procedure is:
        1. Select triangles according to area distribution.
        2. Uniformly sample points inside each selected triangle using barycentric coordinates.
        3. Assign colors based on triangle materials, optionally modulated by view-dependent shading.

    Args:
        mesh: Input triangle mesh.
        num_points: Number of points to sample.
        seed: Random seed.
        use_shading: Whether to apply view-dependent shading.
        shading_ambient: Ambient shading coefficient.
        shading_diffuse: Diffuse shading coefficient.
        eye_pos: Camera position for shading, shape (3,).

    Returns:
        points: Sampled point coordinates of shape (N, 3), float32.
        colors: Sampled point colors of shape (N, 3), float32 in [0, 1].
    """
    rng = np.random.default_rng(seed)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)      # (Nv, 3)
    faces = np.asarray(mesh.triangles, dtype=np.int32)          # (T, 3)
    num_triangles = faces.shape[0]

    if num_triangles == 0:
        raise RuntimeError("The mesh contains no triangles.")

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0
    normals = np.cross(edge1, edge2)  # (T, 3)

    triangle_areas = 0.5 * (np.linalg.norm(normals, axis=1) + 1e-12)
    area_sum = triangle_areas.sum()
    if area_sum <= 0:
        raise RuntimeError("The total triangle area is zero. Cannot sample points.")

    probabilities = triangle_areas / area_sum

    triangle_ids = rng.choice(
        num_triangles,
        size=num_points,
        replace=True,
        p=probabilities
    )

    a = v0[triangle_ids]
    b = v1[triangle_ids]
    c = v2[triangle_ids]

    r1 = rng.random(num_points, dtype=np.float32)
    r2 = rng.random(num_points, dtype=np.float32)
    sqrt_r1 = np.sqrt(r1)

    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2

    points = (
        u[:, None] * a +
        v[:, None] * b +
        w[:, None] * c
    ).astype(np.float32)

    triangle_colors = get_triangle_colors_from_materials(mesh)
    base_colors = triangle_colors[triangle_ids]

    if use_shading:
        sampled_normals = normals[triangle_ids].astype(np.float32)
        sampled_normals /= (np.linalg.norm(sampled_normals, axis=1, keepdims=True) + 1e-12)

        view_dirs = eye_pos[None, :] - points
        view_dirs /= (np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-12)

        cosine = np.abs(np.sum(sampled_normals * view_dirs, axis=1, keepdims=True))
        cosine = np.clip(cosine, 0.0, 1.0)

        shading = shading_ambient + shading_diffuse * cosine
        colors = base_colors * shading
    else:
        colors = base_colors

    colors = np.clip(colors, 0.0, 1.0).astype(np.float32)
    return points, colors


def main():
    args = parse_args()

    if args.num_points <= 0:
        raise ValueError("--num_points must be a positive integer.")

    eye_pos = np.array(args.eye_pos, dtype=np.float32)

    print("=== Colored Point Cloud Sampling ===")
    print(f"Input mesh      : {args.input_obj}")
    print(f"Output pointcloud: {args.output_ply}")
    print(f"Number of points: {args.num_points}")
    print(f"Random seed     : {args.seed}")
    print(f"Use shading     : {args.use_shading}")

    mesh = o3d.io.read_triangle_mesh(args.input_obj, enable_post_processing=False)
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load OBJ mesh: {args.input_obj}")

    if args.center_mesh:
        bbox = mesh.get_axis_aligned_bounding_box()
        mesh.translate(-bbox.get_center())

    mesh.compute_triangle_normals()

    has_material_ids = (
        hasattr(mesh, "triangle_material_ids") and
        len(np.asarray(mesh.triangle_material_ids)) == len(mesh.triangles)
    )

    print(f"[INFO] Number of triangles: {len(mesh.triangles)}")
    print(f"[INFO] Material IDs available: {has_material_ids}")

    points, colors = sample_points_colored_from_mesh(
        mesh=mesh,
        num_points=args.num_points,
        seed=args.seed,
        use_shading=args.use_shading,
        shading_ambient=args.shading_ambient,
        shading_diffuse=args.shading_diffuse,
        eye_pos=eye_pos,
    )

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    output_dir = os.path.dirname(args.output_ply)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    success = o3d.io.write_point_cloud(
        args.output_ply,
        point_cloud,
        write_ascii=args.write_ascii,
        compressed=args.compressed,
    )
    if not success:
        raise RuntimeError("Failed to write the point cloud to disk.")

    colors_np = np.asarray(point_cloud.colors)
    print(f"[DONE] Saved point cloud to: {args.output_ply}")
    print(f"[DONE] Total sampled points: {len(points)}")
    print(
        "[DONE] Color statistics "
        f"(min / max / mean): "
        f"{colors_np.min():.6f} / {colors_np.max():.6f} / {colors_np.mean(axis=0)}"
    )


if __name__ == "__main__":
    main()