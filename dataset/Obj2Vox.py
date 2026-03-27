#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert OBJ Mesh to Voxel Grid with Optional Post-processing and Visualization

Author: Hao Yun
Date: 2026-03-25

Description:
    This script voxelizes an OBJ mesh and saves the result as either:
        - .npz: containing occupancy grid, voxel pitch, and grid origin
        - .npy: containing only the occupancy grid

    It supports:
        1. Robust OBJ loading for both Trimesh and Scene inputs
        2. True mesh-space rotation before voxelization
        3. Optional solid voxel filling
        4. Optional voxel-domain post-processing (gap closing + hole filling)
        5. Optional visualization via slice preview, 3D scatter, or isosurface

Saved array convention:
    - occ_zyx: occupancy grid in (Z, Y, X) order
    - origin_xyz: physical origin of the voxel grid in mesh coordinates
    - pitch: voxel size in the same unit as the input mesh coordinates

Example:
    python ./dataset/Obj2Vox.py \
        --obj ./dataset/BunnyDragon.obj \
        --out ./dataset/Voxel/BunnyDragon_voxel.npz \
        --pitch 0.05 \
        --solid --post_fill --close_iter 3 \
        --rot_x 155 --rot_about center \
        --vis --vis_dir ./Voxel/vis \
        --vis_mode isosurface \
        --vis_iso_sigma 0.15 \
        --vis_iso_level 0.33 \
        --vis_iso_max_dim 800 \
        --only_view --no_axes --view_elev 20 --view_azim -90

Hint:
    You can modify the scale of the voxel by adjusting the value of pitch (0.01 for example)
"""

import os
import argparse
import numpy as np
import trimesh


# -----------------------------------------------------------------------------
# Mesh Loading
# -----------------------------------------------------------------------------
def load_mesh_any(obj_path: str) -> trimesh.Trimesh:
    """
    Load an OBJ file as either a single mesh or a scene.

    If the OBJ is loaded as a Scene, all valid sub-meshes are concatenated into
    one mesh.

    Args:
        obj_path: Path to the OBJ file.

    Returns:
        A trimesh.Trimesh object.
    """
    loaded = trimesh.load(obj_path, force=None)

    if isinstance(loaded, trimesh.Scene):
        geometries = []
        for _, geom in loaded.geometry.items():
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0 and len(geom.faces) > 0:
                geometries.append(geom)

        if len(geometries) == 0:
            raise RuntimeError("OBJ was loaded as a Scene, but no valid mesh geometries were found.")

        mesh = trimesh.util.concatenate(geometries)
        print(f"[INFO] Loaded Scene with {len(geometries)} geometries and concatenated them into one mesh.")
        return mesh

    if isinstance(loaded, trimesh.Trimesh):
        print("[INFO] Loaded OBJ as a single Trimesh.")
        return loaded

    raise RuntimeError(f"Unsupported loaded type: {type(loaded)}")


# -----------------------------------------------------------------------------
# Mesh Rotation
# -----------------------------------------------------------------------------
def rotate_mesh(
    mesh: trimesh.Trimesh,
    rot_x_deg: float = 0.0,
    rot_y_deg: float = 0.0,
    rot_z_deg: float = 0.0,
    about: str = "center"
) -> trimesh.Trimesh:
    """
    Rotate the mesh in 3D space.

    Rotation order:
        X first, then Y, then Z
        Equivalent matrix composition: R = Rz @ Ry @ Rx

    Args:
        mesh: Input mesh.
        rot_x_deg: Rotation angle around the X axis in degrees.
        rot_y_deg: Rotation angle around the Y axis in degrees.
        rot_z_deg: Rotation angle around the Z axis in degrees.
        about: Rotation center. Either:
            - "center": rotate about the mesh bounding-box center
            - "origin": rotate about the global origin

    Returns:
        Rotated mesh.
    """
    from trimesh.transformations import rotation_matrix

    rot_x = rotation_matrix(np.deg2rad(rot_x_deg), [1, 0, 0])
    rot_y = rotation_matrix(np.deg2rad(rot_y_deg), [0, 1, 0])
    rot_z = rotation_matrix(np.deg2rad(rot_z_deg), [0, 0, 1])
    rotation = rot_z @ rot_y @ rot_x

    if about == "center":
        center = mesh.bounds.mean(axis=0)
        translate_to_origin = np.eye(4)
        translate_to_origin[:3, 3] = -center
        translate_back = np.eye(4)
        translate_back[:3, 3] = center
        transform = translate_back @ rotation @ translate_to_origin
    else:
        transform = rotation

    rotated = mesh.copy()
    rotated.apply_transform(transform)
    return rotated


# -----------------------------------------------------------------------------
# Voxel-domain Post-processing
# -----------------------------------------------------------------------------
def voxel_postprocess_fill(
    occ_zyx: np.ndarray,
    do_fill_3d: bool = True,
    do_fill_2d: bool = True,
    close_iter: int = 1
) -> np.ndarray:
    """
    Apply voxel-domain post-processing:
        - binary closing to bridge small gaps
        - 3D hole filling
        - 2D slice-wise hole filling

    Args:
        occ_zyx: Occupancy grid in (Z, Y, X) order.
        do_fill_3d: Whether to apply 3D hole filling.
        do_fill_2d: Whether to apply 2D hole filling slice-by-slice.
        close_iter: Number of binary-closing iterations.

    Returns:
        Post-processed occupancy grid as uint8.
    """
    from scipy.ndimage import binary_fill_holes, binary_closing

    occ = occ_zyx.astype(bool)

    if close_iter > 0:
        occ = binary_closing(occ, iterations=int(close_iter))

    if do_fill_3d:
        occ = binary_fill_holes(occ)

    if do_fill_2d:
        for z_idx in range(occ.shape[0]):
            occ[z_idx] = binary_fill_holes(occ[z_idx])

    return occ.astype(np.uint8)


# -----------------------------------------------------------------------------
# Voxelization
# -----------------------------------------------------------------------------
def voxelize_mesh(
    mesh: trimesh.Trimesh,
    pitch: float,
    solid: bool,
    post_fill: bool,
    close_iter: int
):
    """
    Voxelize a mesh.

    Args:
        mesh: Input mesh.
        pitch: Voxel size.
        solid: Whether to fill the voxel grid as a solid object.
        post_fill: Whether to apply voxel-domain post-processing.
        close_iter: Number of closing iterations used in post-processing.

    Returns:
        occ_zyx: Occupancy grid in (Z, Y, X) order.
        origin_xyz: Grid origin in mesh coordinates.
    """
    voxel_grid = mesh.voxelized(pitch=pitch)
    matrix_xyz = voxel_grid.matrix.astype(np.uint8)  # trimesh order: (X, Y, Z)

    if solid:
        try:
            matrix_xyz = voxel_grid.fill().matrix.astype(np.uint8)
        except Exception as exc:
            print(f"[WARN] voxel_grid.fill() failed. Falling back to voxel post-processing only. Error: {exc}")

    occ_zyx = np.transpose(matrix_xyz, (2, 1, 0))  # -> (Z, Y, X)
    origin_xyz = np.array(voxel_grid.origin, dtype=np.float32)

    if solid and post_fill:
        try:
            num_before = int(occ_zyx.sum())
            occ_zyx = voxel_postprocess_fill(
                occ_zyx,
                do_fill_3d=True,
                do_fill_2d=True,
                close_iter=close_iter
            )
            num_after = int(occ_zyx.sum())
            print(f"[INFO] Voxel post-processing enabled: close_iter={close_iter}, occupied voxels {num_before} -> {num_after}")
        except Exception as exc:
            print(f"[WARN] Voxel post-processing failed (SciPy may be missing): {exc}")

    return occ_zyx, origin_xyz


def save_outputs(out_path: str, occ_zyx: np.ndarray, pitch: float, origin_xyz: np.ndarray):
    """
    Save voxelization results.

    If the output ends with .npz, save:
        - occ
        - pitch
        - origin_xyz

    Otherwise save only the occupancy grid as .npy.

    Args:
        out_path: Output file path.
        occ_zyx: Occupancy grid in (Z, Y, X) order.
        pitch: Voxel size.
        origin_xyz: Grid origin in mesh coordinates.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if out_path.endswith(".npz"):
        np.savez_compressed(
            out_path,
            occ=occ_zyx.astype(np.uint8),
            pitch=np.float32(pitch),
            origin_xyz=origin_xyz.astype(np.float32)
        )
    else:
        np.save(out_path, occ_zyx.astype(np.uint8))


# -----------------------------------------------------------------------------
# Visualization Helpers
# -----------------------------------------------------------------------------
def _set_black_background(fig, ax):
    """
    Set figure and axis background to black, and hide 3D pane edges when possible.
    """
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


def visualize_slices(occ_zyx: np.ndarray, out_dir: str = None, num_slices: int = 9):
    """
    Visualize several Z slices from the voxel grid.
    """
    import matplotlib.pyplot as plt

    nz, ny, nx = occ_zyx.shape
    if nz <= 0:
        print("[WARN] nz <= 0. Cannot visualize slices.")
        return

    if num_slices <= 1:
        slice_indices = [nz // 2]
    else:
        slice_indices = np.linspace(0, nz - 1, num_slices).round().astype(int).tolist()

    num_cols = int(np.ceil(np.sqrt(len(slice_indices))))
    num_rows = int(np.ceil(len(slice_indices) / num_cols))

    plt.figure(figsize=(3.2 * num_cols, 3.2 * num_rows), dpi=160)
    for i, z_idx in enumerate(slice_indices, start=1):
        plt.subplot(num_rows, num_cols, i)
        plt.imshow(occ_zyx[z_idx], cmap="gray", vmin=0, vmax=1)
        plt.title(f"z={z_idx}/{nz - 1}")
        plt.axis("off")

    plt.tight_layout()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        out_png = os.path.join(out_dir, "voxel_slices.png")
        plt.savefig(out_png, bbox_inches="tight")
        print(f"[VIS] Saved slice preview to: {out_png}")

    plt.show()


def visualize_3d_scatter(
    occ_zyx: np.ndarray,
    pitch: float,
    origin_xyz: np.ndarray,
    out_dir: str = None,
    max_points: int = 600000,
    view_elev: float = 20.0,
    view_azim: float = -90.0,
    no_axes: bool = True
):
    """
    Visualize occupied voxels as a 3D scatter plot on a black background.
    """
    import matplotlib.pyplot as plt

    indices = np.argwhere(occ_zyx > 0)  # (N, 3) in (Z, Y, X)
    num_points = indices.shape[0]
    print(f"[VIS] Occupied voxels: {num_points}")

    if num_points == 0:
        print("[WARN] No occupied voxels to visualize.")
        return

    if num_points > max_points:
        selected = np.random.choice(num_points, size=max_points, replace=False)
        indices = indices[selected]
        num_points = max_points
        print(f"[VIS] Downsampled to {num_points} points for visualization.")

    z = indices[:, 0].astype(np.float32)
    y = indices[:, 1].astype(np.float32)
    x = indices[:, 2].astype(np.float32)

    x_world = origin_xyz[0] + (x + 0.5) * pitch
    y_world = origin_xyz[1] + (y + 0.5) * pitch
    z_world = origin_xyz[2] + (z + 0.5) * pitch

    fig = plt.figure(figsize=(8, 7), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x_world, y_world, z_world, s=0.2, alpha=0.9)

    ax.view_init(elev=float(view_elev), azim=float(view_azim))
    _set_black_background(fig, ax)

    if no_axes:
        ax.set_axis_off()
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    plt.tight_layout(pad=0)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        out_png = os.path.join(out_dir, "voxel_scatter_blackbg_defaultcolor.png")
        plt.savefig(out_png, bbox_inches="tight", pad_inches=0, facecolor=fig.get_facecolor())
        print(f"[VIS] Saved scatter visualization to: {out_png}")

    plt.show()


def visualize_isosurface(
    occ_zyx: np.ndarray,
    pitch: float,
    origin_xyz: np.ndarray,
    out_dir: str = None,
    sigma: float = 0.6,
    level: float = 0.5,
    max_dim: int = 240,
    view_elev: float = 20.0,
    view_azim: float = -90.0,
    no_axes: bool = True
):
    """
    Visualize the voxel grid as an isosurface on a black background.
    """
    import matplotlib.pyplot as plt
    from skimage.measure import marching_cubes

    nz, ny, nx = occ_zyx.shape
    step = max(1, int(np.ceil(max(nz, ny, nx) / max_dim)))
    volume = occ_zyx[::step, ::step, ::step].astype(np.float32)

    if sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            volume = gaussian_filter(volume, sigma=float(sigma))
        except Exception as exc:
            print(f"[WARN] SciPy is unavailable for gaussian_filter. Smoothing skipped. Error: {exc}")

    print(f"[VIS] Isosurface settings: step={step}, volume_shape={volume.shape}, sigma={sigma}, level={level}")

    vertices, faces, normals, values = marching_cubes(volume, level=float(level))

    spacing = pitch * step
    z_world = origin_xyz[2] + (vertices[:, 0] + 0.5) * spacing
    y_world = origin_xyz[1] + (vertices[:, 1] + 0.5) * spacing
    x_world = origin_xyz[0] + (vertices[:, 2] + 0.5) * spacing

    fig = plt.figure(figsize=(9, 7), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        x_world,
        y_world,
        z_world,
        triangles=faces,
        linewidth=0.0,
        alpha=1.0,
        shade=True
    )

    ax.view_init(elev=float(view_elev), azim=float(view_azim))

    x_range = (x_world.min(), x_world.max())
    y_range = (y_world.min(), y_world.max())
    z_range = (z_world.min(), z_world.max())
    max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0])

    cx = 0.5 * (x_range[0] + x_range[1])
    cy = 0.5 * (y_range[0] + y_range[1])
    cz = 0.5 * (z_range[0] + z_range[1])

    ax.set_xlim(cx - max_range / 2, cx + max_range / 2)
    ax.set_ylim(cy - max_range / 2, cy + max_range / 2)
    ax.set_zlim(cz - max_range / 2, cz + max_range / 2)

    _set_black_background(fig, ax)

    if no_axes:
        ax.set_axis_off()
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    plt.tight_layout(pad=0)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        out_png = os.path.join(
            out_dir,
            f"voxel_isosurface_blackbg_defaultcolor_sigma{sigma}_level{level}.png"
        )
        plt.savefig(out_png, bbox_inches="tight", pad_inches=0, facecolor=fig.get_facecolor())
        print(f"[VIS] Saved isosurface visualization to: {out_png}")

    plt.show()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Voxelize an OBJ mesh and optionally visualize the result."
    )

    parser.add_argument("--obj", required=True, help="Path to the input OBJ file.")
    parser.add_argument("--out", required=True, help="Output .npz (recommended) or .npy file.")
    parser.add_argument(
        "--pitch",
        type=float,
        default=0.5,
        help="Voxel size in the same unit as the mesh coordinates."
    )

    parser.add_argument(
        "--solid",
        action="store_true",
        help="Fill the interior to produce a solid voxel object."
    )
    parser.add_argument(
        "--force_watertight",
        action="store_true",
        help="Attempt mesh repair before solid voxelization."
    )

    parser.add_argument(
        "--post_fill",
        action="store_true",
        help="Apply voxel-domain gap closing and hole filling after voxelization."
    )
    parser.add_argument(
        "--close_iter",
        type=int,
        default=2,
        help="Number of binary closing iterations for voxel post-processing."
    )

    parser.add_argument("--rot_x", type=float, default=0.0, help="Rotation angle around X in degrees.")
    parser.add_argument("--rot_y", type=float, default=0.0, help="Rotation angle around Y in degrees.")
    parser.add_argument("--rot_z", type=float, default=0.0, help="Rotation angle around Z in degrees.")
    parser.add_argument(
        "--rot_about",
        type=str,
        default="center",
        choices=["center", "origin"],
        help='Rotation center: "center" for mesh bounding-box center, or "origin".'
    )

    parser.add_argument("--vis", action="store_true", help="Enable visualization.")
    parser.add_argument("--vis_dir", default="", help="Directory for saving visualization images.")
    parser.add_argument(
        "--only_view",
        action="store_true",
        help="Render only the 3D view and skip slice visualization."
    )
    parser.add_argument(
        "--vis_mode",
        type=str,
        default="isosurface",
        choices=["scatter", "isosurface"],
        help="Visualization mode."
    )

    parser.add_argument(
        "--vis_slices",
        type=int,
        default=9,
        help="Number of Z slices to preview when slice visualization is enabled."
    )
    parser.add_argument(
        "--vis_max_points",
        type=int,
        default=600000,
        help="Maximum number of points for scatter visualization."
    )

    parser.add_argument(
        "--vis_iso_sigma",
        type=float,
        default=0.6,
        help="Gaussian smoothing sigma for isosurface visualization."
    )
    parser.add_argument(
        "--vis_iso_level",
        type=float,
        default=0.5,
        help="Marching cubes isosurface level."
    )
    parser.add_argument(
        "--vis_iso_max_dim",
        type=int,
        default=240,
        help="Maximum grid dimension used for isosurface downsampling."
    )

    parser.add_argument(
        "--view_elev",
        type=float,
        default=20.0,
        help="Camera elevation angle in degrees."
    )
    parser.add_argument(
        "--view_azim",
        type=float,
        default=-90.0,
        help="Camera azimuth angle in degrees."
    )
    parser.add_argument(
        "--no_axes",
        action="store_true",
        help="Hide axes, ticks, and grid for clean visualization output."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.pitch <= 0:
        raise ValueError("--pitch must be positive.")
    if args.close_iter < 0:
        raise ValueError("--close_iter must be non-negative.")
    if args.vis_slices <= 0:
        raise ValueError("--vis_slices must be positive.")
    if args.vis_max_points <= 0:
        raise ValueError("--vis_max_points must be positive.")
    if args.vis_iso_max_dim <= 0:
        raise ValueError("--vis_iso_max_dim must be positive.")

    mesh = load_mesh_any(args.obj)
    if mesh.is_empty:
        raise RuntimeError("Loaded mesh is empty. Please check the OBJ path or file format.")

    if abs(args.rot_x) > 1e-6 or abs(args.rot_y) > 1e-6 or abs(args.rot_z) > 1e-6:
        mesh = rotate_mesh(
            mesh,
            rot_x_deg=args.rot_x,
            rot_y_deg=args.rot_y,
            rot_z_deg=args.rot_z,
            about=args.rot_about
        )
        print(
            "[INFO] Applied rotation: "
            f"rot_x={args.rot_x}, rot_y={args.rot_y}, rot_z={args.rot_z}, about={args.rot_about}"
        )

    if args.force_watertight:
        mesh = mesh.process(validate=True)
        trimesh.repair.fill_holes(mesh)

    print(f"[INFO] Mesh vertices: {len(mesh.vertices)}")
    print(f"[INFO] Mesh faces   : {len(mesh.faces)}")
    print(f"[INFO] Watertight   : {mesh.is_watertight}")
    print(f"[INFO] Bounds (xyz) :\n{mesh.bounds}")

    occ_zyx, origin_xyz = voxelize_mesh(
        mesh=mesh,
        pitch=args.pitch,
        solid=args.solid,
        post_fill=args.post_fill,
        close_iter=args.close_iter
    )

    nz, ny, nx = occ_zyx.shape
    occupancy_ratio = float(occ_zyx.mean())
    print(f"[INFO] Voxel grid (Z, Y, X) = ({nz}, {ny}, {nx})")
    print(f"[INFO] Pitch                = {args.pitch}")
    print(f"[INFO] Occupancy ratio      = {occupancy_ratio:.6f}")

    save_outputs(args.out, occ_zyx, args.pitch, origin_xyz)
    print(f"[DONE] Saved voxel grid to: {args.out}")

    if args.vis:
        vis_dir = args.vis_dir.strip() or None

        if not args.only_view:
            visualize_slices(
                occ_zyx=occ_zyx,
                out_dir=vis_dir,
                num_slices=args.vis_slices
            )

        if args.vis_mode == "scatter":
            visualize_3d_scatter(
                occ_zyx=occ_zyx,
                pitch=args.pitch,
                origin_xyz=origin_xyz,
                out_dir=vis_dir,
                max_points=args.vis_max_points,
                view_elev=args.view_elev,
                view_azim=args.view_azim,
                no_axes=args.no_axes
            )
        else:
            visualize_isosurface(
                occ_zyx=occ_zyx,
                pitch=args.pitch,
                origin_xyz=origin_xyz,
                out_dir=vis_dir,
                sigma=args.vis_iso_sigma,
                level=args.vis_iso_level,
                max_dim=args.vis_iso_max_dim,
                view_elev=args.view_elev,
                view_azim=args.view_azim,
                no_axes=args.no_axes
            )


if __name__ == "__main__":
    main()