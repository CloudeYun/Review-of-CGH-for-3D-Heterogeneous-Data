#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert OBJ Mesh to Nx12 Triangle TXT with Face Colors

Author: Hao Yun
Date: 2026-03-25

Description:
    This script converts an OBJ mesh into a triangle-based Nx12 TXT file, where
    each row is formatted as:

        x1 y1 z1 x2 y2 z2 x3 y3 z3 r g b

    The processing pipeline includes:
        1. Mesh loading with optional face material extraction
        2. Degenerate face removal
        3. Optional mesh simplification
        4. Material ID remapping after simplification
        5. Optional mesh rotation
        6. Face color assignment
        7. Export to Nx12 triangle TXT

Coloring priority:
    1. Use OBJ face material information if available
    2. Fall back to connected-component based coloring

Example:
    python ./dataset/Obj2Mesh.py \
        --obj_path ./dataset/BunnyDragon.obj \
        --out_txt  ./dataset/Mesh/BunnyDragon_2k.txt \
        --target_faces 2000 \
        --rot_x 90 --rot_y 0 --rot_z 0
"""

import os
import argparse
import numpy as np
import trimesh


# -----------------------------------------------------------------------------
# Mesh Loading and Preprocessing
# -----------------------------------------------------------------------------
def load_mesh_with_material_info(obj_path: str):
    """
    Load a mesh from an OBJ file and extract face material IDs if available.

    Returns:
        mesh: trimesh.Trimesh
            The loaded triangular mesh.
        face_material_ids: np.ndarray or None
            Per-face material IDs of shape (F,), if available.
        material_names: list[str] or None
            Material names if available.
    """
    loaded = trimesh.load(obj_path, process=False)

    # Case 1: loaded as Scene
    if isinstance(loaded, trimesh.Scene):
        geometries = []
        all_face_materials = []
        material_names = []
        material_name_to_global = {}

        for geom_name, geom in loaded.geometry.items():
            if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
                continue

            geometries.append(geom)
            num_faces = len(geom.faces)

            local_face_materials = None
            local_material_names = None

            if hasattr(geom.visual, "face_materials") and geom.visual.face_materials is not None:
                face_materials = np.asarray(geom.visual.face_materials)
                if face_materials.ndim == 1 and len(face_materials) == num_faces:
                    local_face_materials = face_materials.astype(np.int64)

            if hasattr(geom.visual, "material") and geom.visual.material is not None:
                material = geom.visual.material
                if hasattr(material, "name") and material.name is not None:
                    local_material_names = [str(material.name)]

            if hasattr(geom.visual, "materials") and geom.visual.materials is not None:
                try:
                    local_material_names = [
                        str(m.name) if hasattr(m, "name") else f"mat_{i}"
                        for i, m in enumerate(geom.visual.materials)
                    ]
                except Exception:
                    pass

            if local_face_materials is None:
                if local_material_names is not None and len(local_material_names) >= 1:
                    material_name = local_material_names[0]
                else:
                    material_name = f"geom_{geom_name}"

                if material_name not in material_name_to_global:
                    material_name_to_global[material_name] = len(material_name_to_global)
                    material_names.append(material_name)

                global_id = material_name_to_global[material_name]
                local_face_materials = np.full((num_faces,), global_id, dtype=np.int64)

            else:
                remapped = np.empty_like(local_face_materials)
                if local_material_names is not None and len(local_material_names) > 0:
                    for local_id in np.unique(local_face_materials):
                        local_id_int = int(local_id)
                        if 0 <= local_id_int < len(local_material_names):
                            material_name = local_material_names[local_id_int]
                        else:
                            material_name = f"{geom_name}_mat_{local_id_int}"

                        if material_name not in material_name_to_global:
                            material_name_to_global[material_name] = len(material_name_to_global)
                            material_names.append(material_name)

                        remapped[local_face_materials == local_id] = material_name_to_global[material_name]
                else:
                    for local_id in np.unique(local_face_materials):
                        local_id_int = int(local_id)
                        material_name = f"{geom_name}_mat_{local_id_int}"

                        if material_name not in material_name_to_global:
                            material_name_to_global[material_name] = len(material_name_to_global)
                            material_names.append(material_name)

                        remapped[local_face_materials == local_id] = material_name_to_global[material_name]

                local_face_materials = remapped.astype(np.int64)

            all_face_materials.append(local_face_materials)

        if not geometries:
            raise RuntimeError("OBJ was loaded as a Scene, but no valid mesh geometry was found.")

        mesh = trimesh.util.concatenate(geometries)

        face_material_ids = None
        if len(all_face_materials) > 0:
            face_material_ids = np.concatenate(all_face_materials, axis=0).astype(np.int64)
            if len(face_material_ids) != len(mesh.faces):
                print("[WARN] face_material_ids length mismatch after concatenation. Material coloring will be ignored.")
                face_material_ids = None

        if mesh.faces.shape[1] != 3:
            mesh = mesh.triangulate()

        return mesh, face_material_ids, material_names if len(material_names) > 0 else None

    # Case 2: loaded as a single mesh
    mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise RuntimeError("Loaded object is not a valid triangular mesh.")

    if mesh.faces.shape[1] != 3:
        mesh = mesh.triangulate()

    face_material_ids = None
    material_names = None

    if hasattr(mesh.visual, "face_materials") and mesh.visual.face_materials is not None:
        face_materials = np.asarray(mesh.visual.face_materials)
        if face_materials.ndim == 1 and len(face_materials) == len(mesh.faces):
            face_material_ids = face_materials.astype(np.int64)

    if hasattr(mesh.visual, "materials") and mesh.visual.materials is not None:
        try:
            material_names = [
                str(m.name) if hasattr(m, "name") else f"mat_{i}"
                for i, m in enumerate(mesh.visual.materials)
            ]
        except Exception:
            material_names = None
    elif hasattr(mesh.visual, "material") and mesh.visual.material is not None:
        material = mesh.visual.material
        if hasattr(material, "name") and material.name is not None:
            material_names = [str(material.name)]

    return mesh, face_material_ids, material_names


def remove_degenerate_faces(
    mesh: trimesh.Trimesh,
    face_material_ids=None,
    eps: float = 1e-12
):
    """
    Remove degenerate triangles with near-zero area.

    Returns:
        mesh_new: trimesh.Trimesh
        face_material_ids_new: np.ndarray or None
    """
    vertices = mesh.vertices
    faces = mesh.faces

    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]

    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    keep_mask = area2 > eps

    if np.all(keep_mask):
        return mesh, face_material_ids

    mesh_new = trimesh.Trimesh(vertices=vertices, faces=faces[keep_mask], process=False)

    face_material_ids_new = None
    if face_material_ids is not None and len(face_material_ids) == len(faces):
        face_material_ids_new = np.asarray(face_material_ids)[keep_mask].astype(np.int64)

    return mesh_new, face_material_ids_new


def simplify_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """
    Simplify the mesh using quadric decimation.

    The function tries several possible keyword conventions for compatibility
    across trimesh versions.
    """
    target_faces = int(target_faces)
    if target_faces <= 0:
        return mesh
    if len(mesh.faces) <= target_faces:
        return mesh

    mesh_processed = mesh.copy()
    mesh_processed.process(validate=True)

    num_faces = len(mesh_processed.faces)
    reduction = 1.0 - (target_faces / float(num_faces))
    reduction = float(np.clip(reduction, 0.0, 1.0))

    tried_kwargs = []
    last_error = None

    for kwargs in (
        {"face_count": target_faces},
        {"target_count": target_faces},
        {"faces": target_faces},
        {"target_reduction": reduction},
    ):
        try:
            tried_kwargs.append(kwargs)
            mesh_simplified = mesh_processed.simplify_quadric_decimation(**kwargs)
            if mesh_simplified.faces.shape[1] != 3:
                mesh_simplified = mesh_simplified.triangulate()
            return mesh_simplified
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"simplify_quadric_decimation failed after trying {tried_kwargs}. "
        f"Last error: {last_error}"
    )


def remap_face_materials_after_simplify(
    mesh_before: trimesh.Trimesh,
    mesh_after: trimesh.Trimesh,
    face_material_ids_before
):
    """
    Approximate material remapping after simplification.

    Each simplified face inherits the material ID of the nearest original face
    centroid. This is a centroid-based nearest-neighbor approximation.
    """
    if face_material_ids_before is None:
        return None

    vertices_before = np.asarray(mesh_before.vertices, dtype=np.float64)
    faces_before = np.asarray(mesh_before.faces, dtype=np.int64)
    tri_before = vertices_before[faces_before]
    centroids_before = tri_before.mean(axis=1)

    vertices_after = np.asarray(mesh_after.vertices, dtype=np.float64)
    faces_after = np.asarray(mesh_after.faces, dtype=np.int64)
    tri_after = vertices_after[faces_after]
    centroids_after = tri_after.mean(axis=1)

    remapped_ids = np.empty((len(centroids_after),), dtype=np.int64)
    chunk_size = 2048

    for start in range(0, len(centroids_after), chunk_size):
        end = min(len(centroids_after), start + chunk_size)
        query = centroids_after[start:end]
        dist2 = np.sum((query[:, None, :] - centroids_before[None, :, :]) ** 2, axis=2)
        nn_idx = np.argmin(dist2, axis=1)
        remapped_ids[start:end] = np.asarray(face_material_ids_before, dtype=np.int64)[nn_idx]

    return remapped_ids


def rotate_mesh(
    mesh: trimesh.Trimesh,
    rot_x_deg: float = 0.0,
    rot_y_deg: float = 0.0,
    rot_z_deg: float = 0.0
) -> trimesh.Trimesh:
    """
    Rotate the mesh by Euler angles (degrees) in X-Y-Z order.
    """
    rx = np.deg2rad(float(rot_x_deg))
    ry = np.deg2rad(float(rot_y_deg))
    rz = np.deg2rad(float(rot_z_deg))

    rot_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(rx), -np.sin(rx)],
        [0.0, np.sin(rx),  np.cos(rx)]
    ], dtype=np.float64)

    rot_y = np.array([
        [ np.cos(ry), 0.0, np.sin(ry)],
        [0.0, 1.0, 0.0],
        [-np.sin(ry), 0.0, np.cos(ry)]
    ], dtype=np.float64)

    rot_z = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz),  np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    rotation = rot_z @ rot_y @ rot_x

    mesh_rotated = mesh.copy()
    vertices = np.asarray(mesh_rotated.vertices, dtype=np.float64)
    mesh_rotated.vertices = (rotation @ vertices.T).T
    return mesh_rotated


# -----------------------------------------------------------------------------
# Connected Components Fallback
# -----------------------------------------------------------------------------
class UnionFind:
    """
    Union-Find / Disjoint Set Union for triangle connectivity grouping.
    """
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def triangle_connected_components(
    mesh: trimesh.Trimesh,
    eps_ratio: float = 1e-6,
    min_keep: int = 30
):
    """
    Compute triangle connected components using quantized shared vertices.

    Returns:
        roots: np.ndarray
            Root component ID for each triangle.
        main_roots: np.ndarray or None
            IDs of large connected components sorted by size, or None if
            fewer than two valid components are found.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    num_triangles = faces.shape[0]

    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    span = vmax - vmin
    scale = float(np.max(span))
    eps = max(scale * eps_ratio, 1e-12)

    vertices_quantized = np.round(vertices / eps).astype(np.int64)

    vertex_id_map = {}
    vertex_ids = np.empty((vertices_quantized.shape[0],), dtype=np.int64)
    next_id = 0

    for i in range(vertices_quantized.shape[0]):
        key = (
            int(vertices_quantized[i, 0]),
            int(vertices_quantized[i, 1]),
            int(vertices_quantized[i, 2]),
        )
        if key in vertex_id_map:
            vertex_ids[i] = vertex_id_map[key]
        else:
            vertex_id_map[key] = next_id
            vertex_ids[i] = next_id
            next_id += 1

    tri_vertex_ids = vertex_ids[faces]

    uf = UnionFind(num_triangles)
    first_triangle_of_vertex = {}

    for tri_idx in range(num_triangles):
        for k in range(3):
            vertex_id = int(tri_vertex_ids[tri_idx, k])
            if vertex_id in first_triangle_of_vertex:
                uf.union(tri_idx, first_triangle_of_vertex[vertex_id])
            else:
                first_triangle_of_vertex[vertex_id] = tri_idx

    roots = np.array([uf.find(i) for i in range(num_triangles)], dtype=np.int64)
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    counts = np.bincount(inverse)

    keep_mask = counts >= int(min_keep)
    kept_roots = unique_roots[keep_mask]
    kept_counts = counts[keep_mask]

    if kept_roots.size < 2:
        return roots, None

    order = np.argsort(-kept_counts)
    main_roots = kept_roots[order]
    return roots, main_roots


# -----------------------------------------------------------------------------
# Coloring
# -----------------------------------------------------------------------------
def build_palette_for_n_materials(n_materials: int):
    """
    Build a default color palette.

    Intended defaults:
        - 2 materials: bunny + dragon
        - 3 materials: bunny + dragon + third object
        - more than 3: extended palette
    """
    base_palette = [
        np.array([0.62, 0.58, 0.50], dtype=np.float64),
        np.array([0.82, 0.82, 0.82], dtype=np.float64),
        np.array([0.71, 0.53, 0.39], dtype=np.float64),
        np.array([0.62, 0.72, 0.56], dtype=np.float64),
        np.array([0.78, 0.64, 0.70], dtype=np.float64),
        np.array([0.68, 0.68, 0.50], dtype=np.float64),
    ]

    if n_materials <= len(base_palette):
        return base_palette[:n_materials]

    return [base_palette[i % len(base_palette)] for i in range(n_materials)]


def colors_from_materials(
    mesh: trimesh.Trimesh,
    face_material_ids,
    material_names=None
):
    """
    Assign face colors directly from material IDs.

    Returns:
        colors: np.ndarray or None
            Face colors of shape (F, 3), or None if material coloring is unavailable.
    """
    num_faces = len(mesh.faces)
    if face_material_ids is None or len(face_material_ids) != num_faces:
        return None

    material_ids = np.asarray(face_material_ids, dtype=np.int64)
    unique_materials = np.unique(material_ids)
    num_materials = len(unique_materials)

    if num_materials < 2:
        return None

    palette = build_palette_for_n_materials(num_materials)
    colors = np.zeros((num_faces, 3), dtype=np.float64)

    print(f"[INFO] Material-based coloring detected: {num_materials} materials")
    for idx, material_id in enumerate(unique_materials):
        colors[material_ids == material_id] = palette[idx]

        if material_names is not None and int(material_id) < len(material_names):
            material_name = material_names[int(material_id)]
        else:
            material_name = f"material_{int(material_id)}"

        print(
            f"[INFO]   {material_name:<20s} -> "
            f"color={palette[idx].tolist()}  "
            f"tris={int(np.sum(material_ids == material_id))}"
        )

    return colors


def colors_from_connected_components(mesh: trimesh.Trimesh):
    """
    Fallback face coloring using the largest connected components.

    Supports 2 or 3 major components. Components are ordered by mean X position.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    center_x = triangles[:, :, 0].mean(axis=1)

    roots, main_roots = triangle_connected_components(mesh)
    num_faces = len(faces)

    if main_roots is None or len(main_roots) < 2:
        print("[WARN] Could not identify at least two large connected components. Falling back to uniform gray.")
        return np.tile(np.array([0.8, 0.8, 0.8], dtype=np.float64)[None, :], (num_faces, 1))

    num_selected = min(3, len(main_roots))
    selected_roots = main_roots[:num_selected]

    components = []
    for root_id in selected_roots:
        mask = (roots == root_id)
        mean_x = float(center_x[mask].mean())
        components.append((mean_x, mask))

    components.sort(key=lambda item: item[0])

    palette = build_palette_for_n_materials(len(components))
    colors = np.tile(np.array([0.8, 0.8, 0.8], dtype=np.float64)[None, :], (num_faces, 1))

    print(f"[INFO] Connected-component fallback coloring: {len(components)} components")
    for idx, (_, mask) in enumerate(components):
        colors[mask] = palette[idx]
        print(f"[INFO]   component_{idx} -> color={palette[idx].tolist()} tris={int(mask.sum())}")

    return colors


def export_txt_nx12_with_colors(
    mesh: trimesh.Trimesh,
    out_txt: str,
    face_material_ids=None,
    material_names=None,
    fmt: str = "%.10f"
):
    """
    Export triangles to an Nx12 TXT file, where each row is:

        x1 y1 z1 x2 y2 z2 x3 y3 z3 r g b
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    triangles_nx9 = triangles.reshape(-1, 9)
    num_faces = triangles_nx9.shape[0]

    colors = colors_from_materials(mesh, face_material_ids, material_names)

    if colors is None:
        print("[INFO] Falling back to connected-component coloring.")
        colors = colors_from_connected_components(mesh)

    if colors.shape != (num_faces, 3):
        raise RuntimeError(f"Color shape mismatch: expected {(num_faces, 3)}, got {colors.shape}")

    triangles_nx12 = np.concatenate([triangles_nx9, colors], axis=1)

    out_dir = os.path.dirname(out_txt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savetxt(out_txt, triangles_nx12, fmt=fmt)
    print(f"[DONE] Saved {triangles_nx12.shape[0]} triangles (Nx12) to: {out_txt}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an OBJ mesh to an Nx12 triangle TXT with face colors."
    )

    parser.add_argument(
        "--obj_path",
        type=str,
        required=True,
        help="Path to the input OBJ file."
    )
    parser.add_argument(
        "--out_txt",
        type=str,
        required=True,
        help="Path to the output Nx12 TXT file."
    )

    parser.add_argument(
        "--target_faces",
        type=int,
        default=60000,
        help="Target number of faces after simplification."
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Simplification ratio in (0, 1]. Overrides --target_faces if provided."
    )
    parser.add_argument(
        "--deg_eps",
        type=float,
        default=1e-12,
        help="Threshold for degenerate triangle removal."
    )
    parser.add_argument(
        "--fmt",
        type=str,
        default="%.10f",
        help="Numeric format used in np.savetxt."
    )

    parser.add_argument(
        "--rot_x",
        type=float,
        default=0.0,
        help="Rotation angle around X axis in degrees."
    )
    parser.add_argument(
        "--rot_y",
        type=float,
        default=0.0,
        help="Rotation angle around Y axis in degrees."
    )
    parser.add_argument(
        "--rot_z",
        type=float,
        default=0.0,
        help="Rotation angle around Z axis in degrees."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    mesh, face_material_ids, material_names = load_mesh_with_material_info(args.obj_path)
    print(f"[INFO] Loaded mesh: V={len(mesh.vertices)}, F={len(mesh.faces)}")

    if face_material_ids is not None:
        unique_materials = np.unique(face_material_ids)
        print(f"[INFO] Detected face materials: {len(unique_materials)} unique IDs -> {unique_materials.tolist()}")
        if material_names is not None:
            print(f"[INFO] Material names: {material_names}")

    mesh, face_material_ids = remove_degenerate_faces(
        mesh, face_material_ids, eps=args.deg_eps
    )
    print(f"[INFO] After degenerate-face removal: V={len(mesh.vertices)}, F={len(mesh.faces)}")

    if args.ratio is not None:
        if not (0.0 < args.ratio <= 1.0):
            raise ValueError("--ratio must be in the interval (0, 1].")
        target_faces = int(len(mesh.faces) * float(args.ratio))
    else:
        target_faces = int(args.target_faces)

    if 0 < target_faces < len(mesh.faces):
        print(f"[INFO] Simplifying mesh to target_faces={target_faces} (from {len(mesh.faces)})")
        mesh_before = mesh.copy()
        face_material_ids_before = None if face_material_ids is None else face_material_ids.copy()

        mesh = simplify_mesh(mesh, target_faces=target_faces)

        if face_material_ids_before is not None:
            print("[INFO] Remapping material IDs after simplification...")
            face_material_ids = remap_face_materials_after_simplify(
                mesh_before, mesh, face_material_ids_before
            )

        print(f"[INFO] After simplification: V={len(mesh.vertices)}, F={len(mesh.faces)}")
    else:
        print("[INFO] Simplification skipped.")

    mesh, face_material_ids = remove_degenerate_faces(
        mesh, face_material_ids, eps=args.deg_eps
    )
    print(f"[INFO] After post-simplification cleanup: V={len(mesh.vertices)}, F={len(mesh.faces)}")

    if (abs(args.rot_x) > 1e-12) or (abs(args.rot_y) > 1e-12) or (abs(args.rot_z) > 1e-12):
        print(f"[INFO] Rotating mesh: Rx={args.rot_x}, Ry={args.rot_y}, Rz={args.rot_z} degrees")
        mesh = rotate_mesh(
            mesh,
            rot_x_deg=args.rot_x,
            rot_y_deg=args.rot_y,
            rot_z_deg=args.rot_z
        )
    else:
        print("[INFO] Rotation skipped.")

    export_txt_nx12_with_colors(
        mesh=mesh,
        out_txt=args.out_txt,
        face_material_ids=face_material_ids,
        material_names=material_names,
        fmt=args.fmt
    )


if __name__ == "__main__":
    main()