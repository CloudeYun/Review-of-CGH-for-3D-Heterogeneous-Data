#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mesh CGH (Triangular Mesh) - GPU (CuPy) - Full-Color RGB

Author: Hao Yun
Date: 2026-03-25

Description:
    This script generates full-color computer-generated holograms (CGH) from a
    triangular mesh represented as an Nx12 text file:

        x1 y1 z1 x2 y2 z2 x3 y3 z3 r g b

    The pipeline includes:
        1. Mesh loading from Nx12 or Nx9 text
        2. Coordinate normalization and scaling
        3. Per-channel hologram generation on GPU
        4. Checkpointing and resume support
        5. RGB reconstruction across a z-range

Example:
    python ./model/Mesh_ComplexHologram.py \
      --mesh_txt /workspace/yh/project/CGHReview/dataset/Mesh3_n/BunnyDragon_RGB_50k_y0.txt \
      --outdir   /workspace/yh/project/CGHReview/result/Mesh3_n/50k_4 \
      --Nx 4096 --Ny 4096 --dx 8e-6 --dy 8e-6 \cz
      --lam_r 532e-9 --lam_g 532e-9 --lam_b 532e-9 \
      --shiftZ 0.05 --objectScaleRatio 2 \
      --target_depth 0.01 --z_boost 1.0 \
      --shading continuous --illu 0 0 1 --log_every 200 \
      --z_min 0.045 --z_max 0.055 --z_step 0.0001 \
      --save_every 1 \
      --resume 1 \
      --device 0
"""

import os
import math
import json
import argparse
import contextlib

import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
import cupy as cp


# =========================================================
# Compatibility
# =========================================================
@contextlib.contextmanager
def cp_errstate(**kwargs):
    """Compatibility wrapper for CuPy error-state handling."""
    err = getattr(cp, "errstate", None)
    if err is None:
        yield
    else:
        with err(**kwargs):
            yield


# =========================================================
# Image I/O Utilities
# =========================================================
def save_color_u8(path: str, img_rgb: np.ndarray):
    """Save an RGB image in uint8 format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_arr = np.asarray(img_rgb)

    if img_arr.ndim == 2:
        img_arr = np.repeat(img_arr[:, :, None], 3, axis=2)
    elif img_arr.ndim == 3:
        if img_arr.shape[2] == 1:
            img_arr = np.repeat(img_arr, 3, axis=2)
        elif img_arr.shape[2] > 3:
            img_arr = img_arr[:, :, :3]
    else:
        raise ValueError(f"save_color_u8 expects 2D or 3D array, got shape {img_arr.shape}")

    img_u8 = np.clip(img_arr * 255.0, 0.0, 255.0).astype(np.uint8)
    imageio.imwrite(path, img_u8)


def save_gray_u8(path: str, img_gray: np.ndarray):
    """Save a grayscale image in uint8 format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_u8 = np.clip(img_gray * 255.0, 0.0, 255.0).astype(np.uint8)
    imageio.imwrite(path, img_u8)


def save_phase_u8(path: str, phase_rad: np.ndarray):
    """Save a phase image in [-pi, pi] mapped to [0, 255]."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    norm = (phase_rad + np.pi) / (2 * np.pi)
    img_u8 = np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)
    imageio.imwrite(path, img_u8)


# =========================================================
# Checkpointing
# =========================================================
def checkpoint_path(outdir: str) -> str:
    """Return the checkpoint JSON path."""
    return os.path.join(outdir, "checkpoint.json")


def build_run_signature(args) -> dict:
    """Build a signature describing the current run configuration."""
    mesh_path = os.path.abspath(args.mesh_txt)
    try:
        mesh_stat = os.stat(mesh_path)
        mesh_mtime = float(mesh_stat.st_mtime)
        mesh_size = int(mesh_stat.st_size)
    except FileNotFoundError:
        mesh_mtime = None
        mesh_size = None

    return {
        "mesh_txt": mesh_path,
        "mesh_mtime": mesh_mtime,
        "mesh_size": mesh_size,
        "Nx": int(args.Nx),
        "Ny": int(args.Ny),
        "dx": float(args.dx),
        "dy": float(args.dy),
        "lam_r": float(args.lam_r),
        "lam_g": float(args.lam_g),
        "lam_b": float(args.lam_b),
        "shiftX": float(args.shiftX),
        "shiftY": float(args.shiftY),
        "shiftZ": float(args.shiftZ),
        "objectScaleRatio": float(args.objectScaleRatio),
        "target_depth": float(args.target_depth),
        "z_boost": float(args.z_boost),
        "shading": str(args.shading),
        "illu": [float(v) for v in args.illu],
        "z_min": float(args.z_min),
        "z_max": float(args.z_max),
        "z_step": float(args.z_step),
    }


def load_checkpoint(path: str):
    """Load checkpoint JSON if available."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] Failed to read checkpoint: {path} ({exc})")
        return None


def save_checkpoint(path: str, payload: dict):
    """Save checkpoint JSON atomically."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def try_load_hologram(holo_path: str):
    """Try loading an existing hologram .npy file."""
    if not os.path.exists(holo_path):
        return None
    try:
        hologram_np = np.load(holo_path)
        return cp.asarray(hologram_np)
    except Exception as exc:
        print(f"[WARN] Failed to load existing hologram: {holo_path} ({exc})")
        return None


def ensure_hologram_outputs(hologram: cp.ndarray, outdir: str, base_name: str):
    """Ensure amplitude/phase outputs exist for an already generated hologram."""
    phase_path = os.path.join(outdir, "Holograms", f"{base_name}_phase.npy")
    phase_png = os.path.join(outdir, "Holograms", f"{base_name}_phase.png")
    amp_path = os.path.join(outdir, "Holograms", f"{base_name}_amp.npy")
    amp_png = os.path.join(outdir, "Holograms", f"{base_name}_amp.png")

    if not (os.path.exists(phase_path) and os.path.exists(phase_png)):
        phase = cp.angle(hologram)
        np.save(phase_path, phase.get().astype(np.float32))
        save_phase_u8(phase_png, phase.get())

    if not (os.path.exists(amp_path) and os.path.exists(amp_png)):
        amp = cp.abs(hologram)
        amp_norm = amp / (cp.max(amp) + 1e-30)
        np.save(amp_path, amp.get().astype(np.float32))
        save_gray_u8(amp_png, amp_norm.get())


# =========================================================
# Mesh normalize/scale-shift (coordinates only)
# =========================================================
def normalize_centering_obj_np(obj_nx9: np.ndarray) -> np.ndarray:
    """Center the mesh and normalize by its maximum span."""
    obj = obj_nx9.astype(np.float64, copy=True)
    vertices = obj.reshape(-1, 3)
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = 0.5 * (vmin + vmax)

    vertices_centered = vertices - center
    span = vmax - vmin
    max_dim = float(np.max(span))
    if max_dim < 1e-12:
        max_dim = 1.0

    vertices_norm = vertices_centered / max_dim
    return vertices_norm.reshape(obj.shape)


def scale_shift_obj_np(obj_nx9: np.ndarray, scale_xyz, shift_xyz) -> np.ndarray:
    """Apply anisotropic scaling and translation to mesh coordinates."""
    sx, sy, sz = map(float, scale_xyz)
    tx, ty, tz = map(float, shift_xyz)

    out = np.zeros_like(obj_nx9, dtype=np.float64)
    out[:, 0:9:3] = obj_nx9[:, 0:9:3] * sx + tx
    out[:, 1:9:3] = obj_nx9[:, 1:9:3] * sy + ty
    out[:, 2:9:3] = obj_nx9[:, 2:9:3] * sz + tz
    return out


def debug_span(name: str, obj_nx9: np.ndarray):
    """Print mesh coordinate statistics for debugging."""
    vertices = obj_nx9.reshape(-1, 3)
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    span = vmax - vmin

    print(f"[DEBUG] {name}:")
    print(f"        min = ({vmin[0]:.6e}, {vmin[1]:.6e}, {vmin[2]:.6e})")
    print(f"        max = ({vmax[0]:.6e}, {vmax[1]:.6e}, {vmax[2]:.6e})")
    print(f"        span= ({span[0]:.6e}, {span[1]:.6e}, {span[2]:.6e})")
    return span


# =========================================================
# Vertex normals (CPU)
# =========================================================
def find_vertex_normal_vector_np(obj_nx9: np.ndarray):
    """Compute face normals and per-vertex normals on CPU."""
    obj = obj_nx9.astype(np.float64, copy=False)
    n_tri = obj.shape[0]

    p1, p2, p3 = obj[:, 0:3], obj[:, 3:6], obj[:, 6:9]
    face_normals = np.cross(p1 - p2, p3 - p2)

    norm = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    face_normals_unit = face_normals / norm

    all_vertices = np.zeros((3 * n_tri, 3), dtype=np.float64)
    all_vertices[0::3, :] = p1
    all_vertices[1::3, :] = p2
    all_vertices[2::3, :] = p3

    tri_occ = (np.arange(3 * n_tri) // 3).astype(np.int64)
    occ_face_normals = face_normals_unit[tri_occ]

    unique_vertices, inv = np.unique(all_vertices, axis=0, return_inverse=True)
    n_unique = unique_vertices.shape[0]

    acc = np.zeros((n_unique, 3), dtype=np.float64)
    np.add.at(acc, inv, occ_face_normals)

    acc_norm = np.linalg.norm(acc, axis=1, keepdims=True)
    acc_norm[acc_norm < 1e-12] = 1.0
    acc_unit = acc / acc_norm

    vertex_normals_list = acc_unit[inv]
    vertex_normals = np.zeros((n_tri, 9), dtype=np.float64)
    vertex_normals[:, 0:3] = vertex_normals_list[0::3, :]
    vertex_normals[:, 3:6] = vertex_normals_list[1::3, :]
    vertex_normals[:, 6:9] = vertex_normals_list[2::3, :]

    return face_normals_unit, vertex_normals


# =========================================================
# GPU kernel
# =========================================================
def as_continuous_gpu(
    x_np_3x3,
    ref_np_3x2,
    FX,
    FY,
    FZ,
    wavelength,
    illu_np,
    face_normals_np,
    vertex_normals_np,
    tri_idx,
    tri_color_intensity,
):
    """Compute one triangle contribution in the Fourier domain."""
    n = face_normals_np[tri_idx, :]
    if n[2] <= 0 or np.allclose(n, 0.0, atol=1e-15):
        return None

    theta = math.atan2(n[0], n[2]) if not (abs(n[0]) < 1e-15 and abs(n[2]) < 1e-15) else 0.0
    phi = math.atan2(n[1], math.sqrt(n[0] ** 2 + n[2] ** 2) + 1e-30)

    rot_np = np.array([
        [ math.cos(theta),                 0.0, -math.sin(theta)],
        [-math.sin(phi) * math.sin(theta),  math.cos(phi), -math.cos(theta) * math.sin(phi)],
        [ math.cos(phi) * math.sin(theta),  math.sin(phi),  math.cos(theta) * math.cos(phi)]
    ], dtype=np.float64)

    X = x_np_3x3
    X_local = rot_np @ X
    c = -X_local[:, 0].copy()
    X_local = X_local + c[:, None]

    if abs(X_local[0, 2] * X_local[1, 1] - X_local[1, 2] * X_local[0, 1]) < 1e-30:
        return None

    Xld = X_local[0:2, :].T
    b = Xld[2, 0] * Xld[1, 1] - Xld[2, 1] * Xld[1, 0]
    if abs(b) < 1e-30:
        return None

    A = np.zeros((2, 2), dtype=np.float64)
    for l in range(2):
        for m in range(2):
            col = 1 - m
            sign = 1.0 if m == 0 else -1.0
            A[l, m] = (ref_np_3x2[2, l] * Xld[1, col] - ref_np_3x2[1, l] * Xld[2, col]) / (b * sign)

    detA = float(np.linalg.det(A))
    if abs(detA) < 1e-30:
        return None
    tA = np.linalg.inv(A.T)

    illu = np.array(illu_np, dtype=np.float64)
    illu = illu / (np.linalg.norm(illu) + 1e-30)

    vn1 = vertex_normals_np[tri_idx, 0:3]
    vn2 = vertex_normals_np[tri_idx, 3:6]
    vn3 = vertex_normals_np[tri_idx, 6:9]

    av1 = max(float(vn1 @ illu + 0.1), 0.0)
    av2 = max(float(vn2 @ illu + 0.1), 0.0)
    av3 = max(float(vn3 @ illu + 0.1), 0.0)

    # Keep original channel modulation
    a1 = av1 * tri_color_intensity
    a2 = av3 * tri_color_intensity
    a3 = av2 * tri_color_intensity

    rot = cp.asarray(rot_np, dtype=cp.float64)
    c_cp = cp.asarray(c, dtype=cp.float64)
    tA_cp = cp.asarray(tA, dtype=cp.float64)

    flx = rot[0, 0] * FX + rot[0, 1] * FY + rot[0, 2] * FZ
    fly = rot[1, 0] * FX + rot[1, 1] * FY + rot[1, 2] * FZ
    flz = cp.sqrt(((1.0 / wavelength) ** 2 - flx ** 2 - fly ** 2) + 0j)

    uc = cp.asarray([0.0, 0.0, 1.0], dtype=cp.float64)
    du = (1.0 / wavelength) * cp.asarray(
        [cp.dot(rot[0, :], uc), cp.dot(rot[1, :], uc)],
        dtype=cp.float64
    )

    flux = flx - du[0]
    fluy = fly - du[1]
    flxA = tA_cp[0, 0] * flux + tA_cp[0, 1] * fluy
    flyA = tA_cp[1, 0] * flux + tA_cp[1, 1] * fluy

    EPS = 1e-12
    with cp_errstate(divide="ignore", invalid="ignore"):
        D1 = (
            cp.exp(-1j * 2 * cp.pi * (flxA + flyA)) * (1j - 2 * cp.pi * (flxA + flyA))
            / (8 * cp.pi ** 3 * flyA * (flxA + flyA) ** 2)
            + cp.exp(-1j * 2 * cp.pi * flxA) * (2 * cp.pi * flxA - 1j)
            / (8 * cp.pi ** 3 * flxA ** 2 * flyA)
            + (1j * (2 * flxA + flyA)) / (8 * cp.pi ** 3 * flxA ** 2 * (flxA + flyA) ** 2)
        )

        D2 = (
            cp.exp(-1j * 2 * cp.pi * (flxA + flyA))
            * (1j * (flxA + 2 * flyA) - 2 * cp.pi * flyA * (flxA + flyA))
            / (8 * cp.pi ** 3 * flyA ** 2 * (flxA + flyA) ** 2)
            + cp.exp(-1j * 2 * cp.pi * flxA) * (-1j)
            / (8 * cp.pi ** 3 * flxA * flyA ** 2)
            + 1j / (8 * cp.pi ** 3 * flxA * (flxA + flyA) ** 2)
        )

        D3 = (
            -cp.exp(-1j * 2 * cp.pi * (flxA + flyA)) / (4 * cp.pi ** 2 * flyA * (flxA + flyA))
            + cp.exp(-1j * 2 * cp.pi * flxA) / (4 * cp.pi ** 2 * flxA * flyA)
            - 1.0 / (4 * cp.pi ** 2 * flxA * (flxA + flyA))
        )

        m = cp.isclose(flxA, -flyA, atol=EPS)
        D1 = cp.where(
            m,
            (
                (-2 * cp.pi * flxA + 1j) / (8 * cp.pi ** 3 * flxA ** 3) * cp.exp(-1j * 2 * cp.pi * flxA)
                - (1j * 2 * cp.pi ** 2 * flxA ** 2 + 1j) / (8 * cp.pi ** 3 * flxA ** 3)
            ),
            D1
        )
        D2 = cp.where(
            m,
            (
                (-1j) / (8 * cp.pi ** 3 * flxA ** 3) * cp.exp(-1j * 2 * cp.pi * flxA)
                + (-1j * 2 * cp.pi ** 2 * flxA ** 2 + 1j + 2 * cp.pi * flxA) / (8 * cp.pi ** 3 * flxA ** 3)
            ),
            D2
        )
        D3 = cp.where(
            m,
            (
                (-1j) / (4 * cp.pi ** 2 * flxA ** 2) * cp.exp(-1j * 2 * cp.pi * flxA)
                + (-1j * 2 * cp.pi * flxA + 1) / (4 * cp.pi ** 2 * flxA ** 2)
            ),
            D3
        )

        m = (~cp.isclose(flxA, 0.0, atol=EPS)) & cp.isclose(flyA, 0.0, atol=EPS)
        D1m = (
            (1j * 4 * cp.pi ** 2 * flxA ** 2 + 4 * cp.pi * flxA - 2j) / (8 * cp.pi ** 3 * flxA ** 3)
            * cp.exp(-1j * 2 * cp.pi * flxA)
            + 1j / (4 * cp.pi ** 3 * flxA ** 3)
        )
        D1 = cp.where(m, D1m, D1)
        D2 = cp.where(m, 0.5 * D1m, D2)
        D3 = cp.where(
            m,
            (((1j * 2 * cp.pi * flxA + 1) * cp.exp(-1j * 2 * cp.pi * flxA) - 1) / (4 * cp.pi ** 2 * flxA ** 2)),
            D3
        )

        m = cp.isclose(flxA, 0.0, atol=EPS) & (~cp.isclose(flyA, 0.0, atol=EPS))
        D1 = cp.where(
            m,
            (
                -(2 * cp.pi * flyA + 1) * cp.exp(-1j * 2 * cp.pi * flyA) / (8 * cp.pi ** 3 * flyA ** 3)
                + (-1j * 2 * cp.pi ** 2 * flyA ** 2 + 1) / (8 * cp.pi ** 3 * flyA ** 3)
            ),
            D1
        )
        D2 = cp.where(
            m,
            (
                (-cp.pi * flyA + 1j) / (4 * cp.pi ** 3 * flyA ** 3) * cp.exp(-1j * 2 * cp.pi * flyA)
                - (1j + cp.pi * flyA) / (4 * cp.pi ** 3 * flyA ** 3)
            ),
            D2
        )
        D3 = cp.where(
            m,
            (cp.exp(-1j * 2 * cp.pi * flyA) / (2 * cp.pi * flyA) + (1 - 1j) / (2 * cp.pi * flyA)),
            D3
        )

        m = cp.isclose(flxA, 0.0, atol=EPS) & cp.isclose(flyA, 0.0, atol=EPS)
        D1 = cp.where(m, 1.0 / 3.0, D1)
        D2 = cp.where(m, 1.0 / 6.0, D2)
        D3 = cp.where(m, 1.0 / 2.0, D3)

    G0 = (a2 - a1) * D1 + (a3 - a2) * D2 + a1 * D3
    J = 1.0 / detA
    Gl = J * G0
    phase_local = cp.dot(uc, rot.T @ c_cp)
    Gl = Gl * cp.exp(-1j * 2 * cp.pi / wavelength * phase_local)
    J2 = flz / FZ
    G = Gl * J2 * cp.exp(1j * 2 * cp.pi * (flx * c_cp[0] + fly * c_cp[1] + flz * c_cp[2]))
    return G


def gen_h_gpu(
    obj_nx9: np.ndarray,
    obj_colors_nx3: np.ndarray,
    color_channel_idx: int,
    holo_param: dict,
    shading_param: dict,
    log_every: int = 200,
):
    """Generate one hologram channel on GPU."""
    Nx, Ny = int(holo_param["Nx"]), int(holo_param["Ny"])
    dx, dy = float(holo_param["dx"]), float(holo_param["dy"])
    wavelength = float(holo_param["wavelength"])

    dfx = 1.0 / dx / Nx
    dfy = 1.0 / dy / Ny
    fx_1d = np.arange(-Nx // 2, Nx // 2, dtype=np.float64) * dfx
    fy_1d = np.arange(-Ny // 2, Ny // 2, dtype=np.float64) * dfy

    FX_np, FY_np = np.meshgrid(fx_1d, fy_1d)
    FX = cp.asarray(FX_np, dtype=cp.float64)
    FY = cp.asarray(FY_np, dtype=cp.float64)
    FZ = cp.sqrt(((1.0 / wavelength) ** 2 - FX ** 2 - FY ** 2) + 0j)

    GG = cp.zeros((Ny, Nx), dtype=cp.complex128)
    ref_np = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0]], dtype=np.float64)

    illu_np = np.array(shading_param["illu"], dtype=np.float64)
    face_normals_np, vertex_normals_np = find_vertex_normal_vector_np(obj_nx9)

    n_tri = obj_nx9.shape[0]
    pbar = tqdm(range(n_tri), desc=f"Gen Channel {color_channel_idx}", ncols=110)

    for tri_idx in pbar:
        intensity = float(obj_colors_nx3[tri_idx, color_channel_idx])
        if intensity < 0.01:
            continue

        X = obj_nx9[tri_idx, :].reshape((3, 3), order="F")
        G = as_continuous_gpu(
            X,
            ref_np,
            FX,
            FY,
            FZ,
            wavelength,
            illu_np,
            face_normals_np,
            vertex_normals_np,
            tri_idx,
            intensity,
        )
        if G is not None:
            GG += G

        if log_every > 0 and ((tri_idx + 1) % log_every == 0):
            max_val = float(cp.abs(GG).max().get())
            pbar.set_postfix_str(f"|GG|max={max_val:.2e}")

    hologram = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(GG)))
    return hologram


def fresnel_propagation_as_gpu(inp: cp.ndarray, dx: float, dy: float, z: float, lam: float):
    """Band-limited Fresnel propagation with 2x zero padding."""
    Ny, Nx = inp.shape
    Nxx, Nyy = 2 * Nx, 2 * Ny

    inp2 = cp.zeros((Nyy, Nxx), dtype=cp.complex128)
    start_x = Nxx // 2 - Nx // 2
    start_y = Nyy // 2 - Ny // 2
    inp2[start_y:start_y + Ny, start_x:start_x + Nx] = inp

    dal = 1.0 / (Nxx * dx)
    dbl = 1.0 / (Nyy * dy)
    al_1d = (cp.arange(1, Nxx + 1, dtype=cp.float64) - (Nxx / 2.0)) * dal
    bl_1d = (cp.arange(1, Nyy + 1, dtype=cp.float64) - (Nyy / 2.0)) * dbl
    AL, BL = cp.meshgrid(al_1d, bl_1d)

    A = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(inp2)))
    denom = cp.sqrt((1.0 / lam ** 2 - AL ** 2 - BL ** 2) + 0j)
    prop_kernel = cp.exp(1j * 2 * cp.pi * z * denom)

    sx, sy = 0.0, 0.0
    fla = cp.abs(-AL * z / denom + sx)
    flb = cp.abs(-BL * z / denom + sy)
    prop_kernel = cp.where(fla > 1.0 / (2.0 * dal), 0.0, prop_kernel)
    prop_kernel = cp.where(flb > 1.0 / (2.0 * dbl), 0.0, prop_kernel)

    intermediate = cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(A * prop_kernel)))
    out = intermediate[start_y:start_y + Ny, start_x:start_x + Nx]
    return out


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mesh_txt", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)

    parser.add_argument("--Nx", type=int, default=1920)
    parser.add_argument("--Ny", type=int, default=1080)
    parser.add_argument("--dx", type=float, default=8e-6)
    parser.add_argument("--dy", type=float, default=8e-6)

    parser.add_argument("--lam_r", type=float, default=638e-9)
    parser.add_argument("--lam_g", type=float, default=520e-9)
    parser.add_argument("--lam_b", type=float, default=450e-9)

    parser.add_argument("--shiftX", type=float, default=0.0)
    parser.add_argument("--shiftY", type=float, default=0.0)
    parser.add_argument("--shiftZ", type=float, default=0.1)
    parser.add_argument("--objectScaleRatio", type=float, default=3.0)

    parser.add_argument(
        "--target_depth",
        type=float,
        default=0.02,
        help="Desired object thickness along z after scaling (meters)."
    )
    parser.add_argument(
        "--z_boost",
        type=float,
        default=1.0,
        help="Extra multiplier on z thickness (optional)."
    )

    parser.add_argument("--shading", type=str, default="continuous", choices=["continuous"])
    parser.add_argument("--illu", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--z_min", type=float, required=True)
    parser.add_argument("--z_max", type=float, required=True)
    parser.add_argument("--z_step", type=float, required=True)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument(
        "--resume",
        type=int,
        default=1,
        help="Resume from checkpoint if available (1=yes, 0=no)."
    )

    return parser.parse_args()


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    cp.cuda.Device(args.device).use()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "Holograms"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "recon_rgb"), exist_ok=True)

    ckpt_path = checkpoint_path(outdir)
    run_sig = build_run_signature(args)

    ckpt = None
    if args.resume:
        ckpt = load_checkpoint(ckpt_path)
        if ckpt is not None and ckpt.get("signature") != run_sig:
            print("[WARN] Checkpoint parameters mismatch; starting fresh.")
            ckpt = None

    if ckpt is None:
        ckpt = {"signature": run_sig, "holograms": {}, "recon": {"last_idx": -1}}
    ckpt.setdefault("holograms", {})
    ckpt.setdefault("recon", {"last_idx": -1})
    if "last_idx" not in ckpt["recon"]:
        ckpt["recon"]["last_idx"] = -1

    print(f"[INFO] Loading {args.mesh_txt}")
    data = np.loadtxt(args.mesh_txt, dtype=np.float64)

    if data.ndim != 2 or data.shape[1] not in (9, 12):
        raise RuntimeError(f"mesh_txt must be Nx9 or Nx12, got {data.shape}")

    if data.shape[1] == 12:
        obj = data[:, :9]
        tri_colors = data[:, 9:12]
        tri_colors = np.clip(tri_colors, 0.0, 1.0)
        print("[INFO] Detected Nx12 txt: using embedded per-triangle RGB.")
    else:
        obj = data
        tri_colors = np.tile(np.array([0.8, 0.8, 0.8], dtype=np.float64)[None, :], (obj.shape[0], 1))
        print("[WARN] Nx9 txt has no colors; fallback to all grey.")

    debug_span("RAW input", obj)

    obj_n = normalize_centering_obj_np(obj)
    debug_span("After normalize", obj_n)

    holo_sx = args.Nx * args.dx
    obj_size_xy = holo_sx / float(args.objectScaleRatio)

    vertices_norm = obj_n.reshape(-1, 3)
    z_span_n = float(vertices_norm[:, 2].max() - vertices_norm[:, 2].min()) + 1e-30
    target_depth = float(args.target_depth) * float(args.z_boost)

    sx = obj_size_xy
    sy = obj_size_xy
    sz = target_depth / z_span_n

    print("[INFO] XY size control:")
    print(f"       holo_sx={holo_sx:.6e}, objectScaleRatio={args.objectScaleRatio:.3f}, obj_size_xy={obj_size_xy:.6e}")
    print("[INFO] Z thickness control:")
    print(f"       normalized z_span={z_span_n:.6e}, target_depth={target_depth:.6e}, so sz={sz:.6e}")

    obj = scale_shift_obj_np(obj_n, [sx, sy, sz], [args.shiftX, args.shiftY, args.shiftZ])
    debug_span("After scaleShift (final)", obj)

    z_all = obj[:, 2:9:3]
    print(f"[DEBUG] final z thickness = {float(z_all.max() - z_all.min()):.6e} (m)")

    channels = [
        {"name": "Red", "lam": args.lam_r, "idx": 0},
        {"name": "Green", "lam": args.lam_g, "idx": 1},
        {"name": "Blue", "lam": args.lam_b, "idx": 2},
    ]

    hologram_list = []
    shading_param = dict(illu=np.array(args.illu, dtype=np.float64), con=1)

    # Generate holograms
    for ch in channels:
        print(f"\n=== Processing {ch['name']} Channel (lambda={ch['lam']:.2e}) ===")
        holo_param = dict(Nx=args.Nx, Ny=args.Ny, dx=args.dx, dy=args.dy, wavelength=ch["lam"])
        base_name = f"channel_{ch['name']}"
        holo_path = os.path.join(outdir, "Holograms", f"{base_name}_complex.npy")

        H = None
        if args.resume:
            H = try_load_hologram(holo_path)
            if H is not None:
                print(f"[INFO] Reusing existing hologram: {holo_path}")
                ensure_hologram_outputs(H, outdir, base_name)

        if H is None:
            H = gen_h_gpu(
                obj,
                tri_colors,
                ch["idx"],
                holo_param,
                shading_param,
                log_every=args.log_every,
            )

            # Keep original orientation fix
            H = cp.flipud(H)

            np.save(holo_path, H.get().astype(np.complex128))

            phase = cp.angle(H)
            np.save(
                os.path.join(outdir, "Holograms", f"{base_name}_phase.npy"),
                phase.get().astype(np.float32)
            )
            save_phase_u8(
                os.path.join(outdir, "Holograms", f"{base_name}_phase.png"),
                phase.get()
            )

            amp = cp.abs(H)
            amp_norm = amp / (cp.max(amp) + 1e-30)
            np.save(
                os.path.join(outdir, "Holograms", f"{base_name}_amp.npy"),
                amp.get().astype(np.float32)
            )
            save_gray_u8(
                os.path.join(outdir, "Holograms", f"{base_name}_amp.png"),
                amp_norm.get()
            )

        hologram_list.append({"H": H, "lam": ch["lam"]})
        ckpt["holograms"][base_name] = True
        save_checkpoint(ckpt_path, ckpt)
        cp.get_default_memory_pool().free_all_blocks()

    # Reconstruction scan (keep original RGB-coupled normalization)
    z_list = np.arange(args.z_min, args.z_max + 0.5 * args.z_step, args.z_step, dtype=np.float64)

    start_idx = 0
    if args.resume:
        last_idx = int(ckpt.get("recon", {}).get("last_idx", -1))
        if last_idx >= 0:
            start_idx = min(last_idx + 1, len(z_list))
            if start_idx > 0:
                print(f"[INFO] Resuming reconstruction from step {start_idx}/{len(z_list)}")

    if start_idx >= len(z_list):
        print(f"\n=== RGB Reconstruction Scan already complete ({len(z_list)} steps) ===")
        print(f"[DONE] Results saved to: {outdir}")
        return

    print(f"\n=== Starting RGB Reconstruction Scan ({len(z_list)} steps) ===")
    pbar = tqdm(range(start_idx, len(z_list)), desc="RGB Recon", ncols=110)

    for idx in pbar:
        z = float(z_list[idx])

        rec_amp = []
        for item in hologram_list:
            Hc = item["H"] - cp.mean(item["H"])
            U_rec = fresnel_propagation_as_gpu(Hc, args.dx, args.dy, z, item["lam"])
            rec_amp.append(cp.abs(U_rec).astype(cp.float32))

        amp_rgb = cp.stack(rec_amp, axis=-1)   # (H, W, 3)
        vis = cp.log1p(amp_rgb)                # joint log

        bg = cp.percentile(vis, 20)            # joint background
        vis = cp.maximum(vis - bg, 0.0)

        p_hi = cp.percentile(vis, 99.5)        # joint scale
        img_rgb_cp = cp.clip(vis / (p_hi + 1e-30), 0.0, 1.0)

        if args.save_every > 0 and (idx % args.save_every == 0):
            save_color_u8(
                os.path.join(outdir, "recon_rgb", f"recon_rgb_z_{z:.6f}.png"),
                img_rgb_cp.get()
            )

        ckpt["recon"]["last_idx"] = idx
        save_checkpoint(ckpt_path, ckpt)

    print(f"[DONE] Results saved to: {outdir}")


if __name__ == "__main__":
    main()