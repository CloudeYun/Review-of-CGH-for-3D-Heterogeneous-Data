'''
LDI:

python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_ASM.py \
 --amp_r /workspace/yh/project/CGHReview/result/Layer/LDI_single_complexRGB_3bins_50_53mm/Hologram/amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/Layer/LDI_single_complexRGB_3bins_50_53mm/Hologram/pha_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/Layer/LDI_single_complexRGB_3bins_50_53mm/Hologram/amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/Layer/LDI_single_complexRGB_3bins_50_53mm/Hologram/pha_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/Layer/LDI_single_complexRGB_3bins_50_53mm/Hologram/amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/Layer/LDI_single_complexRGB_3bins_50_53mm/Hologram/pha_B.npy \
  --outdir /workspace/yh/project/CGHReview/result/Layer/LDI_single_complexRGB_3bins_50_53mm/ASM_ReconFromHologram_0.9 \
  --wavelength 532e-9 \
  --pitch 4e-6 \
  --z_list "0.0505,0.0515,0.0525" \
  --gamma 0.9 \
  --out_size 2048





python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_ASM.py \
 --amp_r /workspace/yh/project/CGHReview/result/Layer/RGBD_complexRGB_3bins_50_53mm/Hologram/amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/Layer/RGBD_complexRGB_3bins_50_53mm/Hologram/pha_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/Layer/RGBD_complexRGB_3bins_50_53mm/Hologram/amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/Layer/RGBD_complexRGB_3bins_50_53mm/Hologram/pha_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/Layer/RGBD_complexRGB_3bins_50_53mm/Hologram/amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/Layer/RGBD_complexRGB_3bins_50_53mm/Hologram/pha_B.npy \
  --outdir /workspace/yh/project/CGHReview/result/Layer/RGBD_complexRGB_3bins_50_53mm/ASM_ReconFromHologram_0.9 \
  --wavelength 532e-9 \
  --pitch 4e-6 \
  --zmin 0.050 --zmax 0.053 --step 0.0001 \
  --gamma 0.9 \
  --out_size 2048

  python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_ASM.py \
 --amp_r /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_3bins_50_53mm/Hologram/amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_3bins_50_53mm/Hologram/pha_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_3bins_50_53mm/Hologram/amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_3bins_50_53mm/Hologram/pha_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_3bins_50_53mm/Hologram/amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_3bins_50_53mm/Hologram/pha_B.npy \
  --outdir /workspace/yh/project/CGHReview/result/Layer3/RGBD_complexRGB_3bins_50_53mm/ASM_ReconFromHologram_0.65 \
  --wavelength 532e-9 \
  --pitch 4e-6 \
  --zmin 0.050 --zmax 0.054 --step 0.0001 \
  --gamma 0.65 \
  --out_size 2048
   
PCD:

  python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_ASM.py \
 --amp_r /workspace/yh/project/CGHReview/result/PointCloud2/PCD_complexRGB_2bins_3e6_50_53mm_2/Hologram/amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/PointCloud2/PCD_complexRGB_2bins_3e6_50_53mm_2/Hologram/phase_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/PointCloud2/PCD_complexRGB_2bins_3e6_50_53mm_2/Hologram/amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/PointCloud2/PCD_complexRGB_2bins_3e6_50_53mm_2/Hologram/phase_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/PointCloud2/PCD_complexRGB_2bins_3e6_50_53mm_2/Hologram/amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/PointCloud2/PCD_complexRGB_2bins_3e6_50_53mm_2/Hologram/phase_B.npy \
  --outdir /workspace/yh/project/CGHReview/result/PointCloud2/PCD_complexRGB_2bins_3e6_50_53mm_2/ASM_ReconFromHologram_0.65 \
  --wavelength 532e-9 \
  --pitch 5e-6 \
  --zmin 0.05 --zmax 0.054 --step 0.0001 \
  --gamma 0.65 \
  --out_size 2048

  # --zmin可替换为 --z_list "0.0505,0.0515,0.0525" 

LF:
python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_ASM.py \
 --amp_r /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Hologram/hologram_amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Hologram/hologram_phase_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Hologram/hologram_amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Hologram/hologram_phase_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Hologram/hologram_amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Hologram/hologram_phase_B.npy \
 --outdir /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/ASM_ReconFromHologram_0.9 \
 --wavelength 532e-9 \
 --pitch 2e-6 \
 --zmin -0.053 \
 --zmax -0.050 \
 --step 0.0001 \
 --gamma 0.9

 python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_ASM.py \
 --amp_r /workspace/yh/project/CGHReview/result/LightField3/orth_RGB_complexHologram_20_400/Hologram/hologram_amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/LightField3/orth_RGB_complexHologram_20_400/Hologram/hologram_phase_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/LightField3/orth_RGB_complexHologram_20_400/Hologram/hologram_amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/LightField3/orth_RGB_complexHologram_20_400/Hologram/hologram_phase_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/LightField3/orth_RGB_complexHologram_20_400/Hologram/hologram_amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/LightField3/orth_RGB_complexHologram_20_400/Hologram/hologram_phase_B.npy \
 --outdir /workspace/yh/project/CGHReview/result/LightField3/orth_RGB_complexHologram_20_400/ASM_ReconFromHologram_0.65 \
 --wavelength 532e-9 \
 --pitch 2e-6 \
 --zmin -0.056 \
 --zmax -0.050 \
 --step 0.0001 \
 --gamma 0.65

python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_ASM.py \
 --amp_r /workspace/yh/project/CGHReview/result/LightField/pers_RGB_ComplexHologram_20_400/Hologram/hologram_amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/LightField/pers_RGB_ComplexHologram_20_400/Hologram/hologram_phase_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/LightField/pers_RGB_ComplexHologram_20_400/Hologram/hologram_amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/LightField/pers_RGB_ComplexHologram_20_400/Hologram/hologram_phase_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/LightField/pers_RGB_ComplexHologram_20_400/Hologram/hologram_amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/LightField/pers_RGB_ComplexHologram_20_400/Hologram/hologram_phase_B.npy \
 --outdir /workspace/yh/project/CGHReview/result/LightField/pers_RGB_ComplexHologram_20_400/ASM_ReconFromHologram_0.9 \
 --wavelength 532e-9 \
 --pitch 2e-6 \
 --zmin -0.052 \
 --zmax -0.049 \
 --step 0.0001 \
 --gamma 0.9

 python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_ASM.py \
 --amp_r  /workspace/yh/project/CGHReview/result/Mesh2/BunnyDragon_RGB_FromTxtColor_20k_5_1/Holograms/channel_Red_amp.npy \
 --pha_r /workspace/yh/project/CGHReview/result/Mesh2/BunnyDragon_RGB_FromTxtColor_20k_5_1/Holograms/channel_Red_phase.npy \
 --amp_g /workspace/yh/project/CGHReview/result/Mesh2/BunnyDragon_RGB_FromTxtColor_20k_5_1/Holograms/channel_Green_amp.npy \
 --pha_g /workspace/yh/project/CGHReview/result/Mesh2/BunnyDragon_RGB_FromTxtColor_20k_5_1/Holograms/channel_Green_phase.npy \
 --amp_b /workspace/yh/project/CGHReview/result/Mesh2/BunnyDragon_RGB_FromTxtColor_20k_5_1/Holograms/channel_Blue_amp.npy \
 --pha_b /workspace/yh/project/CGHReview/result/Mesh2/BunnyDragon_RGB_FromTxtColor_20k_5_1/Holograms/channel_Blue_phase.npy \
 --outdir /workspace/yh/project/CGHReview/result/Mesh2/BunnyDragon_RGB_FromTxtColor_20k_5_1/ASM_ReconFromHologram_0.65 \
 --wavelength 532e-9 \
 --pitch 8e-6 \
 --zmin 4.5 \
 --zmax 5.5 \
 --step 0.001 \
 --gamma 0.65

 python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_ASM.py \
 --amp_r /workspace/yh/project/CGHReview/result/LightField3/orth_RGB_complexHologram_20_400/Hologram/hologram_amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/LightField3/orth_RGB_complexHologram_20_400/Hologram/hologram_phase_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/LightField3/orth_RGB_complexHologram_20_400/Hologram/hologram_amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/LightField3/orth_RGB_complexHologram_20_400/Hologram/hologram_phase_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/LightField3/orth_RGB_complexHologram_20_400/Hologram/hologram_amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/LightField3/orth_RGB_complexHologram_20_400/Hologram/hologram_phase_B.npy \
 --outdir /workspace/yh/project/CGHReview/result/LightField3/orth_RGB_complexHologram_20_400/ASM_ReconFromHologram_0.9 \
 --wavelength 532e-9 \
 --pitch 2e-6 \
 --zmin -0.054 \
 --zmax -0.050 \
 --step 0.0001 \
 --gamma 0.65

  python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_ASM.py \
 --amp_r /workspace/yh/project/CGHReview/result/LightField3/pers_RGB_ComplexHologram_40_800_2/Hologram/hologram_amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/LightField3/pers_RGB_ComplexHologram_40_800_2/Hologram/hologram_phase_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/LightField3/pers_RGB_ComplexHologram_40_800_2/Hologram/hologram_amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/LightField3/pers_RGB_ComplexHologram_40_800_2/Hologram/hologram_phase_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/LightField3/pers_RGB_ComplexHologram_40_800_2/Hologram/hologram_amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/LightField3/pers_RGB_ComplexHologram_40_800_2/Hologram/hologram_phase_B.npy \
 --outdir /workspace/yh/project/CGHReview/result/LightField3/pers_RGB_ComplexHologram_40_800_2/ASM_ReconFromHologram_2\
 --wavelength 532e-9 \
 --pitch 2e-6 \
 --zmin -0.045 \
 --zmax -0.040 \
 --step 0.0001 \
 --gamma 0.65

 Voxel
python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_ASM.py \
 --amp_r /workspace/yh/project/CGHReview/result/Voxel2/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/Voxel2/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/pha_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/Voxel2/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/Voxel2/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/pha_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/Voxel2/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/Voxel2/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/pha_B.npy \
  --outdir /workspace/yh/project/CGHReview/result/Voxel2/Voxel_renderBins_complexRGB_100bins_50_53mm/ASM_ReconFromHologram_0.65 \
  --wavelength 532e-9 \
  --pitch 4e-6 \
  --zmin 0.050 --zmax 0.056 --step 0.0001 \
  --gamma 0.65 \
  --out_size 2048

  python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_ASM.py \
 --amp_r /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/pha_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/pha_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/Hologram/pha_B.npy \
  --outdir /workspace/yh/project/CGHReview/result/Voxel3/Voxel_renderBins_complexRGB_100bins_50_53mm/ASM_ReconFromHologram_0.65 \
  --wavelength 532e-9 \
  --pitch 4e-6 \
  --zmin 0.050 --zmax 0.056 --step 0.0001 \
  --gamma 0.65 \
  --out_size 2048

'''
import os
import argparse
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
import cv2

# =========================
# ASM propagation
# =========================
def asm_propagate(u_in: np.ndarray, dist_m: float, wavelength_m: float, pitch_m: float) -> np.ndarray:
    """
    Angular Spectrum Method propagation with evanescent wave masking.
    u_in: complex field (H,W)
    dist_m: propagation distance in meters (+ forward)
    """
    u_in = np.asarray(u_in, dtype=np.complex64)
    H, W = u_in.shape

    dfx = 1.0 / (H * pitch_m)
    dfy = 1.0 / (W * pitch_m)
    fx = (np.arange(H) - H / 2) * dfx
    fy = (np.arange(W) - W / 2) * dfy
    FX, FY = np.meshgrid(fx, fy, indexing="ij")

    k = 2.0 * np.pi / wavelength_m
    term = 1.0 - (wavelength_m * FX) ** 2 - (wavelength_m * FY) ** 2

    Htf = np.zeros_like(term, dtype=np.complex64)
    mask = term >= 0
    Htf[mask] = np.exp(1j * k * dist_m * np.sqrt(term[mask]).astype(np.float32))

    U_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_in)))
    U_z = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U_fft * Htf)))
    return U_z


# =========================
# IO: load amplitude / phase
# =========================
def load_phase(path: str) -> np.ndarray:
    phase = np.load(path)
    if phase.ndim != 2:
        raise ValueError(f"phase.npy must be 2D, got {phase.shape}")
    return phase.astype(np.float32)


def load_amp(path: str) -> np.ndarray:
    amp = np.load(path)
    if amp.ndim != 2:
        raise ValueError(f"amp.npy must be 2D, got {amp.shape}")
    amp = amp.astype(np.float32)
    return np.clip(amp, 0.0, None)


def load_complex_hologram(amp_path: str, phase_path: str) -> np.ndarray:
    amp = load_amp(amp_path)
    phase = load_phase(phase_path)
    if amp.shape != phase.shape:
        raise ValueError("amp / phase shape mismatch")
    return (amp * np.exp(1j * phase)).astype(np.complex64)


# =========================
# Display: global RGB normalize + gamma
# =========================
# def normalize_rgb_global_with_gamma(rgb_float: np.ndarray, gamma: float = 1.0) -> np.ndarray:
#     x = np.asarray(rgb_float, dtype=np.float32)
#     x = np.clip(x, 0.0, None)

#     x = x - x.min()
#     maxv = float(x.max())
#     if maxv > 1e-8:
#         x = x / maxv

#     if gamma is not None and abs(gamma - 1.0) > 1e-12:
#         x = np.power(np.clip(x, 0.0, 1.0), float(gamma))

#     return (x * 255.0).clip(0, 255).astype(np.uint8)

def normalize_rgb_global_with_gamma(
    rgb_float: np.ndarray,
    gamma: float = 1.0,
    p_low: float = 1.0,
    p_high: float = 99.0
) -> np.ndarray:
    x = np.asarray(rgb_float, dtype=np.float32)
    x = np.clip(x, 0.0, None)

    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    x = np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)

    if gamma is not None and abs(gamma - 1.0) > 1e-12:
        x = np.power(x, float(gamma))

    return (x * 255.0).clip(0, 255).astype(np.uint8)


# =========================
# NEW: resize to 2048×2048 (keep aspect, pad)
# =========================
def resize_keep_aspect_to_square(img_u8: np.ndarray, target: int = 2048, pad_value: int = 0) -> np.ndarray:
    """
    img_u8: (H,W,3) uint8
    return: (target,target,3) uint8
    """
    h, w, c = img_u8.shape
    scale = target / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img_u8, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out = np.full((target, target, 3), pad_value, dtype=np.uint8)
    y0 = (target - new_h) // 2
    x0 = (target - new_w) // 2
    out[y0:y0 + new_h, x0:x0 + new_w, :] = resized
    return out


# =========================
# Parse z list
# =========================
def parse_z_list(args):
    if args.z_list.strip():
        return [float(s) for s in args.z_list.split(",") if s.strip()]

    zmin, zmax, step = args.zmin, args.zmax, args.step
    n = int(np.floor((zmax - zmin) / step + 1e-12)) + 1
    return [zmin + i * step for i in range(n)]


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="ASM RGB reconstruction from COMPLEX hologram (amp.npy + phase.npy), output 2048x2048."
    )

    parser.add_argument("--amp_r", required=True)
    parser.add_argument("--pha_r", required=True)
    parser.add_argument("--amp_g", required=True)
    parser.add_argument("--pha_g", required=True)
    parser.add_argument("--amp_b", required=True)
    parser.add_argument("--pha_b", required=True)

    parser.add_argument("--outdir", required=True)

    parser.add_argument("--wavelength", type=float, default=532e-9)
    parser.add_argument("--pitch", type=float, default=2e-6)

    parser.add_argument("--z_list", type=str, default="")
    parser.add_argument("--zmin", type=float, default=-0.053)
    parser.add_argument("--zmax", type=float, default=-0.050)
    parser.add_argument("--step", type=float, default=1e-4)

    parser.add_argument("--use_intensity", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0)

    parser.add_argument("--out_size", type=int, default=2048)

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("[INFO] Load complex holograms ...")
    holo_R = load_complex_hologram(args.amp_r, args.pha_r)
    holo_G = load_complex_hologram(args.amp_g, args.pha_g)
    holo_B = load_complex_hologram(args.amp_b, args.pha_b)

    z_list = parse_z_list(args)
    print(f"[INFO] z count = {len(z_list)}")

    for z in tqdm(z_list, desc="ASM Recon (RGB)"):
        rec_R = asm_propagate(holo_R, z, args.wavelength, args.pitch)
        rec_G = asm_propagate(holo_G, z, args.wavelength, args.pitch)
        rec_B = asm_propagate(holo_B, z, args.wavelength, args.pitch)

        if args.use_intensity:
            img_R = np.abs(rec_R) ** 2
            img_G = np.abs(rec_G) ** 2
            img_B = np.abs(rec_B) ** 2
        else:
            img_R = np.abs(rec_R)
            img_G = np.abs(rec_G)
            img_B = np.abs(rec_B)

        rgb = np.stack([img_R, img_G, img_B], axis=-1).astype(np.float32)
        rgb_u8 = normalize_rgb_global_with_gamma(rgb, gamma=args.gamma)

        # ⭐ NEW: resize to 2048×2048
        rgb_u8 = resize_keep_aspect_to_square(rgb_u8, target=args.out_size)

        out_path = os.path.join(args.outdir, f"recon_rgb_z_{z:+.6f}m.png")
        imageio.imwrite(out_path, rgb_u8)

    print("[DONE] Output:", args.outdir)


if __name__ == "__main__":
    main()
