# Light-Field Data: From an OBJ Mesh to a Complex Hologram

This repository supports two light-field pipelines:

```text
OBJ mesh
  ├── orthographic light field
  │     └── orthographic LF hologram
  └── perspective light field
        └── refocused perspective LF hologram
```

Run all commands from the repository root.

## 1. Files Used

| Pipeline | Data generation | Hologram generation |
| --- | --- | --- |
| Orthographic | [`dataset/Obj2LF_orth.py`](../dataset/Obj2LF_orth.py) | [`model/LightField_Orth_ComplexHologram.py`](../model/LightField_Orth_ComplexHologram.py) |
| Perspective | [`dataset/Obj2LF_pers.py`](../dataset/Obj2LF_pers.py) | [`model/LightField_Pers_ComplexHologram.py`](../model/LightField_Pers_ComplexHologram.py) |

Both data generators save:

```text
output_dir/
├── images/
│   ├── u00_v00.png
│   ├── u00_v01.png
│   └── ...
├── poses.csv
├── intrinsics.csv
└── time.csv
```

## 2. Install Dependencies

```bash
python -m pip install numpy imageio open3d opencv-python tqdm
```

Check:

```bash
python -c "import numpy, imageio, open3d, cv2, tqdm; print('Dependencies OK')"
python -m py_compile \
    dataset/Obj2LF_orth.py \
    dataset/Obj2LF_pers.py \
    model/LightField_Orth_ComplexHologram.py \
    model/LightField_Pers_ComplexHologram.py
```

## 3. Choose a Light-Field Model

Use the orthographic pipeline when rays from different views should preserve
an orthographic projection.

Use the perspective pipeline when each sub-aperture image should use a
perspective camera. The cameras are translated over a regular grid while
their optical axes remain parallel.

The perspective hologram script additionally refocuses the views to a
reference depth before constructing the complex field.

## 4. Memory Warning

The LF-to-complex conversion produces a tiled field:

```text
field height = nv × image height
field width  = nu × image width
```

Examples:

| LF configuration | Complex field |
| --- | --- |
| `4×4` views, `128×128` pixels | `512×512` |
| `10×10` views, `256×256` pixels | `2560×2560` |
| `20×20` views, `400×400` pixels | `8000×8000` |
| `40×40` views, `800×800` pixels | `32000×32000` |

A `32000×32000` complex64 array alone is approximately 7.6 GiB, before FFT
workspaces and RGB channels. Always start with a small test.

## 5. Orthographic Pipeline

### 5.1 Generate a Small Orthographic Light Field

```bash
python ./dataset/Obj2LF_orth.py \
    --input_obj ./dataset/BunnyDragon.obj \
    --output_dir ./dataset/LightField/LF_orth_test \
    --nu 4 \
    --nv 4 \
    --img_width 128 \
    --img_height 128 \
    --fov_deg 7 \
    --target_z_min 50.0 \
    --target_z_max 53.0 \
    --eye_pos 0.0 50.0 0.0 \
    --center_pos 0.0 0.0 0.0 \
    --up_dir 0.0 0.0 1.0 \
    --total_baseline_x 4.0 \
    --total_baseline_z 4.0 \
    --look_at_dist 200.0
```

Inspect the center and corner views:

```bash
ls ./dataset/LightField/LF_orth_test/images
```

The object should remain visible across the angular grid.

### 5.2 Generate an Orthographic Hologram

```bash
python ./model/LightField_Orth_ComplexHologram.py \
    --input_dir ./dataset/LightField/LF_orth_test/images \
    --output_dir ./result/LightField/LF_orth_test \
    --nu 4 \
    --nv 4 \
    --ns 128 \
    --nt 128 \
    --dx 2e-6 \
    --dy 2e-6 \
    --wavelength 532e-9 \
    --distance_rs_to_h 0.05 \
    --random_seed 2026 \
    --vis_mode percentile \
    --p_low 1.0 \
    --p_high 99.0 \
    --gamma 1.0 \
    --z_list -0.048 -0.050 -0.052
```

The four LF dimensions must match the generated dataset:

```text
nu = angular columns
nv = angular rows
ns = image width
nt = image height
```

The current orthographic ASM implementation requires `dx == dy`.

### 5.3 Larger Orthographic Example

```bash
python ./dataset/Obj2LF_orth.py \
    --input_obj ./dataset/BunnyDragon.obj \
    --output_dir ./dataset/LightField/LF_orth_20x20_400 \
    --nu 20 --nv 20 \
    --img_width 400 --img_height 400 \
    --fov_deg 7 \
    --target_z_min 50.0 --target_z_max 53.0 \
    --eye_pos 0.0 50.0 0.0 \
    --center_pos 0.0 0.0 0.0 \
    --up_dir 0.0 0.0 1.0 \
    --total_baseline_x 4.0 \
    --total_baseline_z 4.0 \
    --look_at_dist 200.0

python ./model/LightField_Orth_ComplexHologram.py \
    --input_dir ./dataset/LightField/LF_orth_20x20_400/images \
    --output_dir ./result/LightField/LF_orth_20x20_400 \
    --nu 20 --nv 20 --ns 400 --nt 400 \
    --dx 2e-6 --dy 2e-6 \
    --wavelength 532e-9 \
    --distance_rs_to_h 0.05 \
    --z_list -0.048 -0.050 -0.052
```

Use this only after confirming sufficient RAM.

## 6. Perspective Pipeline

### 6.1 Generate a Small Perspective Light Field

```bash
python ./dataset/Obj2LF_pers.py \
    --input_obj ./dataset/BunnyDragon.obj \
    --output_dir ./dataset/LightField/LF_pers_test \
    --img_width 128 \
    --img_height 128 \
    --fov_deg 7 \
    --nu 4 \
    --nv 4 \
    --total_baseline_x 4.0 \
    --total_baseline_z 4.0 \
    --target_z_min 50.0 \
    --target_z_max 53.0 \
    --eye_pos 0.0 50.0 0.0 \
    --center_pos 0.0 0.0 0.0 \
    --up_dir 0.0 0.0 1.0 \
    --look_at_dist 200.0
```

The baseline and generated `poses.csv` offsets are in millimeters.

### 6.2 Generate a Perspective Hologram

```bash
python ./model/LightField_Pers_ComplexHologram.py \
    --lf_root ./dataset/LightField/LF_pers_test \
    --output_dir ./result/LightField/LF_pers_test \
    --fov_deg 7 \
    --d0 -0.050 \
    --pose_unit mm \
    --wavelength 532e-9 \
    --pitch 2e-6 \
    --dist_rs_to_h 0.05 \
    --gamma 0.65 \
    --out_size 512 \
    --vis_mode percentile \
    --save_holo_png \
    --save_recon_png \
    --p_low 1.0 \
    --p_high 99.0 \
    --z_min -0.052 \
    --z_max -0.048 \
    --z_step 0.001
```

Critical consistency rules:

- Hologram `--fov_deg` must equal the data-generation FOV.
- `Obj2LF_pers.py` writes `dx` and `dz` in millimeters, so use
  `--pose_unit mm`.
- `d0`, `dist_rs_to_h`, and reconstruction Z values are in meters.

### 6.3 Larger Perspective Example

```bash
python ./dataset/Obj2LF_pers.py \
    --input_obj ./dataset/BunnyDragon.obj \
    --output_dir ./dataset/LightField/LF_pers_20x20_400 \
    --img_width 400 --img_height 400 \
    --fov_deg 7 \
    --nu 20 --nv 20 \
    --total_baseline_x 4.0 \
    --total_baseline_z 4.0 \
    --target_z_min 50.0 \
    --target_z_max 53.0

python ./model/LightField_Pers_ComplexHologram.py \
    --lf_root ./dataset/LightField/LF_pers_20x20_400 \
    --output_dir ./result/LightField/LF_pers_20x20_400 \
    --fov_deg 7 \
    --d0 -0.050 \
    --pose_unit mm \
    --wavelength 532e-9 \
    --pitch 2e-6 \
    --dist_rs_to_h 0.05 \
    --save_holo_png \
    --save_recon_png \
    --z_min -0.052 \
    --z_max -0.048 \
    --z_step 0.001
```

## 7. Output Structure

Both hologram scripts create:

```text
result/LightField/example/
├── Hologram/
│   ├── hologram_complex_R.npy
│   ├── hologram_complex_G.npy
│   ├── hologram_complex_B.npy
│   ├── hologram_amp_R.npy
│   ├── hologram_amp_G.npy
│   ├── hologram_amp_B.npy
│   ├── hologram_phase_R.npy
│   ├── hologram_phase_G.npy
│   ├── hologram_phase_B.npy
│   └── optional PNG previews
└── Recon/
    └── reconstructed RGB images
```

## 8. Common Problems

### Missing View Image

Check that every `(u, v)` pair exists and uses the expected
`uXX_vXX.png` name.

### Shape Mismatch

For orthographic LF, `nu`, `nv`, `ns`, and `nt` must exactly match the
dataset. For perspective LF, all images must have identical dimensions.

### Perspective Views Do Not Align

Ensure the FOV matches and use `--pose_unit mm`. Adjust `d0` near the object
depth; for a 50 mm object distance, start with `-0.050`.

### Out of Memory

Reduce angular samples and image size. The product of angular and spatial
dimensions determines the complex-field size.

### Wrong Reconstruction Sign

This pipeline commonly uses negative reconstruction distances because of its
RS-plane/hologram-plane convention. Test values around `-0.050 m`.

## 9. Recommended Debug Sequence

1. Generate `4×4` views at `128×128`.
2. Inspect all generated images.
3. Generate the hologram.
4. Confirm amplitude and phase files exist.
5. Reconstruct three distances.
6. Increase to `10×10`, then `20×20` only if memory permits.
