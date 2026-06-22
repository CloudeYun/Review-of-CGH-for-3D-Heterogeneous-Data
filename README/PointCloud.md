# Point-Cloud Data: From an OBJ Mesh to a Complex Hologram

This guide describes the complete point-cloud pipeline:

```text
OBJ mesh
  → colored PLY point cloud
  → sparse RGB depth bins
  → complex RGB holograms
  → depth reconstructions
```

Run every command from the repository root.

## 1. Files Used

| File | Purpose |
| --- | --- |
| [`dataset/Obj2PCD.py`](../dataset/Obj2PCD.py) | Samples a colored PLY point cloud from an OBJ mesh. |
| [`model/PointCloud_ComplexHologram.py`](../model/PointCloud_ComplexHologram.py) | Rasterizes the point cloud, generates holograms, and reconstructs depth bins. |
| [`model/Reconstruction_layer.py`](../model/Reconstruction_layer.py) | Optionally reconstructs saved holograms at custom distances. |

## 2. Install Dependencies

```bash
python -m pip install numpy open3d opencv-python tqdm imageio
```

Check the scripts:

```bash
python -c "import numpy, open3d, cv2, tqdm; print('Dependencies OK')"
python -m py_compile dataset/Obj2PCD.py model/PointCloud_ComplexHologram.py
```

## 3. Prepare the Mesh

This guide assumes:

```text
dataset/BunnyDragon.obj
```

Check it:

```bash
ls -lh ./dataset/BunnyDragon.obj
```

The OBJ must contain triangle faces. Material IDs are used for coloring when
available; otherwise, the sampler uses a default gray color.

## 4. Step 1: Generate a Colored Point Cloud

Start with a small test:

```bash
python ./dataset/Obj2PCD.py \
    --input_obj ./dataset/BunnyDragon.obj \
    --output_ply ./dataset/PointCloud/bunnydragon_test.ply \
    --num_points 100000 \
    --seed 2024 \
    --center_mesh \
    --use_shading \
    --eye_pos 0.0 1.5 0.0
```

If that succeeds, generate a denser point cloud:

```bash
python ./dataset/Obj2PCD.py \
    --input_obj ./dataset/BunnyDragon.obj \
    --output_ply ./dataset/PointCloud/bunnydragon_10m.ply \
    --num_points 10000000 \
    --seed 2024 \
    --center_mesh \
    --use_shading \
    --eye_pos 0.0 1.5 0.0
```

Important options:

- `--num_points`: point count. More points reduce holes after rasterization.
- `--seed`: makes sampling reproducible.
- `--center_mesh`: centers the mesh before sampling.
- `--use_shading`: applies view-dependent shading to point colors.
- `--write_ascii`: writes a readable but much larger PLY file.
- `--compressed`: requests compressed output.

For large datasets, use the default binary PLY output. Extremely large point
counts require substantial RAM and disk space. Increase the count gradually.

Inspect the file:

```bash
ls -lh ./dataset/PointCloud/bunnydragon_10m.ply

python - <<'PY'
import open3d as o3d
pcd = o3d.io.read_point_cloud("./dataset/PointCloud/bunnydragon_10m.ply")
print("points:", len(pcd.points))
print("has colors:", pcd.has_colors())
print("bounds:", pcd.get_min_bound(), pcd.get_max_bound())
PY
```

Do not continue if the point count is zero.

## 5. Step 2: Generate the Complex Hologram

First use a low-resolution test:

```bash
python ./model/PointCloud_ComplexHologram.py \
    --data_path ./dataset/PointCloud/bunnydragon_test.ply \
    --out_root ./result/PointCloud/bunnydragon_test \
    --target_size 512 \
    --out_size 512 \
    --pixel_pitch 5e-6 \
    --wavelength 532e-9 \
    --z_min_mm 50.0 \
    --z_max_mm 53.0 \
    --num_bins 20 \
    --fit_margin 0.90 \
    --rng_seed 2024 \
    --gamma 0.65 \
    --vis_mode percentile \
    --p_low 1.0 \
    --p_high 99.0 \
    --rot_pitch_deg 90 \
    --rotate_around_center \
    --depth_flip
```

Then run the full-size version:

```bash
python ./model/PointCloud_ComplexHologram.py \
    --data_path ./dataset/PointCloud/bunnydragon_10m.ply \
    --out_root ./result/PointCloud/bunnydragon_10m \
    --target_size 2048 \
    --out_size 2048 \
    --pixel_pitch 5e-6 \
    --wavelength 532e-9 \
    --z_min_mm 50.0 \
    --z_max_mm 53.0 \
    --num_bins 100 \
    --fit_margin 0.90 \
    --rng_seed 2024 \
    --gamma 0.65 \
    --vis_mode percentile \
    --p_low 1.0 \
    --p_high 99.0 \
    --rot_pitch_deg 90 \
    --rotate_around_center \
    --depth_flip
```

The script normalizes the point-cloud depth to the range
`z_min_mm–z_max_mm`. These values are in millimeters. Optical wavelength and
pixel pitch are in meters.

Orientation options:

- `--rot_yaw_deg`: rotation around Z.
- `--rot_pitch_deg`: rotation around X.
- `--rot_roll_deg`: rotation around Y.
- `--rotate_around_center`: rotate around the cloud center.
- `--depth_flip`: reverse the normalized depth direction.

If the reconstructed object is upside down or front/back reversed, first
adjust these options using the 512 × 512 test.

## 6. Output Structure

```text
result/PointCloud/bunnydragon_10m/
├── Hologram/
│   ├── holo_R.npy
│   ├── holo_G.npy
│   ├── holo_B.npy
│   ├── amp_R.npy
│   ├── amp_G.npy
│   ├── amp_B.npy
│   ├── phase_R.npy
│   ├── phase_G.npy
│   ├── phase_B.npy
│   └── amplitude/phase PNG previews
└── Recon/
    └── bins/
        └── reconstructed RGB images
```

The `.npy` files contain numerical hologram data. PNG files are visualization
previews only.

## 7. Optional Custom Reconstruction

```bash
python ./model/Reconstruction_layer.py \
    --amp_r ./result/PointCloud/bunnydragon_10m/Hologram/amp_R.npy \
    --pha_r ./result/PointCloud/bunnydragon_10m/Hologram/phase_R.npy \
    --amp_g ./result/PointCloud/bunnydragon_10m/Hologram/amp_G.npy \
    --pha_g ./result/PointCloud/bunnydragon_10m/Hologram/phase_G.npy \
    --amp_b ./result/PointCloud/bunnydragon_10m/Hologram/amp_B.npy \
    --pha_b ./result/PointCloud/bunnydragon_10m/Hologram/phase_B.npy \
    --outdir ./result/PointCloud/bunnydragon_10m/Reconstruction_custom \
    --mode asm \
    --wavelength 532e-9 \
    --pitch 5e-6 \
    --zmin 0.050 \
    --zmax 0.053 \
    --step 0.0001 \
    --gamma 0.65 \
    --vis_mode percentile \
    --out_size 2048
```

The reconstruction distances are in meters: `0.050 m = 50 mm`.

## 8. Common Problems

### Sparse or Broken Reconstruction

Increase `--num_points`, reduce `--target_size`, or increase
`--fit_margin` slightly. Point-cloud rasterization keeps the front-most point
at each pixel, so insufficient point density creates holes.

### No Color

Confirm `pcd.has_colors()` is true. If the OBJ has no usable material IDs,
the point cloud will use the fallback color.

### Wrong Orientation

Change the Euler rotation or toggle `--depth_flip`. Always diagnose this with
a small point cloud and 512 × 512 hologram.

### Out of Memory

Reduce `--num_points` during point-cloud generation and reduce
`--target_size` during hologram generation.

## 9. Minimal Workflow

```bash
python ./dataset/Obj2PCD.py \
    --input_obj ./dataset/BunnyDragon.obj \
    --output_ply ./dataset/PointCloud/bunnydragon.ply \
    --num_points 10000000 \
    --seed 2024 \
    --center_mesh \
    --use_shading

python ./model/PointCloud_ComplexHologram.py \
    --data_path ./dataset/PointCloud/bunnydragon.ply \
    --out_root ./result/PointCloud/bunnydragon \
    --target_size 2048 \
    --out_size 2048 \
    --pixel_pitch 5e-6 \
    --wavelength 532e-9 \
    --z_min_mm 50.0 \
    --z_max_mm 53.0 \
    --num_bins 100 \
    --fit_margin 0.90 \
    --rotate_around_center \
    --rot_pitch_deg 90 \
    --depth_flip
```
