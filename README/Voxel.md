# Voxel Data: From an OBJ Mesh to a Complex Hologram

This guide covers:

```text
OBJ mesh
  → occupancy voxel grid (.npz)
  → differential volume layers
  → complex RGB holograms
  → reconstructed depth slices
```

Run all commands from the repository root.

## 1. Files Used

| File | Purpose |
| --- | --- |
| [`dataset/Obj2Vox.py`](../dataset/Obj2Vox.py) | Converts an OBJ mesh into an occupancy voxel grid. |
| [`model/Voxel_ComplexHologram.py`](../model/Voxel_ComplexHologram.py) | Generates a hologram using voxel-volume integration. |
| [`model/Reconstruction_layer.py`](../model/Reconstruction_layer.py) | Optionally reconstructs saved holograms. |

## 2. Install Dependencies

```bash
python -m pip install numpy trimesh scipy matplotlib scikit-image opencv-python tqdm
```

Check:

```bash
python -c "import numpy, trimesh, scipy, matplotlib, skimage, cv2; print('Dependencies OK')"
python -m py_compile dataset/Obj2Vox.py model/Voxel_ComplexHologram.py
```

## 3. Step 1: Generate the Voxel Grid

Start with a coarse grid:

```bash
python ./dataset/Obj2Vox.py \
    --obj ./dataset/BunnyDragon.obj \
    --out ./dataset/Voxel/BunnyDragon_test.npz \
    --pitch 0.10 \
    --solid \
    --post_fill \
    --close_iter 2
```

Then use a finer pitch:

```bash
python ./dataset/Obj2Vox.py \
    --obj ./dataset/BunnyDragon.obj \
    --out ./dataset/Voxel/BunnyDragon_voxel.npz \
    --pitch 0.05 \
    --solid \
    --force_watertight \
    --post_fill \
    --close_iter 3 \
    --rot_x 0 \
    --rot_y 0 \
    --rot_z 0 \
    --rot_about center
```

`--pitch` is expressed in the same coordinate unit as the OBJ:

- Larger pitch: fewer voxels, faster, lower detail.
- Smaller pitch: more voxels, slower, higher detail and memory use.

The recommended `.npz` output contains:

```text
occ         occupancy in (Z, Y, X) order
pitch       voxel size
origin_xyz  voxel-grid origin
```

Inspect it:

```bash
python - <<'PY'
import numpy as np
data = np.load("./dataset/Voxel/BunnyDragon_voxel.npz")
occ = data["occ"]
print("keys:", data.files)
print("shape (Z,Y,X):", occ.shape)
print("occupied voxels:", int((occ > 0).sum()))
print("occupancy ratio:", float((occ > 0).mean()))
print("pitch:", float(data["pitch"]))
print("origin:", data["origin_xyz"])
PY
```

Do not continue if the occupied-voxel count is zero.

## 4. Optional Voxel Visualization

Generate an isosurface preview:

```bash
python ./dataset/Obj2Vox.py \
    --obj ./dataset/BunnyDragon.obj \
    --out ./dataset/Voxel/BunnyDragon_voxel.npz \
    --pitch 0.05 \
    --solid \
    --post_fill \
    --close_iter 3 \
    --vis \
    --vis_dir ./dataset/Voxel/vis \
    --vis_mode isosurface \
    --vis_iso_sigma 0.15 \
    --vis_iso_level 0.33 \
    --vis_iso_max_dim 400 \
    --only_view \
    --no_axes \
    --view_elev 20 \
    --view_azim -90
```

Use `--vis_mode scatter` for an occupied-voxel scatter plot.

## 5. Step 2: Generate the Voxel Hologram

Start with a low-resolution test:

```bash
python ./model/Voxel_ComplexHologram.py \
    --npz_path ./dataset/Voxel/BunnyDragon_test.npz \
    --out_root ./result/Voxel/BunnyDragon_test \
    --target_height 512 \
    --target_width 512 \
    --z_min_mm 50.0 \
    --z_max_mm 53.0 \
    --d_percentage 5.0 \
    --bin_axis_rotated y \
    --view_elev 25 \
    --view_azim -90 \
    --pixel_pitch 4e-6 \
    --out_size 512
```

Then run the full-size configuration:

```bash
python ./model/Voxel_ComplexHologram.py \
    --npz_path ./dataset/Voxel/BunnyDragon_voxel.npz \
    --out_root ./result/Voxel/BunnyDragon \
    --target_height 2048 \
    --target_width 2048 \
    --z_min_mm 50.0 \
    --z_max_mm 53.0 \
    --d_percentage 1.0 \
    --bin_axis_rotated y \
    --bin_reverse false \
    --view_elev 25 \
    --view_azim -90 \
    --camera_flip_ud true \
    --camera_flip_lr false \
    --iso_sigma 0.15 \
    --wavelength_r 638e-9 \
    --wavelength_g 520e-9 \
    --wavelength_b 450e-9 \
    --pixel_pitch 4e-6 \
    --holo_z_m 0.0 \
    --rng_seed 2024 \
    --gamma 0.65 \
    --out_size 2048
```

### Understanding `d_percentage`

`d_percentage` is the thickness of one differential volume layer as a
percentage of the complete slicing axis:

| Value | Approximate layer count |
| ---: | ---: |
| `10.0` | 10 |
| `5.0` | 20 |
| `2.0` | 50 |
| `1.0` | 100 |
| `0.5` | 200 |

Smaller values provide finer depth discretization but increase runtime.

The script integrates occupancy inside each differential slab, projects it
along the camera optical axis, adds random phase, and propagates the resulting
2D field to the hologram plane.

The current occupancy input has no per-voxel color. Therefore, RGB channels
share the same object amplitude and differ through their wavelengths.

## 6. Orientation Controls

- `--bin_axis_rotated`: slicing axis after object rotation.
- `--bin_reverse`: swaps low/high physical-depth assignment.
- `--rot_yaw_deg`, `--rot_pitch_deg`, `--rot_roll_deg`: object rotation.
- `--view_elev`, `--view_azim`: camera projection direction.
- `--camera_flip_ud`, `--camera_flip_lr`: final image-coordinate correction.

Use 512 × 512 and `d_percentage=5` while adjusting orientation.

## 7. Output Structure

```text
result/Voxel/BunnyDragon/
├── Binned/
│   ├── full_volume_view.png
│   ├── meta.txt
│   └── plane_*/
│       ├── rgb.png
│       ├── mask.png
│       └── depth_mm.txt
├── Hologram/
│   ├── holo_R.npy
│   ├── holo_G.npy
│   ├── holo_B.npy
│   ├── amp_R.npy
│   ├── amp_G.npy
│   ├── amp_B.npy
│   ├── pha_R.npy
│   ├── pha_G.npy
│   ├── pha_B.npy
│   └── amplitude/phase PNG previews
└── Recon/
    └── Recon_plane*.png
```

Check `Binned/full_volume_view.png` first. If its orientation is wrong, fix
the rotation/view parameters before interpreting the hologram.

## 8. Common Problems

### Voxel Grid Is Too Large

Increase `Obj2Vox.py --pitch`. Halving the pitch can increase voxel count and
memory dramatically in three dimensions.

### Interior Is Hollow

Use:

```text
--solid --force_watertight --post_fill --close_iter 3
```

Mesh repair and filling cannot fix every invalid mesh, so inspect the
isosurface preview.

### Hologram Generation Is Slow

Increase `--d_percentage`, reduce target resolution, or use a coarser voxel
pitch.

### Wrong Depth Direction

Toggle `--bin_reverse true` and inspect the reconstruction sequence.

## 9. Minimal Workflow

```bash
python ./dataset/Obj2Vox.py \
    --obj ./dataset/BunnyDragon.obj \
    --out ./dataset/Voxel/BunnyDragon_voxel.npz \
    --pitch 0.05 \
    --solid \
    --post_fill \
    --close_iter 3

python ./model/Voxel_ComplexHologram.py \
    --npz_path ./dataset/Voxel/BunnyDragon_voxel.npz \
    --out_root ./result/Voxel/BunnyDragon \
    --target_height 2048 \
    --target_width 2048 \
    --z_min_mm 50.0 \
    --z_max_mm 53.0 \
    --d_percentage 1.0 \
    --pixel_pitch 4e-6 \
    --out_size 2048
```
