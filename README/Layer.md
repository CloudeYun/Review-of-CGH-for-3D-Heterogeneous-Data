# Layered Data: From a 3D Mesh to a Complex Hologram

This guide explains the complete layered-data pipeline in this repository:

```text
3D mesh (.obj)
    │
    ├── dataset/Obj2Layer.py
    ▼
Layered RGB-D data
    │
    ├── model/Layer_ComplexHologram.py
    ▼
Complex RGB holograms + depth reconstructions
```

The instructions below are intentionally explicit. All commands should be
executed from the repository root.

## 1. What This Pipeline Does

The layered-data pipeline represents a 3D object as one or more RGB and depth
image pairs:

```text
rgb_layer1.png  +  depth_layer1.npy
rgb_layer2.png  +  depth_layer2.npy
...
```

The number of layers determines the representation:

- `--ldi_layers 1`: generate a standard RGB-D representation.
- `--ldi_layers 2` or greater: generate a Layered Depth Image (LDI).

The hologram generator then:

1. Divides the physical depth range into discrete depth bins.
2. Places RGB pixels into the corresponding depth bins.
3. Converts each depth bin into a complex object field.
4. Propagates every depth field to the hologram plane using the Angular
   Spectrum Method (ASM).
5. Adds all propagated fields to obtain the final RGB complex holograms.
6. Reconstructs the holograms at the depth-bin centers.

## 2. Repository Files Used

| File | Purpose |
| --- | --- |
| [`dataset/Obj2Layer.py`](../dataset/Obj2Layer.py) | Generates RGB-D or LDI layers from an OBJ mesh. |
| [`model/Layer_ComplexHologram.py`](../model/Layer_ComplexHologram.py) | Generates RGB complex holograms and depth reconstructions. |
| [`model/Reconstruction_layer.py`](../model/Reconstruction_layer.py) | Optionally reconstructs existing holograms at custom distances. |

## 3. Environment Setup

Create or activate a Python environment, then install the required packages:

```bash
python -m pip install numpy imageio open3d opencv-python tqdm
```

Confirm that the current directory is the repository root:

```bash
pwd
ls dataset/Obj2Layer.py model/Layer_ComplexHologram.py
```

Verify that the three scripts can be imported:

```bash
python -c "import numpy, imageio, open3d, cv2, tqdm; print('Dependencies OK')"

python -m py_compile \
    dataset/Obj2Layer.py \
    model/Layer_ComplexHologram.py \
    model/Reconstruction_layer.py
```

If both commands finish without an error message, the required Python
packages are importable and the scripts pass a syntax check.

## 4. Prepare the Input Mesh

Place the input OBJ file inside the repository. This guide uses:

```text
dataset/BunnyDragon.obj
```

The mesh must:

- Contain valid vertices and triangle faces.
- Be readable by Open3D.
- Have a nonzero extent along its Y axis.

The Y-axis thickness is important because `Obj2Layer.py` maps that thickness
to the requested physical depth range.

Texture images are not required. The current script assigns colors according
to mesh material IDs. If material IDs are unavailable, it uses a default gray
color.

Check that the file exists:

```bash
ls -lh ./dataset/BunnyDragon.obj
```

## 5. Step 1: Generate RGB-D or LDI Data

### 5.1 Recommended First Test

Start with a small image and one layer. This is much faster and helps verify
the camera orientation before a full-resolution run:

```bash
python ./dataset/Obj2Layer.py \
    --input_obj ./dataset/BunnyDragon.obj \
    --output_dir ./dataset/Layer/BunnyDragon_RGBD_test \
    --img_width 512 \
    --img_height 512 \
    --fov_deg 1.0 \
    --ldi_layers 1 \
    --target_z_min 50.0 \
    --target_z_max 53.0 \
    --bg_depth 100.0
```

Expected output:

```text
dataset/Layer/BunnyDragon_RGBD_test/
├── rgb_layer1.png
├── depth_layer1.npy
└── depth_layer1.png
```

Open the two PNG files and check:

- `rgb_layer1.png` should contain the object on the selected background.
- `depth_layer1.png` should show visible depth variation.
- The object should not be missing, extremely small, or cut off.

The PNG depth image is only a visualization. The hologram generator uses the
floating-point depth values in `depth_layer1.npy`.

### 5.2 Full-Resolution RGB-D Generation

After the test succeeds, generate a 2048 × 2048 RGB-D dataset:

```bash
python ./dataset/Obj2Layer.py \
    --input_obj ./dataset/BunnyDragon.obj \
    --output_dir ./dataset/Layer/BunnyDragon_RGBD \
    --img_width 2048 \
    --img_height 2048 \
    --fov_deg 1.0 \
    --ldi_layers 1 \
    --target_z_min 50.0 \
    --target_z_max 53.0 \
    --bg_depth 100.0
```

### 5.3 Generate an LDI Instead of RGB-D

To preserve multiple intersections along each camera ray, increase
`--ldi_layers`. For example:

```bash
python ./dataset/Obj2Layer.py \
    --input_obj ./dataset/BunnyDragon.obj \
    --output_dir ./dataset/Layer/BunnyDragon_LDI \
    --img_width 2048 \
    --img_height 2048 \
    --fov_deg 1.0 \
    --ldi_layers 4 \
    --eps_shift 1e-4 \
    --target_z_min 50.0 \
    --target_z_max 53.0 \
    --bg_depth 100.0
```

This produces:

```text
dataset/Layer/BunnyDragon_LDI/
├── rgb_layer1.png
├── depth_layer1.npy
├── depth_layer1.png
├── rgb_layer2.png
├── depth_layer2.npy
├── depth_layer2.png
├── ...
├── rgb_layer4.png
├── depth_layer4.npy
└── depth_layer4.png
```

Each layer records the next ray-surface intersection. A higher layer count
can preserve more occluded geometry, but it also increases generation time,
storage, and later preprocessing work.

## 6. Check the Generated Depth Values

Before generating a hologram, verify the numerical depth range:

```bash
python - <<'PY'
import glob
import numpy as np

for path in sorted(glob.glob("./dataset/Layer/BunnyDragon_RGBD/depth_layer*.npy")):
    depth = np.load(path)
    valid = depth < 99.5
    print(
        path,
        "shape=", depth.shape,
        "valid_pixels=", int(valid.sum()),
        "valid_min_mm=", float(depth[valid].min()) if valid.any() else None,
        "valid_max_mm=", float(depth[valid].max()) if valid.any() else None,
    )
PY
```

For the example configuration, valid object depths should be approximately
between 50 mm and 53 mm. Background pixels should be 100 mm.

Do not continue if every pixel is 100 mm. That means the camera rays did not
intersect the mesh.

## 7. Step 2: Generate the Complex Hologram

The same command works for both RGB-D and LDI input. The script automatically
finds all matching `rgb_layerN.png` and `depth_layerN.npy` pairs.

### 7.1 Quick Hologram Test

Use a smaller target size and fewer depth bins first:

```bash
python ./model/Layer_ComplexHologram.py \
    --data_root ./dataset/Layer/BunnyDragon_RGBD_test \
    --out_root ./result/Layer/BunnyDragon_RGBD_test \
    --target_size 512 \
    --z_min_mm 50.0 \
    --z_max_mm 53.0 \
    --num_bins 20 \
    --bg_depth_mm 100.0 \
    --wavelength 532e-9 \
    --pixel_pitch 4e-6 \
    --holo_z_m 0.0 \
    --amp_mode rgb \
    --rng_seed 2024 \
    --gamma 0.65 \
    --out_size 512 \
    --vis_mode percentile \
    --p_low 1.0 \
    --p_high 99.0
```

This command generates the hologram and automatically reconstructs all 20
depth-bin centers.

### 7.2 Full-Resolution Hologram Generation

After the quick test succeeds:

```bash
python ./model/Layer_ComplexHologram.py \
    --data_root ./dataset/Layer/BunnyDragon_RGBD \
    --out_root ./result/Layer/BunnyDragon_RGBD \
    --target_size 2048 \
    --z_min_mm 50.0 \
    --z_max_mm 53.0 \
    --num_bins 100 \
    --bg_depth_mm 100.0 \
    --bg_eps_mm 0.5 \
    --intensity_thresh 1e-4 \
    --wavelength 532e-9 \
    --pixel_pitch 4e-6 \
    --holo_z_m 0.0 \
    --amp_mode rgb \
    --rng_seed 2024 \
    --gamma 0.65 \
    --out_size 2048 \
    --vis_mode percentile \
    --p_low 1.0 \
    --p_high 99.0
```

For LDI input, only change the two directories:

```bash
python ./model/Layer_ComplexHologram.py \
    --data_root ./dataset/Layer/BunnyDragon_LDI \
    --out_root ./result/Layer/BunnyDragon_LDI \
    --target_size 2048 \
    --z_min_mm 50.0 \
    --z_max_mm 53.0 \
    --num_bins 100 \
    --bg_depth_mm 100.0 \
    --wavelength 532e-9 \
    --pixel_pitch 4e-6 \
    --holo_z_m 0.0 \
    --amp_mode rgb \
    --rng_seed 2024 \
    --gamma 0.65 \
    --out_size 2048 \
    --vis_mode percentile \
    --p_low 1.0 \
    --p_high 99.0
```

## 8. Understand the Important Parameters

### 8.1 Depth Parameters

The data-generation and hologram-generation depth settings must agree:

| Data generation | Hologram generation | Unit |
| --- | --- | --- |
| `--target_z_min 50.0` | `--z_min_mm 50.0` | millimeters |
| `--target_z_max 53.0` | `--z_max_mm 53.0` | millimeters |
| `--bg_depth 100.0` | `--bg_depth_mm 100.0` | millimeters |

Internally, the hologram generator converts each depth-bin center from
millimeters to meters before ASM propagation:

```text
50 mm = 0.050 m
53 mm = 0.053 m
```

`--holo_z_m` is already expressed in meters. With `--holo_z_m 0.0`, an
object layer at 50 mm is propagated by:

```text
0.000 m - 0.050 m = -0.050 m
```

### 8.2 Number of LDI Layers and Depth Bins

These parameters do different jobs:

- `--ldi_layers` controls how many ray-surface intersections are stored in
  the input representation.
- `--num_bins` controls how finely the physical depth range is discretized
  during hologram generation.

They do not need to have the same value.

For example:

```text
4 LDI layers → 100 hologram depth bins
```

Recommended values:

| Purpose | `ldi_layers` | `num_bins` |
| --- | ---: | ---: |
| Fast debugging | `1` | `10–20` |
| Standard RGB-D experiment | `1` | `50–100` |
| Layered LDI experiment | `2–8` | `50–200` |
| Very fine depth discretization | Any | `500+` |

A larger `num_bins` increases propagation and reconstruction time. It also
creates one reconstruction image and one `Binned` directory per depth bin.

### 8.3 Optical Parameters

- `--wavelength 532e-9` means a wavelength of 532 nm.
- `--pixel_pitch 4e-6` means a hologram pixel pitch of 4 µm.
- `--amp_mode rgb` uses RGB values directly as field amplitudes.
- `--amp_mode sqrt_rgb` interprets RGB values as intensity and uses their
  square roots as amplitudes.

Use the same wavelength and pixel pitch whenever the saved hologram is
reconstructed later.

## 9. Hologram Output Structure

For:

```text
--out_root ./result/Layer/BunnyDragon_RGBD
```

the script creates:

```text
result/Layer/BunnyDragon_RGBD/
├── Binned/
│   ├── plane_0000_z50.015mm/
│   │   ├── rgb.png
│   │   ├── mask.png
│   │   └── depth_mm.txt
│   ├── plane_0001_z50.045mm/
│   └── ...
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
│   ├── amp_R.png
│   ├── amp_G.png
│   ├── amp_B.png
│   ├── phase_R.png
│   ├── phase_G.png
│   └── phase_B.png
└── Recon/
    ├── recon_plane_0000_z50.015mm.png
    ├── recon_plane_0001_z50.045mm.png
    └── ...
```

The exact bin-center values depend on `z_min_mm`, `z_max_mm`, and
`num_bins`.

Important files:

- `holo_R/G/B.npy`: complete complex-valued holograms.
- `amp_R/G/B.npy`: unnormalized hologram amplitudes.
- `pha_R/G/B.npy`: hologram phases in radians.
- `amp_R/G/B.png`: amplitude visualizations only.
- `phase_R/G/B.png`: phase visualizations only.

Use the `.npy` files for numerical processing. PNG files are only for visual
inspection.

## 10. Optional: Reconstruct an Existing Hologram Again

`Layer_ComplexHologram.py` already reconstructs every depth-bin center. Use
`Reconstruction_layer.py` only when you want a different axial range,
smaller distance step, or different visualization settings without
regenerating the hologram.

```bash
python ./model/Reconstruction_layer.py \
    --amp_r ./result/Layer/BunnyDragon_RGBD/Hologram/amp_R.npy \
    --pha_r ./result/Layer/BunnyDragon_RGBD/Hologram/pha_R.npy \
    --amp_g ./result/Layer/BunnyDragon_RGBD/Hologram/amp_G.npy \
    --pha_g ./result/Layer/BunnyDragon_RGBD/Hologram/pha_G.npy \
    --amp_b ./result/Layer/BunnyDragon_RGBD/Hologram/amp_B.npy \
    --pha_b ./result/Layer/BunnyDragon_RGBD/Hologram/pha_B.npy \
    --outdir ./result/Layer/BunnyDragon_RGBD/Reconstruction_custom \
    --mode asm \
    --wavelength 532e-9 \
    --pitch 4e-6 \
    --zmin 0.050 \
    --zmax 0.053 \
    --step 0.0001 \
    --gamma 0.65 \
    --vis_mode percentile \
    --p_low 1.0 \
    --p_high 99.0 \
    --out_size 2048
```

Notice that this reconstruction script uses meters:

```text
0.050 m = 50 mm
0.053 m = 53 mm
```

Use `--z_list` for selected reconstruction distances:

```bash
python ./model/Reconstruction_layer.py \
    --amp_r ./result/Layer/BunnyDragon_RGBD/Hologram/amp_R.npy \
    --pha_r ./result/Layer/BunnyDragon_RGBD/Hologram/pha_R.npy \
    --amp_g ./result/Layer/BunnyDragon_RGBD/Hologram/amp_G.npy \
    --pha_g ./result/Layer/BunnyDragon_RGBD/Hologram/pha_G.npy \
    --amp_b ./result/Layer/BunnyDragon_RGBD/Hologram/amp_B.npy \
    --pha_b ./result/Layer/BunnyDragon_RGBD/Hologram/pha_B.npy \
    --outdir ./result/Layer/BunnyDragon_RGBD/Reconstruction_selected \
    --mode asm \
    --wavelength 532e-9 \
    --pitch 4e-6 \
    --z_list 0.050,0.0515,0.053 \
    --gamma 0.65 \
    --vis_mode percentile \
    --out_size 2048
```

## 11. Common Problems

### No Valid Intersections

Message:

```text
Error: no valid intersections found.
```

Possible causes:

- The camera does not face the object.
- The object was translated outside the ray field.
- The OBJ is empty or invalid.
- The field of view is too small.

Try increasing the field of view:

```bash
--fov_deg 3.0
```

If necessary, adjust:

```text
--eye_pos
--center_pos
--up_dir
--temp_obj_y
```

### No Valid RGB/Depth Pairs Found

Message:

```text
No valid rgb_layer*.png / depth_layer*.npy pairs found
```

The data directory must contain matching names:

```text
rgb_layer1.png
depth_layer1.npy
```

`depth_layer1.png` is not sufficient because the hologram generator requires
the `.npy` depth file.

### Reconstruction Is Completely Black

Check:

1. The `Binned` folders contain nonempty masks.
2. The valid depths lie inside `z_min_mm` and `z_max_mm`.
3. `bg_depth_mm` matches the background value used during data generation.
4. The reconstruction wavelength and pixel pitch match hologram generation.

For display only, try:

```text
--gamma 0.5
--p_low 0.1
--p_high 99.9
```

### Object Is Too Small or Cropped

Adjust `--fov_deg` during data generation:

- Smaller FOV: the object appears larger.
- Larger FOV: the object appears smaller and more of the scene is visible.

Always test at 512 × 512 before generating 2048 × 2048 data.

### Generation Is Too Slow

Reduce one or more of:

```text
--img_width
--img_height
--ldi_layers
--target_size
--num_bins
--out_size
```

The most expensive hologram settings are usually image resolution and the
number of depth bins.

### Memory Usage Is Too High

A 2048 × 2048 complex64 array requires approximately 32 MiB. The hologram
pipeline keeps several real and complex arrays in memory, so actual memory
usage is substantially larger.

Use the quick-test configuration first:

```text
512 × 512 images
1 LDI layer
20 depth bins
```

## 12. Minimal Copy-and-Paste Workflow

If the environment and mesh are already prepared, the complete RGB-D
workflow is:

```bash
python ./dataset/Obj2Layer.py \
    --input_obj ./dataset/BunnyDragon.obj \
    --output_dir ./dataset/Layer/BunnyDragon_RGBD \
    --img_width 2048 \
    --img_height 2048 \
    --fov_deg 1.0 \
    --ldi_layers 1 \
    --target_z_min 50.0 \
    --target_z_max 53.0 \
    --bg_depth 100.0

python ./model/Layer_ComplexHologram.py \
    --data_root ./dataset/Layer/BunnyDragon_RGBD \
    --out_root ./result/Layer/BunnyDragon_RGBD \
    --target_size 2048 \
    --z_min_mm 50.0 \
    --z_max_mm 53.0 \
    --num_bins 100 \
    --bg_depth_mm 100.0 \
    --wavelength 532e-9 \
    --pixel_pitch 4e-6 \
    --holo_z_m 0.0 \
    --amp_mode rgb \
    --rng_seed 2024 \
    --gamma 0.65 \
    --out_size 2048 \
    --vis_mode percentile \
    --p_low 1.0 \
    --p_high 99.0
```

After both commands finish, the complex holograms are located in:

```text
./result/Layer/BunnyDragon_RGBD/Hologram/
```

and the automatically reconstructed depth slices are located in:

```text
./result/Layer/BunnyDragon_RGBD/Recon/
```
