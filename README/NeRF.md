# NeRF Data: From a Mesh to an instant-ngp Hologram

This guide covers the complete pipeline:

```text
OBJ mesh
  → multi-view NeRF training images
  → instant-ngp training
  → .ingp snapshot
  → intermediate perspective light field
  → complex RGB holograms
  → depth reconstructions
```

The dataset-generation and instant-ngp build details are also documented in
[`dataset/NeRF/README.md`](../dataset/NeRF/README.md).

## 1. Files Used

| File | Purpose |
| --- | --- |
| [`dataset/NeRF/gen_multiview.py`](../dataset/NeRF/gen_multiview.py) | Generates NeRF training images and camera poses. |
| [`model/NeRF_ComplexHologram.py`](../model/NeRF_ComplexHologram.py) | Converts an `.ingp` snapshot directly into a light field, hologram, and reconstructions. |

## 2. Install Dataset-Generation Dependencies

```bash
python -m pip install numpy Pillow pyrender trimesh PyOpenGL
```

Headless rendering uses EGL.

## 3. Step 1: Generate NeRF Training Views

Start small:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python ./dataset/NeRF/gen_multiview.py \
    --mesh_path ./dataset/BunnyDragonRGB.obj \
    --output_dir ./dataset/NeRF/test \
    --num_views 4 \
    --resolution 512 \
    --crop_margin 128 \
    --mirror_mesh_index none \
    --seed 0
```

Full example:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python ./dataset/NeRF/gen_multiview.py \
    --mesh_path ./dataset/BunnyDragonRGB.obj \
    --output_dir ./dataset/NeRF/multiview \
    --num_views 120 \
    --resolution 6144 \
    --crop_margin 2048 \
    --mirror_mesh_index auto \
    --seed 0
```

Output:

```text
dataset/NeRF/multiview/
├── r_000.png
├── r_001.png
├── ...
└── transforms.json
```

The final image size is:

```text
resolution - 2 × crop_margin
```

## 4. Step 2: Download and Build instant-ngp

```bash
git clone --recursive https://github.com/NVlabs/instant-ngp.git
cd instant-ngp
python -m pip install -r requirements.txt
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j
```

If the repository was cloned without submodules:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

The build must produce the `pyngp` Python module because the combined
hologram script uses the instant-ngp Python bindings.

Test it from the instant-ngp directory:

```bash
PYTHONPATH=$PWD/build:$PYTHONPATH \
python -c "import pyngp; print(pyngp)"
```

## 5. Step 3: Train and Save `.ingp`

From the instant-ngp repository:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/run.py \
    --scene /path/to/CGHReviewCode/dataset/NeRF/multiview \
    --n_steps 35000 \
    --save_snapshot /path/to/output/bunny_dragon.ingp
```

The snapshot is the trained NeRF model. Keep the original
`transforms.json`; the hologram script uses it as a camera-coordinate
reference.

Check:

```bash
ls -lh /path/to/output/bunny_dragon.ingp
ls -lh /path/to/CGHReviewCode/dataset/NeRF/multiview/transforms.json
```

## 6. Step 4: Quick `.ingp` to Hologram Test

Return to this repository. Use a tiny light field first:

```bash
PYTHONPATH=/path/to/instant-ngp/build:$PYTHONPATH \
python ./model/NeRF_ComplexHologram.py \
    --snapshot /path/to/output/bunny_dragon.ingp \
    --ref_transforms ./dataset/NeRF/multiview/transforms.json \
    --output_dir ./result/NeRF/test \
    --nu 4 \
    --nv 4 \
    --lf_width 128 \
    --lf_height 128 \
    --total_baseline_x 0.02 \
    --total_baseline_z 0.02 \
    --scene_unit_in_m 1.0 \
    --offset_mode world_xz \
    --spp 4 \
    --linear_to_srgb \
    --d0 -0.050 \
    --wavelength 532e-9 \
    --pitch 2e-6 \
    --dist_rs_to_h 0.05 \
    --z_min -0.052 \
    --z_max -0.048 \
    --z_step 0.002 \
    --out_size 512 \
    --save_holo_png \
    --save_recon_png
```

You may replace `PYTHONPATH` with:

```text
--instant_ngp_build /path/to/instant-ngp/build
```

## 7. Larger NeRF Hologram Run

Increase dimensions gradually:

```bash
PYTHONPATH=/path/to/instant-ngp/build:$PYTHONPATH \
python ./model/NeRF_ComplexHologram.py \
    --snapshot /path/to/output/bunny_dragon.ingp \
    --ref_transforms ./dataset/NeRF/multiview/transforms.json \
    --output_dir ./result/NeRF/bunny_dragon \
    --nu 10 \
    --nv 10 \
    --lf_width 256 \
    --lf_height 256 \
    --total_baseline_x 0.05 \
    --total_baseline_z 0.05 \
    --scene_unit_in_m 1.0 \
    --offset_mode world_xz \
    --spp 8 \
    --linear_to_srgb \
    --d0 -0.050 \
    --wavelength 532e-9 \
    --pitch 2e-6 \
    --dist_rs_to_h 0.05 \
    --z_min -0.052 \
    --z_max -0.048 \
    --z_step 0.001 \
    --out_size 2048 \
    --save_holo_png \
    --save_recon_png
```

## 8. Memory Rule

The tiled field is:

```text
(nv × lf_height) × (nu × lf_width)
```

`40×40` views at `800×800` create a `32000×32000` field. This is extremely
large. The script blocks fields above a conservative threshold unless
`--allow_large_field` is supplied.

Do not use `--allow_large_field` until available RAM and FFT workspace
requirements have been checked.

## 9. Important Parameters

- `frame_idx`: training camera used as the LF center view.
- `fov_deg`: negative reads `camera_angle_x` from `transforms.json`.
- `scene_unit_in_m`: physical scale of one NeRF scene unit.
- `total_baseline_x/z`: camera-array span in scene units.
- `d0`: perspective LF refocus distance in meters.
- `dist_rs_to_h`: RS-plane to hologram-plane propagation distance.
- `z_min/z_max/z_step`: reconstruction distances in meters.
- `spp`: instant-ngp rendering quality.

An incorrect `scene_unit_in_m` changes the physical LF baseline and therefore
the refocusing shifts.

## 10. Output Structure

```text
result/NeRF/bunny_dragon/
├── run_config.json
├── LightField/
│   ├── images/
│   ├── poses.csv
│   ├── intrinsics.csv
│   ├── reference_pose.json
│   └── time.csv
├── Hologram/
│   ├── hologram_complex_{R,G,B}.npy
│   ├── hologram_amp_{R,G,B}.npy
│   ├── hologram_phase_{R,G,B}.npy
│   └── optional PNG previews
└── Recon/
    └── recon_z_*.png
```

The intermediate light field is intentionally retained for inspection and
reuse.

## 11. Common Problems

### `No module named pyngp`

The instant-ngp Python bindings were not built or are not on `PYTHONPATH`.
Rebuild instant-ngp and point to its build directory.

### Snapshot Loads but Images Are Empty

Check that the snapshot and reference `transforms.json` belong to the same
training scene. Try a different `frame_idx`.

### Refocusing Looks Wrong

Check `scene_unit_in_m`, baseline size, `d0`, and offset mode. Start with a
very small baseline.

### Out of Memory

Reduce `nu`, `nv`, `lf_width`, and `lf_height`. The complex-field dimensions
are products of angular and spatial resolution.

### EGL Error During Training-View Generation

Try:

```bash
EGL_DEVICE_ID=0 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python ./dataset/NeRF/gen_multiview.py ...
```

## 12. Minimal End-to-End Summary

```text
1. gen_multiview.py → images + transforms.json
2. instant-ngp scripts/run.py → model.ingp
3. NeRF_ComplexHologram.py → LightField + Hologram + Recon
```
