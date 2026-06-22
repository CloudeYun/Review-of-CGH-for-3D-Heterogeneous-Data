# Triangle-Mesh Data: From OBJ to GPU-Generated Complex Holograms

This pipeline uses triangular primitives directly:

```text
OBJ mesh
  → colored Nx12 triangle TXT
  → GPU mesh CGH
  → RGB reconstruction scan
```

The hologram generator requires an NVIDIA CUDA GPU and CuPy.

## 1. Files Used

| File | Purpose |
| --- | --- |
| [`dataset/Obj2Mesh.py`](../dataset/Obj2Mesh.py) | Converts OBJ triangles to an Nx12 text representation. |
| [`model/Mesh_ComplexHologram.py`](../model/Mesh_ComplexHologram.py) | Generates RGB mesh holograms on a CUDA GPU. |
| [`model/Reconstruction_layer.py`](../model/Reconstruction_layer.py) | Optional reconstruction from saved amplitude and phase. |

## 2. Install Dependencies

Install general dependencies:

```bash
python -m pip install numpy trimesh imageio tqdm
```

Install a CuPy build matching the installed CUDA version. For example:

```bash
python -m pip install cupy-cuda12x
```

Use `cupy-cuda11x` for a CUDA 11 installation.

Check the GPU and CuPy:

```bash
nvidia-smi
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

## 3. Step 1: Convert OBJ to Nx12 TXT

Each output row has:

```text
x1 y1 z1 x2 y2 z2 x3 y3 z3 r g b
```

Generate a small test mesh:

```bash
python ./dataset/Obj2Mesh.py \
    --obj_path ./dataset/BunnyDragon.obj \
    --out_txt ./dataset/Mesh/BunnyDragon_2k.txt \
    --target_faces 2000 \
    --rot_x 90 \
    --rot_y 0 \
    --rot_z 0
```

Generate a higher-detail mesh:

```bash
python ./dataset/Obj2Mesh.py \
    --obj_path ./dataset/BunnyDragon.obj \
    --out_txt ./dataset/Mesh/BunnyDragon_50k.txt \
    --target_faces 50000 \
    --rot_x 90
```

Alternatively, simplify by ratio:

```bash
python ./dataset/Obj2Mesh.py \
    --obj_path ./dataset/BunnyDragon.obj \
    --out_txt ./dataset/Mesh/BunnyDragon_half.txt \
    --ratio 0.5
```

`--ratio` overrides `--target_faces`.

Inspect the TXT:

```bash
python - <<'PY'
import numpy as np
x = np.loadtxt("./dataset/Mesh/BunnyDragon_2k.txt")
print("shape:", x.shape)
print("expected columns:", 12)
print("coordinate range:", x[:, :9].min(), x[:, :9].max())
print("color range:", x[:, 9:12].min(), x[:, 9:12].max())
PY
```

The array must have 12 columns. The final three columns are RGB face colors.

## 4. Step 2: Quick GPU Hologram Test

Start with a low hologram resolution and 2,000 triangles:

```bash
python ./model/Mesh_ComplexHologram.py \
    --mesh_txt ./dataset/Mesh/BunnyDragon_2k.txt \
    --outdir ./result/Mesh/BunnyDragon_test \
    --Nx 512 \
    --Ny 512 \
    --dx 8e-6 \
    --dy 8e-6 \
    --lam_r 638e-9 \
    --lam_g 520e-9 \
    --lam_b 450e-9 \
    --shiftZ 0.050 \
    --objectScaleRatio 2 \
    --target_depth 0.003 \
    --z_boost 1.0 \
    --shading continuous \
    --illu 0 0 1 \
    --log_every 200 \
    --z_min 0.045 \
    --z_max 0.055 \
    --z_step 0.001 \
    --save_every 1 \
    --resume 1 \
    --device 0
```

## 5. Full-Resolution Hologram

After the test succeeds:

```bash
python ./model/Mesh_ComplexHologram.py \
    --mesh_txt ./dataset/Mesh/BunnyDragon_50k.txt \
    --outdir ./result/Mesh/BunnyDragon_50k \
    --Nx 2048 \
    --Ny 2048 \
    --dx 8e-6 \
    --dy 8e-6 \
    --lam_r 638e-9 \
    --lam_g 520e-9 \
    --lam_b 450e-9 \
    --shiftX 0.0 \
    --shiftY 0.0 \
    --shiftZ 0.050 \
    --objectScaleRatio 2 \
    --target_depth 0.003 \
    --z_boost 1.0 \
    --shading continuous \
    --illu 0 0 1 \
    --log_every 500 \
    --z_min 0.045 \
    --z_max 0.055 \
    --z_step 0.0001 \
    --save_every 1 \
    --resume 1 \
    --device 0
```

All mesh hologram distances are in meters.

Important geometry parameters:

- `shiftZ`: center position of the object along the optical axis.
- `target_depth`: desired object thickness along Z.
- `objectScaleRatio`: controls the transverse object size relative to the
  hologram aperture.
- `z_boost`: multiplies the target Z thickness.
- `shiftX`, `shiftY`: lateral object displacement.

Keep the reconstruction range around the object:

```text
object center: 0.050 m
object thickness: 0.003 m
reasonable scan: approximately 0.045–0.055 m
```

## 6. Resume and Checkpoints

The script saves `checkpoint.json`. With `--resume 1`, it:

- Reuses existing RGB holograms.
- Continues an interrupted reconstruction scan.
- Avoids repeating completed work when the run configuration matches.

Use a new output directory when changing major geometry, wavelength, or
sampling parameters.

## 7. Output Structure

```text
result/Mesh/BunnyDragon_50k/
├── checkpoint.json
├── Holograms/
│   ├── channel_Red_complex.npy
│   ├── channel_Red_amp.npy
│   ├── channel_Red_phase.npy
│   ├── channel_Green_complex.npy
│   ├── channel_Green_amp.npy
│   ├── channel_Green_phase.npy
│   ├── channel_Blue_complex.npy
│   ├── channel_Blue_amp.npy
│   ├── channel_Blue_phase.npy
│   └── PNG previews
└── recon_rgb/
    └── recon_rgb_z_*.png
```

The complex mesh holograms are saved as complex128 arrays. Amplitude and
phase arrays are used for later reconstruction.

## 8. Optional Reconstruction with the Shared Script

Mesh holograms use the padded/band-limited propagation mode:

```bash
python ./model/Reconstruction_layer.py \
    --amp_r ./result/Mesh/BunnyDragon_50k/Holograms/channel_Red_amp.npy \
    --pha_r ./result/Mesh/BunnyDragon_50k/Holograms/channel_Red_phase.npy \
    --amp_g ./result/Mesh/BunnyDragon_50k/Holograms/channel_Green_amp.npy \
    --pha_g ./result/Mesh/BunnyDragon_50k/Holograms/channel_Green_phase.npy \
    --amp_b ./result/Mesh/BunnyDragon_50k/Holograms/channel_Blue_amp.npy \
    --pha_b ./result/Mesh/BunnyDragon_50k/Holograms/channel_Blue_phase.npy \
    --outdir ./result/Mesh/BunnyDragon_50k/Reconstruction_custom \
    --mode asm_pad \
    --wavelength 532e-9 \
    --pitch 8e-6 \
    --zmin 0.045 \
    --zmax 0.055 \
    --step 0.0001 \
    --gamma 1.0 \
    --vis_mode mesh_style \
    --out_size 2048
```

The shared reconstruction script accepts one wavelength for all channels. If
the hologram was generated with distinct RGB wavelengths, the built-in mesh
reconstruction is the physically consistent option.

## 9. Common Problems

### CuPy Cannot Find CUDA

Confirm that the installed CuPy package matches the system CUDA major
version. Check with `nvidia-smi`.

### GPU Out of Memory

Reduce `Nx`, `Ny`, or triangle count. Start at 512 × 512 and 2,000 faces.

### Object Is Too Large

Increase `objectScaleRatio`. The transverse size is inversely related to this
ratio in the current script.

### Reconstruction Is at the Wrong Distance

Make sure `shiftZ`, `z_min`, and `z_max` use the same meter convention.

### Simplification Fails

Use a smaller target reduction, update Trimesh, or export a pre-simplified
mesh from a modeling application.

## 10. Minimal Workflow

```bash
python ./dataset/Obj2Mesh.py \
    --obj_path ./dataset/BunnyDragon.obj \
    --out_txt ./dataset/Mesh/BunnyDragon_2k.txt \
    --target_faces 2000 \
    --rot_x 90

python ./model/Mesh_ComplexHologram.py \
    --mesh_txt ./dataset/Mesh/BunnyDragon_2k.txt \
    --outdir ./result/Mesh/BunnyDragon_2k \
    --Nx 512 --Ny 512 \
    --dx 8e-6 --dy 8e-6 \
    --lam_r 638e-9 --lam_g 520e-9 --lam_b 450e-9 \
    --shiftZ 0.050 \
    --objectScaleRatio 2 \
    --target_depth 0.003 \
    --z_min 0.045 --z_max 0.055 --z_step 0.001 \
    --device 0 \
    --resume 1
```
