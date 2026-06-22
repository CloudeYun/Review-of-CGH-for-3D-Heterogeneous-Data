# Hologram Reconstruction Guide

This guide explains how to reconstruct saved RGB holograms without
regenerating them.

Two scripts are available:

| File | Output |
| --- | --- |
| [`model/Reconstruction_layer.py`](../model/Reconstruction_layer.py) | One focused RGB image per propagation distance. |
| [`model/Reconstruction_mutiview.py`](../model/Reconstruction_mutiview.py) | Multiple sub-aperture views per distance. |

## 1. Required Hologram Files

Each RGB channel needs:

```text
amplitude.npy
phase.npy
```

The scripts reconstruct the complex hologram as:

```text
H = amplitude × exp(i × phase)
```

Use unnormalized `.npy` arrays. Do not use amplitude or phase PNG previews.

Common naming conventions:

| Pipeline | Amplitude | Phase |
| --- | --- | --- |
| Layer | `amp_R.npy` | `pha_R.npy` |
| Point cloud | `amp_R.npy` | `phase_R.npy` |
| Voxel | `amp_R.npy` | `pha_R.npy` |
| Light field / NeRF | `hologram_amp_R.npy` | `hologram_phase_R.npy` |
| Mesh | `channel_Red_amp.npy` | `channel_Red_phase.npy` |

Always inspect the actual Hologram directory before copying a command.

## 2. Standard Single-View Reconstruction

Example for a layer hologram:

```bash
python ./model/Reconstruction_layer.py \
    --amp_r ./result/Layer/example/Hologram/amp_R.npy \
    --pha_r ./result/Layer/example/Hologram/pha_R.npy \
    --amp_g ./result/Layer/example/Hologram/amp_G.npy \
    --pha_g ./result/Layer/example/Hologram/pha_G.npy \
    --amp_b ./result/Layer/example/Hologram/amp_B.npy \
    --pha_b ./result/Layer/example/Hologram/pha_B.npy \
    --outdir ./result/Layer/example/Reconstruction_custom \
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

The Z values are in meters.

## 3. Reconstruct Selected Distances

`z_list` is a comma-separated string:

```bash
python ./model/Reconstruction_layer.py \
    --amp_r ./result/Layer/example/Hologram/amp_R.npy \
    --pha_r ./result/Layer/example/Hologram/pha_R.npy \
    --amp_g ./result/Layer/example/Hologram/amp_G.npy \
    --pha_g ./result/Layer/example/Hologram/pha_G.npy \
    --amp_b ./result/Layer/example/Hologram/amp_B.npy \
    --pha_b ./result/Layer/example/Hologram/pha_B.npy \
    --outdir ./result/Layer/example/Reconstruction_selected \
    --mode asm \
    --wavelength 532e-9 \
    --pitch 4e-6 \
    --z_list 0.050,0.0515,0.053 \
    --gamma 0.65 \
    --vis_mode percentile \
    --out_size 2048
```

## 4. Choose the Propagation Mode

### Standard ASM

Use:

```text
--mode asm
```

This is appropriate for layer, point-cloud, voxel, light-field, and NeRF
holograms generated with standard ASM.

### Padded/Band-Limited ASM

Use:

```text
--mode asm_pad
```

This matches the mesh pipeline's padded propagation more closely.

## 5. Match Optical Parameters

The reconstruction wavelength and pixel pitch must match hologram generation:

```text
generation wavelength = reconstruction wavelength
generation pitch      = reconstruction pitch
```

If these values differ, the reconstructed depth scale and image content will
be wrong.

The shared reconstruction scripts currently use one wavelength for all RGB
channels. For holograms generated with distinct RGB wavelengths, prefer the
reconstruction built into the original generator.

## 6. Amplitude Versus Intensity

Default display:

```text
|U|
```

With:

```text
--use_intensity
```

the script displays:

```text
|U|²
```

This affects visualization only; it does not change propagation.

## 7. Visualization Modes

### Percentile

```text
--vis_mode percentile --p_low 1 --p_high 99
```

Usually the best default because it suppresses extreme outliers.

### Min-Max

```text
--vis_mode minmax
```

Uses the global minimum and maximum. It may appear dark if a few pixels are
very bright.

### Mesh Style

```text
--vis_mode mesh_style
```

Uses logarithmic/background suppression designed for mesh holograms.

`gamma < 1` brightens darker structures. Try `0.5–0.8`.

## 8. Orientation and DC Options

Single-view reconstruction supports:

```text
--flipud
```

Use this only if the hologram orientation must be vertically reversed.

Multi-view reconstruction supports:

```text
--flipud_holo
--remove_dc 1
```

DC removal subtracts the hologram mean and can reduce the zero-order term.

## 9. Multi-View Reconstruction

This script propagates the hologram to a selected depth, crops shifted
sub-apertures in the Fourier plane, and saves an angular view grid.

```bash
python ./model/Reconstruction_mutiview.py \
    --amp_r ./result/LightField/example/Hologram/hologram_amp_R.npy \
    --pha_r ./result/LightField/example/Hologram/hologram_phase_R.npy \
    --amp_g ./result/LightField/example/Hologram/hologram_amp_G.npy \
    --pha_g ./result/LightField/example/Hologram/hologram_phase_G.npy \
    --amp_b ./result/LightField/example/Hologram/hologram_amp_B.npy \
    --pha_b ./result/LightField/example/Hologram/hologram_phase_B.npy \
    --outdir ./result/LightField/example/Multiview \
    --mode asm \
    --wavelength 532e-9 \
    --pitch 2e-6 \
    --z_list -0.050 \
    --view_grid 9 \
    --aperture_ratio 0.35 \
    --max_shift_ratio 0.18 \
    --recenter_subaperture 1 \
    --gamma 0.65 \
    --vis_mode percentile \
    --out_size 512 \
    --save_mosaic
```

Output:

```text
Multiview/
└── z_-0.050000m/
    ├── v_00_00.png
    ├── v_00_01.png
    ├── ...
    └── mosaic.png
```

Important parameters:

- `view_grid 9`: generates `9×9 = 81` views.
- `aperture_ratio`: sub-aperture size relative to the spectrum.
- `max_shift_ratio`: angular displacement range.
- `recenter_subaperture 1`: recommended to remove the carrier introduced by
  shifted spectrum cropping.

Large `view_grid` and `out_size` can produce very large mosaics. A
`9×9` mosaic with `out_size=2048` is `18432×18432`.

## 10. Sign Conventions

Different generators use different propagation conventions:

- Layer, point cloud, and voxel examples commonly reconstruct near
  `+0.050 m`.
- Perspective light-field and NeRF examples commonly reconstruct near
  `-0.050 m`.
- Mesh reconstruction is centered around its configured `shiftZ`.

Use the distances documented by the original generator. If uncertain, scan a
small interval on both sides using a coarse step.

## 11. Common Problems

### Amplitude/Phase Shape Mismatch

All six arrays must have identical 2D shapes.

### Completely Black Output

Try percentile visualization, lower gamma, verify Z sign, and confirm optical
parameters.

### Reconstruction Is Blurry at Every Depth

Check wavelength, pitch, propagation mode, and whether the amplitude/phase
files belong to the same run.

### Multi-View Crop Warning

The script may clip `max_shift` to keep the sub-aperture inside the spectrum.
Reduce `max_shift_ratio` or `aperture_ratio`.

### Mosaic Uses Too Much Memory

Reduce `view_grid`, `out_size`, or omit `--save_mosaic`.

## 12. Practical Debugging Order

1. Reconstruct one Z distance.
2. Use `out_size=512`.
3. Confirm the center view.
4. Scan a coarse Z range.
5. Refine the Z step.
6. Only then enable multi-view mosaics.
