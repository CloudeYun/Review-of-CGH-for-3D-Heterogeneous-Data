# NeRF Multi-View Dataset Generation

This directory provides a utility for generating multi-view RGB images and
camera poses from a 3D mesh. The generated dataset follows a
NeRF-compatible layout and can be used as input for NeRF training pipelines
that accept a `transforms.json` file.

## Overview

The script [`gen_multiview.py`](./gen_multiview.py) performs the following
operations:

1. Loads all valid geometries from an OBJ file or another mesh format
   supported by Trimesh.
2. Centers and uniformly scales the complete scene while preserving the
   relative poses of its sub-meshes.
3. Optionally mirrors one selected sub-mesh along its local Y axis.
4. Samples camera positions over the upper hemisphere.
5. Renders and crops RGB images using Pyrender.
6. Exports camera-to-world matrices to `transforms.json`.

The first camera pose is deterministic. The remaining poses are randomly
sampled using a configurable seed, allowing the dataset to be reproduced.

## Workflow

The NeRF dataset generation and training workflow consists of three stages:

```text
3D mesh
   │
   ├─ gen_multiview.py
   ▼
RGB training views + transforms.json
   │
   ├─ instant-ngp training
   ▼
trained NeRF snapshot (.ingp)
```

The `.ingp` file is a trained instant-ngp snapshot. It is not generated
directly by `gen_multiview.py`.

## Requirements

The multi-view generation script requires Python 3 and the following
packages:

```text
numpy
Pillow
pyrender
trimesh
PyOpenGL
```

Install the dependencies with:

```bash
pip install numpy Pillow pyrender trimesh PyOpenGL
```

Headless rendering uses EGL. A working OpenGL/EGL driver is therefore
required on the host system.

instant-ngp additionally requires an NVIDIA GPU, CUDA, a C++ compiler, and
CMake. Refer to the
[official instant-ngp repository](https://github.com/NVlabs/instant-ngp)
for the current platform requirements.

## Step 1: Generate the NeRF Training Dataset

Run the script from the repository root:

```bash
python ./dataset/NeRF/gen_multiview.py \
    --mesh_path ./dataset/BunnyDragonRGB.obj \
    --output_dir ./dataset/NeRF/multiview \
    --num_views 120 \
    --resolution 6144 \
    --crop_margin 2048 \
    --mirror_mesh_index auto \
    --seed 0
```

With the settings above, each image is first rendered at 6144 × 6144 pixels.
A margin of 2048 pixels is then removed from every side, producing final
images of 2048 × 2048 pixels.

For a quick test, use a lower resolution and a small number of views:

```bash
python ./dataset/NeRF/gen_multiview.py \
    --mesh_path ./dataset/BunnyDragonRGB.obj \
    --output_dir ./dataset/NeRF/test \
    --num_views 4 \
    --resolution 512 \
    --crop_margin 128 \
    --mirror_mesh_index none
```

The generated directory contains the RGB training views and the
`transforms.json` file required by instant-ngp.

## Step 2: Download and Build instant-ngp

Clone the official repository with all submodules:

```bash
git clone --recursive https://github.com/NVlabs/instant-ngp.git
cd instant-ngp
```

If the repository was cloned without `--recursive`, initialize its
submodules manually:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Configure and build instant-ngp:

```bash
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j
```

If the parallel build consumes too much memory, remove `-j` or specify a
smaller job count, for example:

```bash
cmake --build build --config RelWithDebInfo -j 4
```

The Python module `pyngp` should be available in the instant-ngp build
directory after a successful build.

### GPU Architecture Selection

On systems with multiple or unusual NVIDIA GPUs, automatic CUDA architecture
detection may fail. In that case, set `TCNN_CUDA_ARCHITECTURES` before
building. For example, NVIDIA Ampere GPUs such as the RTX 30 series and A40
use compute capability 86:

```bash
TCNN_CUDA_ARCHITECTURES=86 \
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo

cmake --build build --config RelWithDebInfo -j
```

Select the GPU used for training with `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py --help
```

## Step 3: Train NeRF and Save an `.ingp` Snapshot

From the instant-ngp repository, train on the directory generated in Step 1:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/run.py \
    --scene /path/to/CGHReviewCode/dataset/NeRF/multiview \
    --n_steps 35000 \
    --save_snapshot /path/to/output/bunny_dragon.ingp
```

The training directory supplied to `--scene` must contain:

```text
multiview/
├── r_000.png
├── r_001.png
├── ...
└── transforms.json
```

The `--n_steps` value controls the number of optimization steps. A larger
value can improve quality but increases training time. The appropriate value
depends on scene complexity, image count, resolution, and desired quality.

After training finishes, the requested snapshot is saved as:

```text
/path/to/output/bunny_dragon.ingp
```

The official `scripts/run.py` interface accepts both `.ingp` and `.msgpack`
as recommended snapshot extensions.

### Interactive Training and Snapshot Export

To inspect training through the graphical interface:

```bash
./build/testbed \
    --scene /path/to/CGHReviewCode/dataset/NeRF/multiview
```

Depending on the instant-ngp build, the executable may also be available as
`./instant-ngp`. In the GUI, use the **Snapshot** panel to save the trained
model as an `.ingp` file.

### Reloading a Snapshot

A saved snapshot can be loaded again with:

```bash
python scripts/run.py \
    --load_snapshot /path/to/output/bunny_dragon.ingp \
    --gui
```

## Command-Line Arguments

| Argument | Default | Description |
| --- | ---: | --- |
| `--mesh_path` | Required | Path to the input mesh. |
| `--output_dir` | Required | Directory for rendered images and camera metadata. |
| `--num_views` | `120` | Number of camera views to render. |
| `--resolution` | `6144` | Square image resolution before cropping. |
| `--crop_margin` | `2048` | Pixels removed from each image border. |
| `--scale_factor` | `0.8` | Maximum extent of the normalized scene. |
| `--camera_radius` | `3.0` | Distance from each camera to the scene origin. |
| `--camera_fov_deg` | `60.0` | Vertical camera field of view in degrees. |
| `--max_polar_angle_deg` | Approximately `81.82` | Maximum polar angle for upper-hemisphere sampling. |
| `--mirror_mesh_index` | `auto` | Sub-mesh index to mirror, `auto`, or `none`. |
| `--seed` | `0` | Random seed for camera sampling. |
| `--progress_interval` | `30` | Number of views between progress messages. |

The final image size is:

```text
final_size = resolution - 2 × crop_margin
```

The crop margin must be smaller than half of the rendering resolution.

## Sub-Mesh Mirroring

The `--mirror_mesh_index` option accepts three forms:

- `auto`: select the most compact sub-mesh using a bounding-box heuristic.
- An integer such as `0` or `1`: mirror the specified sub-mesh.
- `none`: disable mirroring.

The script prints geometric statistics for every sub-mesh before rendering.
These statistics can be used to choose an index manually when automatic
selection is unsuitable.

Mirroring is performed around the selected mesh's bounding-box center along
its local Y axis. Triangle winding is corrected after reflection to preserve
the expected surface orientation.

## Camera Sampling

All cameras look toward the scene origin.

- View `0` uses a fixed camera pose for reproducible inspection.
- The remaining views are sampled over the upper hemisphere.
- `--camera_radius` controls the camera distance.
- `--max_polar_angle_deg` controls the hemisphere coverage.
- `--seed` makes the sampled camera sequence reproducible.

The exported matrices use the OpenGL camera convention and represent
camera-to-world transformations.

## Output Structure

The output directory has the following structure:

```text
multiview/
├── r_000.png
├── r_001.png
├── r_002.png
├── ...
└── transforms.json
```

The metadata file contains the camera field of view and one transformation
matrix for each rendered image:

```json
{
    "camera_angle_x": 1.0471975511965976,
    "aabb_scale": 4,
    "frames": [
        {
            "file_path": "r_000.png",
            "transform_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        }
    ]
}
```

The matrix above is only an illustrative example. Actual values depend on
the generated camera pose.

## Headless EGL Rendering

The script configures Pyrender to use EGL and selects EGL device `0` by
default. A different device can be selected before execution:

```bash
EGL_DEVICE_ID=1 \
python ./dataset/NeRF/gen_multiview.py \
    --mesh_path ./dataset/BunnyDragonRGB.obj \
    --output_dir ./dataset/NeRF/multiview
```

### `GLIBCXX` Version Error

Some Conda environments include an older `libstdc++.so.6` that is
incompatible with the system OpenGL driver. An error may contain a message
similar to:

```text
version `GLIBCXX_3.4.30' not found
```

On Linux systems where the compatible library is located at
`/usr/lib/x86_64-linux-gnu/libstdc++.so.6`, run:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python ./dataset/NeRF/gen_multiview.py \
    --mesh_path ./dataset/BunnyDragonRGB.obj \
    --output_dir ./dataset/NeRF/multiview
```

The exact system library path may differ across Linux distributions.

### EGL Device Permission Error

If EGL attempts to access a render device without permission, explicitly
select an accessible EGL device:

```bash
EGL_DEVICE_ID=0 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python ./dataset/NeRF/gen_multiview.py \
    --mesh_path ./dataset/BunnyDragonRGB.obj \
    --output_dir ./dataset/NeRF/multiview
```

If no EGL device is accessible, consult the system administrator regarding
GPU and render-device permissions.

## Performance Considerations

Rendering at 6144 × 6144 pixels requires substantial GPU memory, system
memory, storage, and execution time. Before generating a complete dataset,
it is recommended to verify the mesh orientation and camera configuration
with:

- A low rendering resolution
- One to four camera views
- A temporary output directory

After inspecting the test images, increase the resolution and number of
views for the final dataset.
