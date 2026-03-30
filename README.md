# The Code in Review of CGH for 3D Heterogeneous Data

## 代码配置步骤 ✨

### 1. 将该项目拷贝至你的本地运行环境中

拷贝后，你的项目结构应如下所示

```python
./Review-of-CGH-for-3D-Heterogeneous-Data
├── dataset/                                     # 生成六大类格式数据的目录及代码
     ├── Layer/                                  # 存放层化数据的目录
     ├── LightField/                             # 存放光场数据的目录           
     ├── Mesh/                                   # 存放Mesh数据的目录
     ├── PointCloud/                             # 存放PointCloud数据的目录
     ├── Voxe/                                   # 存放体素数据的目录
     ├── Obj2Layer.py                            # 生成层化数据的代码
     ├── Obj2LF_orth.py                          # 生成正交光场的代码
     ├── Obj2LF_pers.py                          # 生成透视光场的代码
     ├── Obj2Mesh.py                             # 生成Mesh的代码
     ├── Obj2PCD.py                              # 生成PointCloud的代码                                
     └── Obj2Voxel.py                            # 生成体素的代码
    
 ├── model                                       # 生成全息图并重建的代码
     ├── Layer_ComplexHologram.py                # 层化数据——生成全息图并重建的代码
     ├── LightField_Orth_ComplexHologram.py      # 正交光场——生成全息图并重建的代码
     ├── LightField_Pers_ComplexHologram.py      # 透视光场——生成全息图并重建的代码
     ├── Mesh_ComplexHologram.py                 # Mesh数据——生成全息图并重建的代码
     ├── PointCloud_ComplexHologram.py           # 点云数据——生成全息图并重建的代码
     ├── Reconstruction_layer.py                 # 层化重建代码
     ├── Reconstruction_mutiview.py              # 多视角重建代码           
     └── Voxel_ComplexHologram.py                # 体素数据——生成全息图并重建的代码    
     
 ├── result\                                     #存放重建结果的目录
      ├── Layer\                                 #存放层化数据重建结果的目录
      ├── LightField\                            #存放光场数据重建结果的目录
      ├── Mesh\                                  #存放Mesh数据重建结果的目录
      ├── NeRF\                                  #存放NeRF数据重建结果的目录
      ├── PointCloud\                            #存放点云数据重建结果的目录
      └── Voxel\                                 #存放体素数据重建结果的目录

 ├── README\
      ├── Layer.md                               #层化数据README
      ├── LightField.md\                         #光场数据README
      ├── Mesh.md\                               #Mesh数据README
      ├── NeRF.md\                               #NeRF数据README
      ├── PointCloud.md\                         #点云数据README
      └── Voxel.md\                              #体素数据README

└── environment.yml                              #环境配置文件


```
### 2.获取初始数据集
初始数据集（BunnyDragon.obj）获取网址如下：

通过网盘分享的文件：Review of CGH for 3D Heterogeneous Data

链接: https://pan.baidu.com/s/1v8WpkBr7jnyo2QVmDijYJA?pwd=kam7 

提取码: kam7

将下载好的BunnyDragon.obj放入文件夹dataset中 

### 3.配置环境

采用environment.yml文件配置conda环境，执行
``` bash
conda env create -f environment.yml
```

## 运行流程 ✨

### 1.层化数据 🎯

生成层化数据代码：

生成层化数据全息图代码：

详细指导内容：

层重建结果：

<div style="display: flex; gap: 12px; justify-content: center;">
  <figure style="text-align: center;">
    <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Layer/layer_recon_1.png" width="30%">
    <figcaption>图1：50mm</figcaption>
  </figure>

  <figure style="text-align: center;">
    <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Layer/layer_recon_2.png" width="30%">
    <figcaption>图2：51mm</figcaption>
  </figure>

  <figure style="text-align: center;">
    <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Layer/layer_recon_3.png" width="30%">
    <figcaption>图3：52mm</figcaption>
  </figure>
</div>

多视重建结果

### 2.点云数据 ❄️

生成点云数据代码：

生成点云数据全息图代码：

详细指导内容：

### 3.光场数据 ⚡️

生成光场数据代码：

生成光场数据全息图代码：

详细指导内容：

### 4.Mesh数据 ▶️

生成Mesh数据代码：

生成Mesh数据全息图代码：

详细指导内容：

### 5.体素数据 ♦️

生成体素数据代码：

生成体素数据全息图代码：

详细指导内容：

### 6.NeRF数据 🧠

生成NeRF数据代码：

生成NeRF数据全息图代码：

详细指导内容：

### 7.重建 🪜

层重建代码：

多视角重建代码：

详细指导内容:

#### 1.层化数据

运行./CGH_Review/dataset/mesh2LDI.py      一些可修改参数如下（代码开头位置）

```python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) #定位当前目录
OBJ_PATH = os.path.join(SCRIPT_DIR, "Mesh/BunnyDragonRGB.obj") #输入的obj数据不用动
OUT_DIR  = os.path.join(SCRIPT_DIR, "Layer/BunnyDragonLDI_50_53mm") #输出的LDI数据目录

IMG_W = 2048  #图像大小，不用动
IMG_H = 2048  #图像大小，不用动
FOV_DEG = 1.5 #视角，不用动
LDI_LAYERS = 10 #生成的LDI层数，这里是双面10层
EPS_SHIFT = 1e-4 #不用动
```

这样就在./CGH_Review/dataset/Layer/ 目录下的到了10层正反面的LDI数据（就是现在已经给你的BunnyDragonColor_double_LDI）

如果要生成正面的LDI，则挑选第1、3、5、7、9面重新整合一个目录（就是现在已经给你的BunnyDragonColor_single_LDI）

如果要生成RGBD，则挑选第1面重新整合一个目录（就是现在已经给你的BunnyDragonColor_RGBD）

#### 2.点云数据

运行./CGH_Review/dataset/mesh2pcd_color.py    一些可修改的参数如下（代码开头位置）

```python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_MESH_DIR = "Mesh/BunnyDragonRGB.obj" #输入的obj数据不用动
OUTPUT_PCD_DIR = "PointCloud/PointCloud_3000000/bunnydragonRGB_pointcloud_color_3e7.ply" #输出的点云数据目录

OBJ_PATH = os.path.join(SCRIPT_DIR, INPUT_MESH_DIR)
OUT_PLY  = os.path.join(SCRIPT_DIR, OUTPUT_PCD_DIR)

N_POINTS = 3000000  # 点数量
```

这样就在./CGH_Review/dataset/PointCloud/ 目录下的到了不同采样点数量下的点云数据

#### 3.光场数据

##### a.正交光场

运行./CGH_Review/dataset/mesh2LF_orth.py  一些可修改的参数如下（代码开头位置）

```python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_MESH_DIR = "Mesh/BunnyDragonRGB.obj"  #输入的obj数据不用动
OUTPUT_PCD_DIR = "LightField/LF_orth_40x40_rgb_fix_400" #输出的正交光场目录

OBJ_PATH     = os.path.join(SCRIPT_DIR, INPUT_MESH_DIR)
OUT_DIR      = os.path.join(SCRIPT_DIR, OUTPUT_PCD_DIR)

# Matlab 示例里的 Nu,Nv（角度采样数）
NU = 40   # u
NV = 40  # vwa

# Matlab 示例里的 Ns,Nt（每张图分辨率）：temp=zeros(Nt,Ns,Nv,Nu)
IMG_W = 400   # Ns
IMG_H = 400   # N
```

##### b.透视光场

运行./CGH_Review/dataset/mesh2LF_pers.py  一些可修改的参数如下（代码开头位置）

```python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_MESH_DIR = "Mesh/BunnyDragonRGB.obj"
OUTPUT_DIR     = "LightField/LF_pers_30x30_rgb_fix_400_50_53mm_parallel"

OBJ_PATH = os.path.join(SCRIPT_DIR, INPUT_MESH_DIR)
OUT_DIR  = os.path.join(SCRIPT_DIR, OUTPUT_DIR)

IMG_W = 400  #可修改的图像长
IMG_H = 400  #可修改的图像宽
FOV_DEG = 10.0

NU = 30 #可修改的视角数
NV = 30 #可修改的视角数
```



### 二、生成各种数据的全息图（以彩色重建为主，灰色重建就先不写了）

#### 1.层化数据

运行./CGH_Review/model/Layer_RGB_ComplexHologram.py 一些可修改的参数如下（代码开头位置）

```python
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(SCRIPT_DIR)

    DATA_ROOT = os.path.join(PARENT_DIR, "dataset/Layer/BunnyDragonColor_RGBD") #输入数据目录
    OUT_ROOT  = os.path.join(PARENT_DIR, "result/Layer/RGBD_complexRGB_20bins_50_53mm") #输出数据目录

    DIR_BIN  = os.path.join(OUT_ROOT, "Binned")
    DIR_HOLO = os.path.join(OUT_ROOT, "Hologram")
    DIR_RECON= os.path.join(OUT_ROOT, "Recon")

    # 图像尺寸
    TARGET_SIZE = (2048, 2048)  # (H, W)

    # 深度（mm）——不要动
    Z_MIN_MM = 50.0
    Z_MAX_MM = 53.0
    NUM_BINS = 20  #这个是切片数量，可以修改
```

##### a.生成RGBD数据全息图

输入路径：dataset/Layer/BunnyDragonColor_RGBD

输出路径：result/Layer/RGBD_complexRGB_20bins_50_53mm （这里是20层，所以中间是20bins，如果要修改为3层的话，可以改为3bins，类似于这样）

##### b.生成LDI数据全息图

输入路径：dataset/Layer/BunnyDragonColor_single_LDI（单层LDI）或者 dataset/Layer/BunnyDragonColor_double_LDI

输出路径:  result/Layer/LDI_double_complexRGB_20bins_50_53mm 或者 result/Layer/LDI_single_complexRGB_20bins_50_53mm

（总之就是输出路径名字可以随便改，但是一些参数要在输出路径名字中记录一下，要不然会忘记，比如多少层xxbins)

##### 输出结构

```python
./CGHReview/result/Layer/RGBD_complexRGB_20bins_50_53mm(拿这个举例)
    ├── Binned/      #不用管
    ├── Hologram/    #存放全息图的目录
         ├── amp_R(G/B).png(.npy) #RGB三个通道的振幅全息图的npy与png格式文件，共6个
         ├── pha_R(G/B).npy       #RGB三个通道的相位全息图的npy格式文件，共3个
         ├── phase_R(G/B).png     #RGB三个通道的相位全息图的png格式文件，共3个
         └── holo_R(G/B).npy      #RGB三通道的复全息图的npy格式文件，共3个
    └── Recon/      #测试用的生成不同层的重建结果（下面会用统一代码生成重建结果）
```

#### 2.点云数据

运行./CGH_Review/model/PointCloud_RGB_ComplexHologram.py 一些可修改的参数如下（代码开头位置）

```python
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(SCRIPT_DIR)

    DATA_PATH = os.path.join(PARENT_DIR, "dataset/PointCloud/PointCloud_3000000/bunnydragonRGB_pointcloud_color_3e7.ply")
    OUT_ROOT  = os.path.join(PARENT_DIR, "result/PointCloud/PCD_complexRGB_20bins_3e7_50_53mm")
    DIR_HOLO  = os.path.join(OUT_ROOT, "Hologram")
    DIR_RECON = os.path.join(OUT_ROOT, "Recon")
    
    TARGET_SIZE = (2048, 2048)   # (H,W)
    OUT_SIZE    = 2048

    PIXEL_PITCH = 5e-6
    WAVELENGTH  = 532e-9

    Z_MIN_MM = 50.0
    Z_MAX_MM = 53.0
    NUM_BINS = 20  #网格化生成点云全息的网格数
```

输入路径：dataset/PointCloud/PointCloud_3000000/bunnydragonRGB_pointcloud_color_3e7.ply (其他采样率的使用其他路径)

输出路径：result/PointCloud/PCD_complexRGB_20bins_3e7_50_53mm （这里采用20个网格，所以中间是20bins, 3e7的采样率，所以是3e7）

（同样的输出路径名字可以随便改，但是一些参数要在输出路径名字中记录一下。另外这里20网格效果不错，可以不动这个20）

##### 输出结构

```python
./CGHReview/result/PointCloud/PCD_complexRGB_20bins_3e6_50_53mm(拿这个举例)
    ├── Binned/      #不用管
    ├── Hologram/    #存放全息图的目录
         ├── amp_R(G/B).png(.npy)      #RGB三个通道的振幅全息图的npy与png格式文件，共6个 
         ├── phase_R(G/B).png(.npy)    #RGB三个通道的相位全息图的npy与png格式文件，共6个
         └── holo_R(G/B).npy           #RGB三通道的复全息图的npy格式文件，共3个
    └── Recon/      #测试用的生成不同层的重建结果（下面会用统一代码生成重建结果）
```

#### 3.光场数据

##### a.正交光场

运行./CGH_Review/model/LightField_Orth_RGB_ComplexHologram.py 一些可修改的参数如下（代码开头位置）

```python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

DIR_NAME = os.path.join(PARENT_DIR, "dataset/LightField/LF_orth_20x20_rgb_fix_400/images") #输入正交光场目录
FILE_EXT = "png"

OUT_DIR = os.path.join(PARENT_DIR, "result/LightField/orth_RGB_complexHologram_20_400") #输出正交光场目录
os.makedirs(OUT_DIR, exist_ok=True)

Nu = 20   # angular samples in u 采样数
Nv = 20   # angular samples in v 采样数
Ns = 400  # width  of each view image #分辨率
Nt = 400  # height of each view image #分辨率
```

输入路径：dataset/LightField/LF_orth_20x20_rgb_fix_400/images （不同的采样数与分辨率可以选择不同的输入路径）

输出路径：result/LightField/orth_RGB_complexHologram_20_400 （同上，这里的_20_400就是采样数与分辨率）

（🔔提示，最终重建图像的分辨率大小为采样数*分辨率，这里就是20x400=8000很大，建议400的分辨率不变，采样数最多到40）

###### 输出结构 

```python
./CGHReview/result/LightField/orth_RGB_complexHologram_20_400(拿这个举例)
    ├── Hologram/    #存放全息图的目录
         ├── hologram_amp_R(G/B).png(.npy)      #RGB三个通道的振幅全息图的npy与png格式文件，共6个 
         ├── hologram_phase_R(G/B).png(.npy)    #RGB三个通道的相位全息图的npy与png格式文件，共6个
         └── hologram_complex_R(G/B).npy           #RGB三通道的复全息图的npy格式文件，共3个
    └── Recon/      #测试用的生成不同层的重建结果（下面会用统一代码生成重建结果）
```

##### b.透视光场

运行./CGH_Review/model/LightField_Pers_RGB_ComplexHologram.py 一些可修改的参数如下（代码开头位置）

```python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
LF_ROOT = os.path.join(PARENT_DIR, "dataset", "LightField", "LF_pers_40x40_rgb_fix_400_50_53mm_parallel") #输入透视光场目录
IMAGES_SUBDIR = "images"
POSES_CSV = "poses.csv"

# ---- 输出 ----
OUT_DIR = os.path.join(PARENT_DIR, "result/LightField/pers_RGB_ComplexHologram_40_400") #输出透视光场目录

```

输入路径：dataset/LightField/LF_pers_20x20_rgb_fix_400/images （不同的采样数与分辨率可以选择不同的输入路径）

输出路径：result/LightField/pers_RGB_complexHologram_20_400 （同上，这里的_20_400就是采样数与分辨率）

###### 输出结构

```python
./CGHReview/result/LightField/pers_RGB_complexHologram_20_400(拿这个举例)
    ├── Hologram/    #存放全息图的目录
         ├── hologram_amp_R(G/B).png(.npy)      #RGB三个通道的振幅全息图的npy与png格式文件，共6个 
         ├── hologram_phase_R(G/B).png(.npy)    #RGB三个通道的相位全息图的npy与png格式文件，共6个
         └── hologram_complex_R(G/B).npy           #RGB三通道的复全息图的npy格式文件，共3个
    └── Recon/      #测试用的生成不同层的重建结果（下面会用统一代码生成重建结果）
```



### 三、统一框架针对生成的全息图进行重建

#### 1.层重建

运行代码 ./CGHReview/model/Reconstruction_RGB_ASM.py

##### a.层化

```bash
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
```

⚠️注意事项：

1. 前面的路径要改成你自己的

2. --outdir要定位到输入全息图的路径下，具体结构如下

   ```python
   ./CGHReview/result/Layer/RGBD_complexRGB_3bins_50_53mm(拿这个举例)
       ├── Binned/      #不用管
       ├── Hologram/    #存放全息图的目录
            ├── amp_R(G/B).png(.npy) #RGB三个通道的振幅全息图的npy与png格式文件，共6个
            ├── pha_R(G/B).npy       #RGB三个通道的相位全息图的npy格式文件，共3个
            ├── phase_R(G/B).png     #RGB三个通道的相位全息图的png格式文件，共3个
            └── holo_R(G/B).npy      #RGB三通道的复全息图的npy格式文件，共3个
       ├── Recon/                    #测试用的生成不同层的重建结果（下面会用统一代码生成重建结果）
       └── ASM_ReconFromHologram_0.9 #这里的outdir
   ```

3. --gamma是调整重建亮度的，越小亮度越高，原始是1，一般0.9或者0.8会好一点，用原始的也可以

4. RGB/LDI数据都可以用这个命令，注意修改路径

5. wavelength/pitch/z范围/out_size大小不用调整

##### b.点云

```bash
  python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_ASM.py \
 --amp_r /workspace/yh/project/CGHReview/result/PointCloud/PCD_complexRGB_20bins_3e7_50_53mm/Hologram/amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/PointCloud/PCD_complexRGB_20bins_3e7_50_53mm/Hologram/phase_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/PointCloud/PCD_complexRGB_20bins_3e7_50_53mm/Hologram/amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/PointCloud/PCD_complexRGB_20bins_3e7_50_53mm/Hologram/phase_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/PointCloud/PCD_complexRGB_20bins_3e7_50_53mm/Hologram/amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/PointCloud/PCD_complexRGB_20bins_3e7_50_53mm/Hologram/phase_B.npy \
  --outdir /workspace/yh/project/CGHReview/result/PointCloud/PCD_complexRGB_20bins_3e7_50_53mm/ASM_ReconFromHologram_0.9 \
  --wavelength 532e-9 \
  --pitch 5e-6 \
  --zmin 0.05 --zmax 0.053 --step 0.0001 \
  --gamma 0.9 \
  --out_size 2048
```

⚠️注意事项：

这里和层化差不多，一些参数会有细微的改变（这个不用管），只要注意你的路径地址就可以了

##### c.光场

###### I 透视光场

```bash
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
```

###### II 正交光场

```bash
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
```

⚠️注意事项：

1. 同样的还是注意路径地址和gamma，其他不用修改

2. 透视光场参数和正交光场参数有细微差别，用命令行的时候注意区分

3. 光场重建出来的结果偏小，整体图像占居空间小，需要裁剪放大处理，采用./CGHReview/model/utils.py

   ```bash
   python /workspace/yh/project/CGHReview/model/utils.py \
     --in_dir  /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/ASM_ReconFromHologram_0.9/z_-0.052000m \
     --out_dir /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/ASM_ReconFromHologram_0.9/z_-0.052000m_crop \
     --frac 0.1
   ```

   --frac是裁剪区域大小，越大裁剪的边上越多，可以用0.1或者0.15，自行调整



#### 2.多视重建

运行代码 ./CGHReview/model/Reconstruction_RGB_Multiview.py

##### a.层化

```bash
python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_mutiview.py \
 --amp_r /workspace/yh/project/CGHReview/result/Layer/LDI_single_complexRGB_20bins_50_53mm/Hologram/amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/Layer/LDI_single_complexRGB_20bins_50_53mm/Hologram/pha_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/Layer/LDI_single_complexRGB_20bins_50_53mm/Hologram/amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/Layer/LDI_single_complexRGB_20bins_50_53mm/Hologram/pha_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/Layer/LDI_single_complexRGB_20bins_50_53mm/Hologram/amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/Layer/LDI_single_complexRGB_20bins_50_53mm/Hologram/pha_B.npy \
  --outdir /workspace/yh/project/CGHReview/result/Layer/LDI_single_complexRGB_20bins_50_53mm/Multiview_ReconFromHologram_0.9 \
 --wavelength 532e-9 \
 --pitch 4e-6 \
 --zmin 0.05 --zmax 0.052 --step 0.0005 \
 --gamma 0.9 \
 --out_size 2048 \
 --view_grid 9 \
 --aperture_ratio 0.35 \
 --max_shift_ratio 0.8 \
 --save_mosaic
```

⚠️注意事项：

1. 前面的路径要改成你自己的

2. --outdir要定位到输入全息图的路径下，具体结构如下

```python
./CGHReview/result/Layer/RGBD_complexRGB_3bins_50_53mm(拿这个举例)
    ├── Binned/      #不用管
    ├── Hologram/    #存放全息图的目录
         ├── amp_R(G/B).png(.npy)        #RGB三个通道的振幅全息图的npy与png格式文件，共6个
         ├── pha_R(G/B).npy              #RGB三个通道的相位全息图的npy格式文件，共3个
         ├── phase_R(G/B).png            #RGB三个通道的相位全息图的png格式文件，共3个
         └── holo_R(G/B).npy             #RGB三通道的复全息图的npy格式文件，共3个
    ├── Recon/                           #测试用的生成不同层的重建结果（下面会用统一代码生成重建结果）
    ├── ASM_ReconFromHologram_0.9/       #层化重建的结果
    └── Multiview_ReconFromHologram_0.9/ #这里的--outdir
          ├── z+0.050000m                #不同距离下的多视重建结果，但我觉得结果差不多，可以看注意事项7
          ├── z+0.050500m
          ├── z_0.051000m
          ...
```

3. --gamma还是调整亮度的，和之前一样

4. --view_grid 视角数量，这里9代表9*9的视角（这个参数合适，可以不用调整了）
5. --max_shift_ratio 视角最大偏移量 (这里0.8能明显看出多视角的偏移)

6. 这个生成时间很长，代码是按照不同距离生成的多视图，但是我感觉不同距离的多视结果是差不多的,所以可以生成一个距离的多视图就能看结果。

##### b.点云

```bash
 python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_mutiview.py \
 --amp_r /workspace/yh/project/CGHReview/result/PointCloud/PCD_complexRGB_20bins_3e7_50_53mm/Hologram/amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/PointCloud/PCD_complexRGB_20bins_3e7_50_53mm/Hologram/phase_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/PointCloud/PCD_complexRGB_20bins_3e7_50_53mm/Hologram/amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/PointCloud/PCD_complexRGB_20bins_3e7_50_53mm/Hologram/phase_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/PointCloud/PCD_complexRGB_20bins_3e7_50_53mm/Hologram/amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/PointCloud/PCD_complexRGB_20bins_3e7_50_53mm/Hologram/phase_B.npy \
 --outdir /workspace/yh/project/CGHReview/result/PointCloud/PCD_complexRGB_20bins_3e7_50_53mm/Multiview_ReconFromHologram \
 --wavelength 532e-9 \
 --pitch 5e-6 \
 --zmin 0.051 --zmax 0.052 --step 0.0005 \
 --gamma 0.9 \
 --out_size 2048 \
 --view_grid 9 \
 --aperture_ratio 0.35 \
 --max_shift_ratio 0.8 \
 --save_mosaic
```

##### c.光场

###### I 正交光场

```bash
python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_mutiview.py \
 --amp_r /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Hologram/hologram_amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Hologram/hologram_phase_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Hologram/hologram_amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Hologram/hologram_phase_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Hologram/hologram_amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Hologram/hologram_phase_B.npy \
 --outdir /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Multiview_ReconFromHologram_0.9 \
 --wavelength 532e-9 \
 --pitch 2e-6 \
 --zmin -0.052 --zmax -0.050 --step 0.0005 \
 --gamma 0.9 \
 --out_size 2048 \
 --view_grid 9 \
 --aperture_ratio 0.35 \
 --max_shift_ratio 0.8 \
 --save_mosaic
```

######  II 透视光场

```bash
python /workspace/yh/project/CGHReview/model/Reconstruction_RGB_mutiview.py \
 --amp_r /workspace/yh/project/CGHReview/result/LightField/pers_RGB_ComplexHologram_20_400/Hologram/hologram_amp_R.npy \
 --pha_r /workspace/yh/project/CGHReview/result/LightField/pers_RGB_ComplexHologram_20_400/Hologram/hologram_phase_R.npy \
 --amp_g /workspace/yh/project/CGHReview/result/LightField/pers_RGB_ComplexHologram_20_400/Hologram/hologram_amp_G.npy \
 --pha_g /workspace/yh/project/CGHReview/result/LightField/pers_RGB_ComplexHologram_20_400/Hologram/hologram_phase_G.npy \
 --amp_b /workspace/yh/project/CGHReview/result/LightField/pers_RGB_ComplexHologram_20_400/Hologram/hologram_amp_B.npy \
 --pha_b /workspace/yh/project/CGHReview/result/LightField/pers_RGB_ComplexHologram_20_400/Hologram/hologram_phase_B.npy \
 --outdir /workspace/yh/project/CGHReview/result/LightField/pers_RGB_ComplexHologram_20_400/Multiview_ReconFromHologram_0.9 \
 --wavelength 532e-9 \
 --pitch 2e-6 \
 --zmin -0.051 --zmax -0.049 --step 0.0005 \
 --gamma 0.9 \
 --out_size 2048 \
 --view_grid 9 \
 --aperture_ratio 0.35 \
 --max_shift_ratio 0.8 \
 --save_mosaic
```

⚠️注意事项
同样的，光场重建出来的结果偏小，整体图像占居空间小，需要裁剪放大处理，采用./CGHReview/model/utils.py

```bash
python /workspace/yh/project/CGHReview/model/utils.py \
  --in_dir  /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Multiview_ReconFromHologram_0.9/z_-0.052000m \
  --out_dir /workspace/yh/project/CGHReview/result/LightField/orth_RGB_complexHologram_20_400/Multiview_ReconFromHologram_0.9/z_-0.052000m_crop \
  --frac 0.15
```

--frac是裁剪区域大小，越大裁剪的边上越多，可以用0.1或者0.15，自行调整

## ⚠️整体注意事项

重建出来的结果需要挑选，比如50mm-53mm距离内，并不是所有数据的结果都是选择50mm、51mm、52mm、53mm处的重建结果，因为方法不同，并且同一种方法中的选择参数不同也会对结果有影响（代码原因（调整可能会出bug又要改半天），可能会出现53mm结果在前，50mm结果在后，要具体去看重建的结果），可以挑选（50mm:兔子清晰的图像，51mm:龙头清晰的图像, 52mm:龙尾清晰的图像,53mm再往后虚的图像）
