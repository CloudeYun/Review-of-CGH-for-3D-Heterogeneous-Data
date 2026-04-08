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

生成层化数据代码：[Layer_data](https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/dataset/Obj2Layer.py)

生成层化数据全息图代码：[Layer_hologram](https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/model/Layer_ComplexHologram.py)

详细指导内容：[Layer_README](https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/README/Layer.md)

层重建结果：50mm --------------------> 53mm

<div align="center">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Layer_resize/layer_recon_1.png" width="30%" alt="img1">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Layer_resize/layer_recon_2.png" width="30%" alt="img2">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Layer_resize/layer_recon_3.png" width="30%" alt="img3">
</div>

多视重建结果

<div align="center">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Layer/Layer_multiview_recon.png" width="80%" alt="img1">
</div>

层化数据的主要特点在于其将三维场景表示为若干沿深度方向分布的二维切片，从而将复杂的三维全息生成问题转化为多个二维层面的处理与传播过程。因此对于层化数据，实验部分采用了基于深度分层表示的全息图生成方法，即首先将三维场景沿深度方向表示为若干二维图像切片并记录深度信息，再将各深度层上的光场信息分别通过ASM传播至全息面并进行叠加。

基于这种表示方式，层化数据在计算流程上较为清晰，物理意义直观，实现难度相对较低，因此尤其适用于深度结构较为明确、遮挡关系相对简单、且需要突出聚焦与离焦变化效果的场景，例如包含多平面结构的室内场景、墙体场景以及桌面类场景等。结合实验结果可以看出，层化数据能够较为直观地反映不同深度平面对应的成像差异，在焦点切换与轴向层次表达方面具有一定优势。

然而，层化数据的局限性也较为明显。从层重建的结果看，由于其本质上是对连续三维场景进行离散分层表示，因此在不同层之间的过渡区域容易出现较强的层间割裂感，尤其在物体边界和深度变化较连续的区域，这种不连续性会更加明显。同时，从连续视角重建结果来看，层化数据在视差连续性的表达方面表现相对有限，难以充分保持复杂场景中的连续表面细节与自然视角变化。因此，在需要高质量表面连续性、复杂遮挡关系建模以及平滑视差呈现的应用中，层化数据通常并不是最优选择。

### 2.点云数据 ❄️

生成点云数据代码：[PointCloud_data](https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/dataset/Obj2PointCloud.py)

生成点云数据全息图代码：[PointCloud_hologram](https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/model/PointCloud_ComplexHologram.py)   

详细指导内容：[PointCloud_README](https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/README/PointCloud.md)  

层重建结果：50mm --------------------> 53mm

300k采样率

<div align="center">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/PointCloud_resize/PCD_recon_1_3e5.png" width="30%" alt="img1">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/PointCloud_resize/PCD_recon_2_3e5.png" width="30%" alt="img2">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/PointCloud_resize/PCD_recon_3_3e5.png" width="30%" alt="img3">
</div>

3M采样率

<div align="center">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/PointCloud_resize/PCD_recon_1_3e6.png" width="30%" alt="img1">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/PointCloud_resize/PCD_recon_2_3e6.png" width="30%" alt="img2">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/PointCloud_resize/PCD_recon_3_3e6.png" width="30%" alt="img3">
</div>

30M采样率

<div align="center">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/PointCloud_resize/PCD_recon_1_3e7.png" width="30%" alt="img1">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/PointCloud_resize/PCD_recon_2_3e7.png" width="30%" alt="img2">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/PointCloud_resize/PCD_recon_3_3e7.png" width="30%" alt="img3">
</div>

多视重建结果

<div align="center">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/PointCloud/PCD_multiview_recon.png" width="80%" alt="img1">
</div>

点云数据的突出特点在于其对真实三维场景采集具有良好的适配性，能够直接承接激光雷达、三维扫描以及多视几何重建等方法得到的原始输出结果，因此在“从现实世界场景获取到全息显示表达”的技术链路中具有较强的实用价值。因此，对于点云数据，实验部分采用了WRP全息图生成方法，即首先将三维点云按照空间位置映射到目标成像平面，并结合深度信息将其划分到不同轴向位置的稀疏深度层中，再将各深度层上的点云光场信息分别通过 ASM 传播至全息面并进行叠加，从而完成三维场景的全息编码与重建。

基于这种表示方式，点云数据能够在不显式构建表面或体结构的前提下，直接保留场景的空间分布信息，因而在面向真实场景快速建模与显示时具有明显优势。结合实验结果可以看出，点云数据能够较好地恢复场景的整体空间结构，并且随着采样点数的增加，重建结果的轮廓完整性与细节表现也会逐步提升。因此，点云数据特别适用于三维扫描、空间测绘、实时环境感知以及自动驾驶感知结果可视化等应用场景。

然而，点云数据的局限性同样较为突出。由于其本质上属于非结构化离散表示，且点分布通常存在密度不均匀的问题，因此全息重建质量往往对采样密度具有较强依赖性。当采样点数较少时，重建结果容易出现结构不完整、轮廓模糊以及细节缺失等现象；而当点数进一步增大时，虽然可以在一定程度上改善重建质量，但同时也会显著增加数据存储开销与计算负担。因此，如何在保证点云数据原始真实性与表达灵活性的基础上，进一步缓解非结构化离散采样带来的质量退化问题，并降低高密度点云所导致的存储与计算压力，仍需要依赖更高效的建模、采样优化与全息生成算法来加以解决。

### 3.光场数据 ⚡️

生成光场数据代码：[LightField_data_orth](https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/dataset/Obj2LF_orth.py)
[LightField_data_pers](https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/dataset/Obj2LF_pers.py)        

生成光场数据全息图代码：[LightField_hologram_orth](https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/model/LightField_Orth_ComplexHologram.py)
[LightField_hologram_pers](https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/model/LightField_Pers_ComplexHologram.py)

详细指导内容：[LightField_README](https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/README/LightField.md)

层重建结果：50mm --------------------> 53mm

正交光场

<div align="center">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/LightField_resize/LF_orth_recon_1.png" width="30%" alt="img1">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/LightField_resize/LF_orth_recon_2.png" width="30%" alt="img2">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/LightField_resize/LF_orth_recon_3.png" width="30%" alt="img3">
</div>

透视光场

<div align="center">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/LightField_resize/LF_pers_recon_1.png" width="30%" alt="img1">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/LightField_resize/LF_pers_recon_2.png" width="30%" alt="img2">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/LightField_resize/LF_pers_recon_3.png" width="30%" alt="img3">
</div>

多视重建结果

<div align="center">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/LightField/LF_multiview_recon.png" width="80%" alt="img1">
</div>

光场数据在数据表示层面的突出优势在于其天然携带丰富的多视角与视差信息。与更侧重场景几何结构描述的数据形式不同，光场数据从获取之初便直接对应于不同观察方向上的光线分布，因此更贴近真实视觉感知过程，能够更自然地表征视角变化所带来的成像差异。因此，对于光场数据，实验部分采用了基于光场重聚焦与波动传播相结合的全息图生成方法，即首先对多视角子视图进行视差对齐与重聚焦处理，使特定参考深度处的场景信息在空间上得到统一表达，再将重聚焦后的光场进一步转换为适于全息编码的复振幅光场，并通过 ASM 传播至全息面，从而完成三维场景的全息图生成与重建。

基于这种表示方式，光场数据能够较好地保留场景的方向信息与视角信息，在表达空间层次关系和连续观察效果方面具有天然优势。结合层化重建与多视角重建结果可以看出，光场数据在不同深度平面上的焦点区域呈现以及多视角视差效果表达方面均表现出较好的性能，能够较为有效地反映场景的空间层次关系和视角连续变化特征。因此，在全息生成任务中，光场数据特别适用于强调视角连续性、多视点观察、裸眼三维显示以及视差驱动显示等应用场景；同时，对于前后遮挡关系较为复杂、需要综合保留空间层次与方向信息的三维场景，光场数据同样具有较强的适应性。

然而，光场数据的不足也较为明显。一方面，为了充分表征场景在不同方向上的光线信息，通常需要采集或存储大量子视图数据，因此其数据规模较大，存储与传输开销较高。另一方面，从实验结果来看，光场数据在全息重建过程中仍会出现一定程度的边界伪影与局部模糊现象，这在一定程度上影响了重建图像的细节保真度。因此，如何在保持光场视差与多视角优势的同时，进一步抑制重建伪影、提升边界质量与重建清晰度，是后续光场全息生成方法中值得重点关注的问题。

### 4.Mesh数据 ▶️

生成Mesh数据代码：[Mesh_data](https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/dataset/Obj2Mesh.py)

生成Mesh数据全息图代码：[Mesh_hologram](https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/model/Mesh_ComplexHologram.py)

详细指导内容：[Mesh_README](https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/README/Mesh.md)

层重建结果: 50mm --------------------> 53mm

<div align="center">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Mesh_resize/Mesh_recon_1.png" width="30%" alt="img1">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Mesh_resize/Mesh_recon_2.png" width="30%" alt="img2">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Mesh_resize/Mesh_recon_3.png" width="30%" alt="img3">
</div>

层重建结果：1000mm --------------------> 1060mm(物体等比例扩大20倍)

<div align="center">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Mesh_resize/Mesh_recon_20_1.png" width="30%" alt="img1">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Mesh_resize/Mesh_recon_20_2.png" width="30%" alt="img2">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Mesh_resize/Mesh_recon_20_3.png" width="30%" alt="img3">
</div>

层重建结果：1000mm --------------------> 1060mm(物体等比例扩大40倍)

<div align="center">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Mesh_resize/Mesh_recon_40_1.png" width="30%" alt="img1">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Mesh_resize/Mesh_recon_40_2.png" width="30%" alt="img2">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Mesh_resize/Mesh_recon_40_3.png" width="30%" alt="img3">
</div>

层重建结果：5000mm --------------------> 5300mm(物体等比例扩大100倍)

右转 0 degree

<div align="center">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Mesh_resize/Mesh_recon_d0_1.png" width="30%" alt="img1">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Mesh_resize/Mesh_recon_d0_2.png" width="30%" alt="img2">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Mesh_resize/Mesh_recon_d0_3.png" width="30%" alt="img3">
</div>

右转 3 degrees

<div align="center">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Mesh_resize/Mesh_recon_d3_1.png" width="30%" alt="img1">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Mesh_resize/Mesh_recon_d3_2.png" width="30%" alt="img2">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Mesh_resize/Mesh_recon_d3_3.png" width="30%" alt="img3">
</div>

右转 5 degrees

<div align="center">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Mesh_resize/Mesh_recon_d5_1.png" width="30%" alt="img1">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Mesh_resize/Mesh_recon_d5_2.png" width="30%" alt="img2">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Mesh_resize/Mesh_recon_d5_3.png" width="30%" alt="img3">
</div>

多视重建结果

20倍

<div align="center">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Mesh/Mesh_multiview_recon.png" width="80%" alt="img1">
</div>

Mesh 数据的突出优势在于其具有明确的几何拓扑结构，场景表面由连续的面片进行描述，且表面法向、材质属性以及面片间连接关系等信息均可被有效利用。因此，对于 Mesh 数据，实验部分采用了基于三角面片表面表示的波面记录平面方法，即首先将三维目标表示为由多个彩色三角面片构成的连续表面，再将各面片作为基本的光学作用单元，通过旋转操作使其平行于全息面，然后对其在全息面上的衍射贡献进行建模与叠加，从而完成三维场景的全彩全息图生成与重建。

基于这种表示方式，Mesh 数据在全息生成过程中更有利于保持目标的结构完整性与表面连续性，从而更容易获得轮廓清晰、形态稳定、物体感较强的重建结果。结合实验结果可以看出，Mesh 数据在轮廓完整性、表面连续性以及整体结构清晰度等方面均表现出较好的稳定性，能够较为准确地反映场景的表面几何特征。因此，Mesh 数据更适用于表面结构复杂但整体连续、细节较为丰富的三维场景，例如 CAD 模型、计算机图形学对象、工业零件以及具有规则表面特征的目标的全息重建任务。

然而，Mesh数据的局限性同样不可忽视。由于其本质上仍是基于离散面片对连续表面的近似表示，因此在全息重建过程中会受到面片离散程度、采样分辨率以及传播模型精度等因素的共同影响，进而产生较明显的波纹效应、纹理失真和局部伪影等问题。此外，从多视角重建结果来看，现有方法在 Mesh数据视差信息的表达上仍存在一定不足，尚不能充分发挥 Mesh 数据在视角变化建模方面的潜在优势。因此，如何在保持Mesh几何结构优势的基础上，进一步抑制重建伪影并增强其多视角视差表现，是后续Mesh全息生成方法值得深入研究的方向。

### 5.体素数据 ♦️

生成体素数据代码：[Voxel_data](https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/dataset/Obj2Vox.py)

生成体素数据全息图代码：[Voxel_hologram](https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/model/Voxel_ComplexHologram.py)

详细指导内容：[Voxel_README](https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/README/Voxel.md)

层重建结果：50mm --------------------> 53mm

<div align="center">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Voxel_resize/Voxel_recon_1.png" width="30%" alt="img1">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Voxel_resize/Voxel_recon_2.png" width="30%" alt="img2">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Voxel_resize/Voxel_recon_3.png" width="30%" alt="img3">
</div>

多视重建结果

<div align="center">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/Voxel/Voxel_multiview_recon.png" width="80%" alt="img1">
</div>

体素数据的主要特点在于其从“空间占据”与“体积属性”的角度对三维场景进行建模，因此在表达目标的内部结构、体密度分布以及半透明体等方面具有天然优势。相较于更侧重表面几何描述的数据形式，体素数据能够更完整地表征三维空间中的体信息。因此，对于体素数据，实验部分采用了基于体素空间占据表示与波动传播相结合的全息图生成方法，即首先依据体素在三维空间中的分布关系，按照轴向位置将场景信息组织到不同深度区间，并将对应体信息转换为可用于全息编码的二维光场表示，再通过 ASM 传播至全息面并进行叠加，从而完成三维场景的全息图生成与重建。

基于这种表示方式，体素数据能够较好地保留场景的体积占据信息，在描述内部结构与体渲染特性方面具有明显优势，因此更适用于医学体数据、CT/MRI 等具有内部组织结构特征的三维场景的全息重建任务。对于需要体现体渲染特性或关注目标内部信息分布的应用，体素表示同样具有较强的适应性和潜在优势。

然而，由实验结果可知，体素数据的局限性同样较为明显。由于其本质上是由离散体元构成的体积表示，因此在全息重建过程中更容易表现出较强的块状感，也更容易出现粗糙感和不连续的割裂现象。在本实验中，所选场景本质上仍以表面结构为主，因而体素数据在内部结构表达方面的优势未能得到充分体现。综合实验结果可以看出，体素数据更适用于强调内部结构表达和体渲染效果的应用场景，在处理具有丰富体信息的三维目标时具有更高价值；而对于以表面形状和边界细节为主要关注对象的普通实体场景，体素数据通常难以获得与表面型表示方法相当的重建效果。

### 6.NeRF数据 🧠

生成NeRF数据代码：

生成NeRF数据全息图代码：

详细指导内容：

层重建结果：50mm --------------------> 53mm

<div align="center">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/NeRF_resize/NeRF_recon1.png" width="30%" alt="img1">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/NeRF_resize/NeRF_recon2.png" width="30%" alt="img2">
  <img src="https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/result/NeRF_resize/NeRF_recon3.png" width="30%" alt="img3">
</div>

神经场数据以 NeRF 为代表，其主要特点在于利用一组二维图像及其对应的相机位姿，对场景中的体密度与视角相关辐射信息进行连续表示，从而在无需显式存储大量点、面或体元的情况下，实现对整个三维场景的紧凑表达与新视角合成。因此，对于神经场数据，实验部分采用了基于神经场重建与光场采样相结合的全息图生成方法，即首先通过环绕场景拍摄得到稀疏视角图像，并利用 Instant-NGP 对场景进行快速神经场建模；随后基于已训练的 NeRF 模型模拟光场相机阵列对场景进行多视角采样，得到对应的光场表示，再结合光场全息生成模型完成三维场景的全息图生成与重建。

基于这种表示方式，神经场数据能够以较为紧凑的形式保留场景的整体外观信息和连续视角变化特征，在场景完整性表达、自由视点合成以及存储效率方面具有明显优势。相较于传统显式三维表示，神经场不需要直接构建复杂的点、面或体元结构，便能够恢复场景的整体空间外观，因此特别适用于三维场景重建、自由视点显示以及对存储效率和场景完整性要求较高的全息显示前端表示任务。

然而，神经场数据的局限性同样较为明显。首先，当输入视图过于稀疏时，神经场对场景几何与辐射信息的估计精度会明显下降，进而在后续的光场采样与全息重建过程中表现为几何误差、边界伪影以及局部模糊等问题。其次，神经场本质上属于隐式表示，其结果更接近于可渲染的连续场景函数，而非可直接参与光学传播计算的显式物理表示，因此在用于全息生成时通常仍需要经过额外的采样、转换或中间表示步骤。在本实验所采用的流程中，这种中间转换具体体现为由 NeRF 进一步采样得到光场表示，再利用光场模型完成全息编码。虽然这一思路能够较好地衔接神经场表示与现有全息生成框架，但也会在一定程度上引入额外的采样误差与计算开销，并部分削弱神经场原本在存储紧凑性方面的优势。此外，尽管 Instant-NGP 在训练和渲染效率上较传统 NeRF 已有显著提升，但高质量神经场重建仍然依赖较强的计算资源支持。综合来看，神经场数据更适用于强调场景完整表示、连续视角合成和紧凑存储的应用场景，而在需要直接进行高精度光学传播建模时，仍需要结合更高效的表示转换与全息编码方法，才能充分发挥其在全息显示中的潜力。

### 7.重建 🪜

层重建代码：[Layer Reconstruction](https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/model/Reconstruction_layer.py)

多视角重建代码：[Multiview Reconstruction](https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/model/Reconstruction_mutiview.py)

详细指导内容: [Reconstruction_README](https://github.com/CloudeYun/Review-of-CGH-for-3D-Heterogeneous-Data/blob/main/README/Reconstruction.md)

------下面的内容待修改------

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
