# MobileNetV2 推理项目

演示完整的深度学习模型转换和性能优化流程：**TFLite → ONNX → TensorRT**/**NCNN**

## 📊 性能结果

在 RTX 3060 上的 MobileNet v2 推理性能对比（使用统一MobileNetV2预处理）：

```
🏆 === 性能对比结果 ===
TensorRT (INT8)     : 0.0014秒 ( 704.4 FPS) 🥇 [最快] - 5.7MB
TensorRT (FP16)     : 0.0015秒 ( 681.9 FPS) [1.0x slower] 🥈 [最快] - 9MB
TensorRT (FP32)     : 0.0016秒 ( 642.3 FPS) [1.1x slower] 🥉 [最快] - 16MB
ONNX (CPU)          : 0.0017秒 ( 593.0 FPS) [1.2x slower] [最快CPU] - 14MB
NCNN                : 0.0033秒 ( 300.0 FPS) [2.3x slower] [移动优化] - 14MB
NCNN (INT8)         : 0.0035秒 ( 287.9 FPS) [2.4x slower] [移动优化] - 3.5MB
TensorFlow Lite     : 0.0056秒 ( 177.9 FPS) [4.0x slower] [跨平台] - 14MB
```

**精度模式说明**：

- **TensorFlow Lite**：跨平台框架，移动端友好，完美精度 (99.7% goldfish 置信度)
- **ONNX Runtime (CPU)**：CPU优化，无GPU依赖，完美精度 (99.7% goldfish 置信度)
- **TensorRT (FP32)**：GPU加速，最快推理，完美精度 (99.7% goldfish 置信度)
- **TensorRT (FP16)**：GPU加速，混合精度，最快推理，完美精度 (99.7% goldfish 置信度)
- **NCNN**：CPU专用，移动端优化，完美精度 (99.7% goldfish 置信度)
- **TensorRT (INT8)**：GPU加速，最快推理，轻量模型，量化精度轻微损失 (99.5% goldfish 置信度)
- **NCNN (INT8)**：CPU专用，移动端优化，模型大小最小，量化精度轻微损失 (99.5% goldfish 置信度)

| 框架                | Top-1预测 | 置信度 | 精度状态               |
| ------------------- | --------- | ------ | ---------------------- |
| **TensorFlow Lite** | goldfish  | 99.7%  | ✅ 最高精度             |
| **ONNX Runtime**    | goldfish  | 99.7%  | ✅ 与TFLite一致         |
| **TensorRT (FP32)** | goldfish  | 99.7%  | ✅ 与TFLite一致         |
| **TensorRT (FP16)** | goldfish  | 99.7%  | ✅ 与TFLite一致         |
| **NCNN**            | goldfish  | 99.7%  | ✅ 完全一致             |
| **TensorRT (INT8)** | goldfish  | 99.2%  | ✅ 量化轻微精度损失0.5% |
| **NCNN (INT8)**     | goldfish  | 99.4%  | ✅ 量化精度轻微损失0.3% |

## 🛠️ 环境配置

### 系统要求
- Ubuntu 18.04+ / WSL2
- NVIDIA GPU (RTX 20/30/40系列) - 仅 TensorRT 需要
- CUDA 12.x + cuDNN - 仅 TensorRT 需要
- Python 3.10

### 依赖说明

```
TensorRT的依赖顺序应该是：
1. CUDA依赖检查 - 最基础的GPU和CUDA环境
2. TensorRT系统库路径 - libnvinfer.so等核心库
3. TensorRT Python包 - tensorrt模块
4. TensorRT工具检查 - trtexec等命令行工具

正确的NCNN依赖顺序：
1. protobuf依赖检查 - 基础依赖，NCNN工具需要这个库才能运行
2. NCNN库路径检查 - 运行时库路径，工具需要这些.so文件
3. NCNN工具检查 - 最后检查工具是否可执行
```

### 安装步骤

```bash
# 1. 创建虚拟环境
conda create -n py310_mobilenetv2 python=3.10 -y
conda activate py310_mobilenetv2

# 2. 安装依赖包
pip install -r requirements.txt
# numpy==1.26.4 - 核心数组计算
# pillow==10.4.0 - 图像处理
# tensorflow==2.16.1 - TFLite支持
# onnx==1.17.0 - ONNX模型格式
# onnxruntime==1.19.2 - ONNX推理 (CPU优化版)
# tflite2onnx==0.4.1 - 模型转换
# pycuda==2025.1.1 - CUDA Python绑定 (仅TensorRT需要) - 最基础的GPU和CUDA环境
pip install ncnn==1.0.20231027 --no-deps # 在安装ncnn时同时锁定所有依赖版本，否则会破坏其它依赖

# 3. 配置TensorRT系统库、Python包、工具
cd ~/work/depend_config/tensorrt
# 3.1 配置系统库 - libnvinfer.so等核心库
# 下载
wget -c -q --show-progress "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz"
# 解压
tar -xzf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
# 设置环境变量
export LD_LIBRARY_PATH=~/work/depend_config/tensorrt/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
# 3.2 配置Python包（使用本地wheel文件） - tensorrt模块
pip install ~/work/depend_config/tensorrt/TensorRT-8.6.1.6/python/tensorrt-8.6.1-cp310-none-linux_x86_64.whl
# 3.3 配置工具 - trtexec等命令行工具
# 设置环境变量
export PATH=~/work/depend_config/tensorrt/TensorRT-8.6.1.6/bin:$PATH
# 验证环境变量
echo $LD_LIBRARY_PATH
echo $PATH

# 4. 配置NCNN依赖、系统库、工具
cd ~/work/depend_config/ncnn
=========================================================================================
# 4.1 配置依赖 - protobuf是基础依赖，NCNN工具需要这个库才能运行
# 下载 protobuf 3.0 版本的 deb 包（NCNN 工具需要 libprotobuf.so.10 库，但系统通常只有更新版本）
wget http://archive.ubuntu.com/ubuntu/pool/main/p/protobuf/libprotobuf10_3.0.0-9.1ubuntu1_amd64.deb
# 解压 deb 包
ar x libprotobuf10_3.0.0-9.1ubuntu1_amd64.deb data.tar.xz
tar -xf data.tar.xz
# 验证库文件存在
ls -la usr/lib/x86_64-linux-gnu/libprotobuf.so.10*
# 设置环境变量
export LD_LIBRARY_PATH=~/work/depend_config/ncnn/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
# 4.2 配置系统库 - 运行时库路径，工具需要这些.so文件
# 下载适用于 Ubuntu 18.04 的 NCNN 预编译工具包
wget https://github.com/Tencent/ncnn/releases/download/20220216/ncnn-20220216-ubuntu-1804-shared.zip
# 解压
python -m zipfile -e ncnn-20220216-ubuntu-1804-shared.zip .
# 设置环境变量
export LD_LIBRARY_PATH=~/work/depend_config/ncnn/ncnn-20220216-ubuntu-1804-shared/lib:$LD_LIBRARY_PATH
# 4.3 配置工具 - onnx2ncnn、ncnnoptimize、ncnn2table、ncnn2int8等命令行工具
# 设置环境变量
export PATH=~/work/depend_config/ncnn/ncnn-20220216-ubuntu-1804-shared/bin:$PATH
# 给工具添加执行权限
chmod +x ncnn-20220216-ubuntu-1804-shared/bin/*
# 验证环境变量
echo $LD_LIBRARY_PATH
echo $PATH
=========================================================================================
# 4.1 配置依赖 - protobuf是基础依赖，NCNN工具需要这个库才能运行
cd ~/work/depend_config/ncnn
# 下载 protobuf 3.6 版本的 deb 包（NCNN 工具需要 libprotobuf.so.17 库，但系统通常只有更新版本）
wget http://archive.ubuntu.com/ubuntu/pool/main/p/protobuf/libprotobuf17_3.6.1.3-2ubuntu5_amd64.deb
# 解压 deb 包
ar x libprotobuf17_3.6.1.3-2ubuntu5_amd64.deb data.tar.xz
tar -xf data.tar.xz
# 验证库文件存在
ls -la usr/lib/x86_64-linux-gnu/libprotobuf.so.17*
# 设置环境变量
export LD_LIBRARY_PATH=~/work/depend_config/ncnn/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
# 4.2 配置系统库 - 运行时库路径，工具需要这些.so文件
# 下载适用于 Ubuntu 20.04 的 NCNN 预编译工具包
wget https://github.com/Tencent/ncnn/releases/download/20231027/ncnn-20231027-ubuntu-2004-shared.zip
# 解压
python -m zipfile -e ncnn-20231027-ubuntu-2004-shared.zip .
# 设置环境变量
export LD_LIBRARY_PATH=~/work/depend_config/ncnn/ncnn-20231027-ubuntu-2004-shared/lib:$LD_LIBRARY_PATH
# 这里要注意下载这个NCNN升级包后，发现符号链接被破坏，libncnn.so.1 变成了23字节的文本文件，要手动链回去
cd /home/xinxin/work/depend_config/ncnn/ncnn-20231027-ubuntu-2004-shared/lib
rm libncnn.so libncnn.so.1
ln -sf libncnn.so.1.0.20231027 libncnn.so.1
ln -sf libncnn.so.1 libncnn.so
# 这样就形成了标准的动态库链接结构：libncnn.so -> libncnn.so.1 -> libncnn.so.1.0.20231027
# 4.3 配置工具 - onnx2ncnn、ncnnoptimize、ncnn2table、ncnn2int8等命令行工具
# 设置环境变量
export PATH=~/work/depend_config/ncnn/ncnn-20231027-ubuntu-2004-shared/bin:$PATH
# 给工具添加执行权限
chmod +x ncnn-20231027-ubuntu-2004-shared/bin/*
# 验证环境变量
echo $LD_LIBRARY_PATH
echo $PATH
=========================================================================================

# 5. 验证环境
cd ~/work/mobilenetv2/01_python
python 01_check_deps.py
```

## 🚀 快速开始

按照编号顺序执行以下脚本：

### 环境检查
```bash
python 01_check_deps.py
```

### 模型和测试数据准备
```bash
# 下载模型和标签文件
cd ~/work/mobilenetv2/model
./download.sh
# 可以去download.sh脚本所填地址下载模型和标签，也可以去这个地方下载
https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn
https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt
```

### 模型转换
```bash
# 只要ONNX
python 02_convert_model.py --tflite ../model/mobilenet_v2_1.0_224.tflite --onnx

# 只要tensorrt fp32
python 02_convert_model.py --tflite ../model/mobilenet_v2_1.0_224.tflite --tensorrt-fp32

# 只要tensorrt fp16
python 02_convert_model.py --tflite ../model/mobilenet_v2_1.0_224.tflite --tensorrt-fp16

# 只要tensorrt int8 (需要校准数据集：单张图片或图片目录或图片列表文件)
python 02_convert_model.py --tflite ../model/mobilenet_v2_1.0_224.tflite --tensorrt-int8 --calibration-dataset ../input/fish_224x224.jpeg

# 只要NCNN
python 02_convert_model.py --tflite ../model/mobilenet_v2_1.0_224.tflite --ncnn

# 只要NCNN INT8量化 (需要校准数据集：单张图片或图片目录或图片列表文件)
python 02_convert_model.py --tflite ../model/mobilenet_v2_1.0_224.tflite --ncnn-int8 --calibration-dataset ../input/fish_224x224.jpeg

# 🔧 使用图片列表文件
# 创建图片列表文件
cat > ../input/dataset.txt << 'EOF'
# MobileNet校准图片列表
/home/xinxin/work/mobilenetv2/input/cat_224x224.jpg
/home/xinxin/work/mobilenetv2/input/dog_224x224.jpg
/home/xinxin/work/mobilenetv2/input/fish_224x224.jpeg
# 注释行会被忽略
EOF

# 要所有格式
python 02_convert_model.py --tflite ../model/mobilenet_v2_1.0_224.tflite --onnx --tensorrt-fp32 --tensorrt-fp16 --tensorrt-int8 --ncnn --ncnn-int8 --calibration-dataset ../input/dataset.txt
```

**校准数据集说明**：

- **单张图片**: `--calibration-dataset ../input/fish_224x224.jpeg` 
- **多图片目录**: `--calibration-dataset ../input` 
- **图片列表文件**: `--calibration-dataset ../input/dataset.txt` 
- **支持图片格式**: 图片格式 `.jpg/.png/.bmp/.tiff/.webp`
- **列表文件特性**: 支持注释行（#开头）、相对路径、空行跳过
- **数量限制**: TensorRT最多使用50张，NCNN最多使用50张（避免校准时间过长）
- **质量提升**: 多图片校准通常比单图片校准获得更好的量化精度

### 性能对比测试

需要先设置运行时库路径：

```bash
# 设置运行时库路径
export LD_LIBRARY_PATH="\
/home/xinxin/work/depend_config/tensorrt/TensorRT-8.6.1.6/lib:\
$LD_LIBRARY_PATH"

# 测试单独的tflite
python 03_benchmark_all.py \
    --tflite ../model/mobilenet_v2_1.0_224.tflite \
    --image ../input/fish_224x224.jpeg \
    --labels ../model/labels.txt
# 测试单独的onnx
python 03_benchmark_all.py \
    --onnx ../model/mobilenet_v2_1.0_224.onnx \
    --image ../input/fish_224x224.jpeg \
    --labels ../model/labels.txt

# 测试单独的TensorRT fp32精度
python 03_benchmark_all.py \
    --tensorrt-fp32 ../model/mobilenet_v2_1.0_224_fp32.trt \
    --image ../input/fish_224x224.jpeg \
    --labels ../model/labels.txt

# 测试单独的TensorRT fp16精度
python 03_benchmark_all.py \
    --tensorrt-fp16 ../model/mobilenet_v2_1.0_224_fp16.trt \
    --image ../input/fish_224x224.jpeg \
    --labels ../model/labels.txt

# 测试单独的TensorRT int8精度
python 03_benchmark_all.py \
    --tensorrt-int8 ../model/mobilenet_v2_1.0_224_int8.trt \
    --image ../input/fish_224x224.jpeg \
    --labels ../model/labels.txt

# 测试单独的NCNN
python 03_benchmark_all.py \
    --ncnn ../model/mobilenet_v2_1.0_224.param \
    --image ../input/fish_224x224.jpeg \
    --labels ../model/labels.txt

# 测试单独的NCNN INT8量化
python 03_benchmark_all.py \
    --ncnn-int8 ../model/mobilenet_v2_1.0_224-int8.param \
    --image ../input/fish_224x224.jpeg \
    --labels ../model/labels.txt

# 完整多后端性能对比 (一次运行所有后端)
python 03_benchmark_all.py \
    --tflite ../model/mobilenet_v2_1.0_224.tflite \
    --onnx ../model/mobilenet_v2_1.0_224.onnx \
    --tensorrt-fp32 ../model/mobilenet_v2_1.0_224_fp32.trt \
    --tensorrt-fp16 ../model/mobilenet_v2_1.0_224_fp16.trt \
    --tensorrt-int8 ../model/mobilenet_v2_1.0_224_int8.trt \
    --ncnn ../model/mobilenet_v2_1.0_224.param \
    --ncnn-int8 ../model/mobilenet_v2_1.0_224-int8.param \
    --image ../input/fish_224x224.jpeg \
    --labels ../model/labels.txt
```

## 🔥 RKNN推理 - 瑞芯微NPU专用

### 环境配置要求

**⚠️ 重要提醒**: RKNN工具链与其他推理框架环境冲突，**必须使用独立环境**！

```bash
# 创建RKNN专用虚拟环境
conda create -n py310_rknn python=3.10 -y
conda activate py310_rknn

# 安装RKNN-Toolkit2 (需要向瑞芯微申请SDK)
# requirements_cp310-2.3.2.txt
# pip install rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

### RKNN模型转换与推理

**支持目标平台**: RK3588/RK3588S/RK3566/RK3568等瑞芯微NPU芯片

```bash
# 激活RKNN专用环境
conda activate py310_rknn

# 运行RKNN转换和推理（需要已有ONNX模型）
python rknn_convert_inference.py
```

**功能说明**:
- **输入格式**: 从ONNX模型转换（需先运行`02_convert_model.py --onnx`）
- **目标平台**: RK3588 NPU芯片
- **量化方式**: 自动INT8量化（使用dataset.txt校准数据集）
- **预处理**: 统一MobileNetV2标准化（mean=127.5, std=127.5）
- **输出模型**: mobilenet_v2_1.0_224.rknn （NPU专用格式）

**性能特点**:
- **NPU加速**: 专为瑞芯微NPU优化，比CPU推理快数倍
- **低功耗**: NPU功耗远低于GPU，适合嵌入式设备
- **INT8量化**: 自动量化优化，模型体积小，推理速度快
- **精度分析**: 内置accuracy_analysis功能验证量化精度损失

**注意事项**:
1. **环境隔离**: 必须在py310_rknn虚拟环境中运行，与主项目环境完全分离
2. **硬件依赖**: 仅适用于瑞芯微RK3588系列NPU芯片
3. **SDK授权**: 需要向瑞芯微申请RKNN-Toolkit2 SDK使用权限
4. **校准数据**: 使用与TensorRT/NCNN相同的校准数据集确保一致性

## 📁 文件说明

```
mobilenetv2/
├── 01_python/
│   ├── 01_check_deps.py             # 环境依赖检查
│   ├── 02_convert_model.py          # 模型转换工具
│   ├── 03_benchmark_all.py          # 多后端性能测试
│   ├── rknn_convert_inference.py    # RKNN转换推理（独立环境）
│   ├── README.md                    # 项目文档
│   └── requirements.txt             # Python依赖包列表
├── model/
│   ├── mobilenet_v2_1.0_224.tflite     # 原始TFLite模型
│   ├── mobilenet_v2_1.0_224.onnx       # ONNX模型
│   ├── mobilenet_v2_1.0_224.rknn       # RKNN模型（NPU专用）
│   ├── mobilenet_v2_1.0_224_fp32.trt   # TensorRT FP32引擎
│   ├── mobilenet_v2_1.0_224_fp16.trt   # TensorRT FP16引擎
│   ├── mobilenet_v2_1.0_224_int8.trt   # TensorRT INT8引擎
│   ├── mobilenet_v2_1.0_224.param      # NCNN网络结构文件 (FP32)
│   ├── mobilenet_v2_1.0_224.bin        # NCNN权重参数文件 (FP32)
│   ├── mobilenet_v2_1.0_224-int8.param # NCNN网络结构文件 (INT8)
│   ├── mobilenet_v2_1.0_224-int8.bin   # NCNN权重参数文件 (INT8)
│   ├── calibration.cache               # INT8校准缓存
│   ├── labels.txt                      # ImageNet分类标签
│   └── download.sh                     # 模型下载脚本
└── input/
    ├── fish_224x224.jpeg               # 测试图像-金鱼
    ├── cat_224x224.jpg                 # 测试图像-猫
    ├── dog_224x224.jpg                 # 测试图像-狗
    └── dataset.txt                     # 校准数据集列表文件
```

## 🔧 常见问题

**Q: 版本兼容性问题？**
A: 严格按照requirements.txt的版本安装，ncnn单独装，并且记住在安装ncnn时同时锁定所有依赖版本，否则会破坏其它依赖

**Q: NCNN 预处理参数设置**
A: **关键在于使用正确的预处理参数**。NCNN需要使用**MobileNetV2官方预处理参数**才能获得正确预测结果：

```python
# 方式1: numpy统一预处理 (推荐，与其他后端统一)
img = (img - 127.5) / 127.5  # [-1, 1] 范围
img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
mat_in = ncnn.Mat(np.ascontiguousarray(img))

# 方式2: NCNN原生预处理 (NCNN的ncnn.Mat(array)需要CHW格式，from_pixels接口会内部做这个转换)
mat_in = ncnn.Mat.from_pixels(img, ncnn.Mat.PixelType.PIXEL_RGB, 224, 224)
mat_in.substract_mean_normalize(
    [127.5, 127.5, 127.5],              # mean = 127.5 (255/2)
    [0.007843137, 0.007843137, 0.007843137]  # 1/127.5
)
```

**参数说明**：

- **MobileNetV2标准化**: (pixel - 127.5) / 127.5 → 将[0,255]映射到[-1,1]
- **mean = 127.5** = 255/2 (像素范围的中点)
- **1/std = 1/127.5** = 0.007843137 (缩放因子)

**错误的预处理方式**：

- ❌ 使用ImageNet标准化参数 (适用于ResNet等，不适用于MobileNet)
  - MobileNetV2模型的预处理参数与常见的ImageNet标准化不同
  - **错误做法**: 使用ImageNet标准化 `(pixel - [123.675, 116.28, 103.53]) / [58.5, 57.0, 57.375]`
  - **正确做法**: 使用MobileNetV2官方预处理 `(pixel - 127.5) / 127.5` ([-1,1]归一化)
- ❌ 仅使用 [0,1] 归一化：`img/255.0`
- ❌ 使用错误的数据格式 (CHW vs HWC 不匹配)

**Q: 权限被拒绝**
A: 确保给NCNN工具添加了执行权限: `chmod +x /path/ncnn-*/bin/*`

**Q: 能否用 pip install onnx2ncnn？**
A: 不能。`onnx2ncnn` 是 NCNN 工具包中的 C++ 编译二进制程序，不是 Python 包。只能通过以下方式获取：

- 下载 NCNN 预编译工具包（推荐）
- 从源码编译 NCNN
- 系统包管理器安装（如 apt、yum）
  pip 上的相关包都是非官方的，可能不稳定或版本过旧

**Q: NCNN量化工具链是否稳定？**
A: 非常稳定！NCNN的量化工具链很成熟：

- **ncnnoptimize**: 模型优化和图融合
- **ncnn2table**: 支持KL散度和ACIQ量化算法  
- **ncnn2int8**: 高效的INT8转换
- **ncnnoptimize** → **ncnn2table** → **ncnn2int8** 一站式量化
- 被广泛用于生产环境，特别是移动端部署

## 🎯 核心价值

**完整转换流程**：从TFLite到所有主流推理框架的端到端实现

**双重量化技术**：TensorRT GPU量化 + NCNN CPU量化完整支持

**统一校准数据集**：TensorRT和NCNN量化统一支持多图片数据集校准，提升量化精度

**公平性能对比**：统一预处理下的真实性能数据

**精度一致性突破**：解决多后端预处理参数不统一的关键问题

**工程实践经验**：解决版本兼容、环境配置、预处理标准化、量化校准等实际问题

**生产就绪方案**：适合GPU/CPU不同场景的推理优化策略

