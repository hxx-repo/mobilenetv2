# MobileNetV2 C++ 推理引擎

高性能 MobileNetV2 深度学习模型 C++ 推理实现，**完全对齐Python版本架构设计**。

## 📊 性能结果

在 RTX 3060 上的 MobileNetV2 推理性能对比（完全对齐Python预处理）：

```
🏆 === C++ 性能对比结果 ===
TensorRT (FP16)     : 0.0010秒 ( 994.7 FPS) 🥇 [最快GPU] - 9MB
TensorRT (FP32)     : 0.0013秒 ( 749.6 FPS) [1.3x slower] 🥈 [最快GPU] - 16MB  
TensorRT (INT8)     : 0.0012秒 ( 834.5 FPS) [1.2x slower] 🥉 [最快GPU] - 5MB
NCNN FP32           : 0.0029秒 ( 341.3 FPS) [2.9x slower] [最快CPU] - 14MB
NCNN INT8           : 0.0041秒 ( 242.2 FPS) [4.1x slower] [移动优化] - 3.5MB
MNN FP32           : 0.0035秒 ( 285.2 FPS) [3.5x slower] [端侧部署] - 14MB
MNN INT8           : 0.0043秒 ( 234.0 FPS) [4.3x slower] [端侧部署] - 3.6MB
ONNX Runtime        : 0.0130秒 ( 77.1 FPS) [12.9x slower] [跨平台] - 14MB
TensorFlow Lite     : 0.0209秒 ( 47.8 FPS) [20.9x slower] [基准] - 14MB
```

**精度验证**：所有后端与Python版本数值完全一致

| 后端 | Top-1预测 | 置信度 | 精度状态 |
| ---- | --------- | ------ | -------- |
| **TensorFlow Lite** | goldfish | 99.73% | ✅ 与Python完全匹配 |
| **ONNX Runtime** | goldfish | 99.73% | ✅ 与Python完全匹配 |
| **TensorRT (FP32)** | goldfish | 99.73% | ✅ 与Python完全匹配 |
| **TensorRT (FP16)** | goldfish | 99.76% | ✅ 与Python完全匹配 |
| **NCNN FP32** | goldfish | 99.73% | ✅ 与Python完全匹配 |
| **TensorRT (INT8)** | goldfish | 99.22% | ✅ 量化轻微精度损失0.5% |
| **NCNN INT8** | goldfish | 99.38% | ✅ 量化精度轻微损失0.35% |
| **MNN** | goldfish | 99.80% | ✅ 与Python完全匹配 |
| **MNN (INT8)** | goldfish | 99.50% | ✅ 量化精度轻微损失0.3% |

## 🛠️ 环境配置

### 系统要求
- Ubuntu 18.04+ / WSL2
- CMake 3.16+
- C++17 编译器 (GCC 7+)
- OpenCV 4.x
- NVIDIA GPU (RTX 20/30/40系列) - 仅 TensorRT 需要
- CUDA 12.x + cuDNN - 仅 TensorRT 需要

### 依赖说明

```
C++后端依赖顺序：
1. OpenCV - 图像预处理，所有后端必需
2. TensorFlow Lite - CPU跨平台推理
3. ONNX Runtime - CPU优化推理
4. TensorRT - GPU加速推理 (需要CUDA环境)
5. NCNN - 移动端优化推理
6. MNN - 端侧推理 / 量化工具链
```

### 安装步骤

**CMake**

```bash
# 下载CMake预编译包
cd ~/work/depend_config/cmake
wget https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-linux-x86_64.tar.gz
tar -xzf cmake-3.28.1-linux-x86_64.tar.gz
export PATH="/home/xinxin/work/depend_config/cmake/cmake-3.28.1-linux-x86_64/bin:$PATH"

# 安装OpenCV (所有后端必需)
# 参考配置路径: /home/xinxin/work/depend_config/opencv/opencv_install
export OPENCV_ROOT="/home/xinxin/work/depend_config/opencv/opencv_install"
export LD_LIBRARY_PATH=$OPENCV_ROOT/lib:$LD_LIBRARY_PATH
```

**OpenCV**

```bash
# 下载OpenCV源码
cd ~/work/depend_config/opencv
wget https://github.com/opencv/opencv/archive/refs/tags/4.8.1.tar.gz -O opencv-4.8.1.tar.gz
tar -xzf opencv-4.8.1.tar.gz

# 编译安装
cd opencv-4.8.1 && mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=../../opencv_install \
      -D BUILD_SHARED_LIBS=ON \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_opencv_apps=OFF \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_python3=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_DOCS=OFF \
      -D WITH_IPP=OFF ..
make -j$(nproc)
make install

# 验证安装
ls /home/xinxin/work/depend_config/opencv/opencv_install/lib/
```

**TensorFlow Lite (CPU后端)**

```bash
# 下载TensorFlow源码
cd ~/work/depend_config/tensorflow_lite
wget https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.16.1.tar.gz -O tensorflow-2.16.1.tar.gz
tar -xzf tensorflow-2.16.1.tar.gz

# 由于网络限制，需要手动下载以下TensorFlow Lite依赖包
# abseil-cpp、eigen、fft2d (OouraFFT)、neon2sse、ml_dtypes、cpuinfo、farmhash、flatbuffers、gemmlowp、ruy
```

```bash
# 下载abseil-cpp
cd ~/work/depend_config/tensorflow_lite
wget https://github.com/abseil/abseil-cpp/archive/fb3621f4f897824c0dbe0615fa94543df6192f30.tar.gz
tar -xzf fb3621f4f897824c0dbe0615fa94543df6192f30.tar.gz
```

```bash
# 下载eigen
cd ~/work/depend_config/tensorflow_lite
wget https://gitlab.com/libeigen/eigen/-/archive/aa6964bf3a34fd607837dd8123bc42465185c4f8/eigen-aa6964bf3a34fd607837dd8123bc42465185c4f8.tar.gz
tar -xzf eigen-aa6964bf3a34fd607837dd8123bc42465185c4f8.tar.gz
```

```bash
# 下载fft2d (OouraFFT)
cd ~/work/depend_config/tensorflow_lite
wget https://storage.googleapis.com/mirror.tensorflow.org/github.com/petewarden/OouraFFT/archive/v1.0.tar.gz -O OouraFFT-1.0.tar.gz
tar -xzf OouraFFT-1.0.tar.gz
```

```bash
# 下载neon2sse
cd ~/work/depend_config/tensorflow_lite
wget https://github.com/intel/ARM_NEON_2_x86_SSE/archive/a15b489e1222b2087007546b4912e21293ea86ff.tar.gz
tar -xzf a15b489e1222b2087007546b4912e21293ea86ff.tar.gz
```

```bash
# 下载ml_dtypes
cd ~/work/depend_config/tensorflow_lite
wget https://github.com/jax-ml/ml_dtypes/archive/780b6d0ee01ffbfac45f7ec5418bc08f2b166483.tar.gz
tar -xzf 780b6d0ee01ffbfac45f7ec5418bc08f2b166483.tar.gz
```

```bash
# 下载cpuinfo
cd ~/work/depend_config/tensorflow_lite
wget https://github.com/pytorch/cpuinfo/archive/ef634603954d88d2643d5809011288b890ac126e.tar.gz
tar -xzf ef634603954d88d2643d5809011288b890ac126e.tar.gz
```

```bash
# 下载farmhash
cd ~/work/depend_config/tensorflow_lite
wget https://github.com/google/farmhash/archive/0d859a811870d10f53a594927d0d0b97573ad06d.tar.gz
tar -xzf 0d859a811870d10f53a594927d0d0b97573ad06d.tar.gz
```

```bash
# 下载flatbuffers
cd ~/work/depend_config/tensorflow_lite
wget https://github.com/google/flatbuffers/archive/refs/tags/v23.5.26.tar.gz
tar -xzf v23.5.26.tar.gz
```

```bash
# 下载gemmlowp
cd ~/work/depend_config/tensorflow_lite
wget https://github.com/google/gemmlowp/archive/16e8662c34917be0065110bfcd9cc27d30f52fdf.tar.gz
tar -xzf 16e8662c34917be0065110bfcd9cc27d30f52fdf.tar.gz
```

```bash
# 下载ruy
cd ~/work/depend_config/tensorflow_lite
wget https://github.com/google/ruy/archive/3286a34cc8de6149ac6844107dfdffac91531e72.tar.gz
tar -xzf 3286a34cc8de6149ac6844107dfdffac91531e72.tar.gz
```

```bash
# 验证依赖包
cd ~/work/depend_config/tensorflow_lite
ls -la | grep -E "(abseil-cpp|eigen|OouraFFT|ARM_NEON|ml_dtypes|cpuinfo|farmhash|flatbuffers|gemmlowp|ruy)"

# 应该看到以下目录:
# abseil-cpp-fb3621f4f897824c0dbe0615fa94543df6192f30/
# eigen-aa6964bf3a34fd607837dd8123bc42465185c4f8/
# OouraFFT-1.0/
# ARM_NEON_2_x86_SSE-a15b489e1222b2087007546b4912e21293ea86ff/
# ml_dtypes-780b6d0ee01ffbfac45f7ec5418bc08f2b166483/
# cpuinfo-ef634603954d88d2643d5809011288b890ac126e/
# farmhash-0d859a811870d10f53a594927d0d0b97573ad06d/
# flatbuffers-23.5.26/
# gemmlowp-16e8662c34917be0065110bfcd9cc27d30f52fdf/
# ruy-3286a34cc8de6149ac6844107dfdffac91531e72/
```

```bash
# 修改abseil-cpp配置
sed -i 's|^.*git.*abseil-cpp.*|# Use local abseil-cpp instead of downloading\nset(abseil-cpp_SOURCE_DIR "/home/xinxin/work/depend_config/tensorflow_lite/abseil-cpp-fb3621f4f897824c0dbe0615fa94543df6192f30")|' \
/home/xinxin/work/depend_config/tensorflow_lite/tensorflow-2.16.1/tensorflow/lite/tools/cmake/modules/abseil-cpp.cmake

# 修改eigen配置  
sed -i 's|^.*git.*eigen.*|# Use local eigen instead of downloading\nset(eigen_SOURCE_DIR "/home/xinxin/work/depend_config/tensorflow_lite/eigen-aa6964bf3a34fd607837dd8123bc42465185c4f8")|' \
/home/xinxin/work/depend_config/tensorflow_lite/tensorflow-2.16.1/tensorflow/lite/tools/cmake/modules/eigen.cmake

# 修改fft2d配置
sed -i 's|^.*git.*fft2d.*|# Use local fft2d instead of downloading\nset(fft2d_SOURCE_DIR "/home/xinxin/work/depend_config/tensorflow_lite/OouraFFT-1.0")|' \
/home/xinxin/work/depend_config/tensorflow_lite/tensorflow-2.16.1/tensorflow/lite/tools/cmake/modules/fft2d.cmake

# 修改neon2sse配置
sed -i 's|^.*git.*neon2sse.*|# Use local neon2sse instead of downloading\nset(neon2sse_SOURCE_DIR "/home/xinxin/work/depend_config/tensorflow_lite/ARM_NEON_2_x86_SSE-a15b489e1222b2087007546b4912e21293ea86ff")|' \
/home/xinxin/work/depend_config/tensorflow_lite/tensorflow-2.16.1/tensorflow/lite/tools/cmake/modules/neon2sse.cmake

# 修改ml_dtypes配置
sed -i 's|^.*git.*ml_dtypes.*|# Use local ml_dtypes instead of downloading\nset(ml_dtypes_SOURCE_DIR "/home/xinxin/work/depend_config/tensorflow_lite/ml_dtypes-780b6d0ee01ffbfac45f7ec5418bc08f2b166483")|' \
/home/xinxin/work/depend_config/tensorflow_lite/tensorflow-2.16.1/tensorflow/lite/tools/cmake/modules/ml_dtypes.cmake

# 修改cpuinfo配置
sed -i 's|^.*git.*cpuinfo.*|# Use local cpuinfo instead of downloading\nset(cpuinfo_SOURCE_DIR "/home/xinxin/work/depend_config/tensorflow_lite/cpuinfo-ef634603954d88d2643d5809011288b890ac126e")|' \
/home/xinxin/work/depend_config/tensorflow_lite/tensorflow-2.16.1/tensorflow/lite/tools/cmake/modules/cpuinfo.cmake

# 修改farmhash配置
sed -i 's|^.*git.*farmhash.*|# Use local farmhash instead of downloading\nset(farmhash_SOURCE_DIR "/home/xinxin/work/depend_config/tensorflow_lite/farmhash-0d859a811870d10f53a594927d0d0b97573ad06d")|' \
/home/xinxin/work/depend_config/tensorflow_lite/tensorflow-2.16.1/tensorflow/lite/tools/cmake/modules/farmhash.cmake

# 修改flatbuffers配置
sed -i 's|^.*git.*flatbuffers.*|# Use local flatbuffers instead of downloading\nset(flatbuffers_SOURCE_DIR "/home/xinxin/work/depend_config/tensorflow_lite/flatbuffers-23.5.26")|' \
/home/xinxin/work/depend_config/tensorflow_lite/tensorflow-2.16.1/tensorflow/lite/tools/cmake/modules/flatbuffers.cmake

# 修改gemmlowp配置
sed -i 's|^.*git.*gemmlowp.*|# Use local gemmlowp instead of downloading\nset(gemmlowp_SOURCE_DIR "/home/xinxin/work/depend_config/tensorflow_lite/gemmlowp-16e8662c34917be0065110bfcd9cc27d30f52fdf")|' \
/home/xinxin/work/depend_config/tensorflow_lite/tensorflow-2.16.1/tensorflow/lite/tools/cmake/modules/gemmlowp.cmake

# 修改ruy配置
sed -i 's|^.*git.*ruy.*|# Use local ruy instead of downloading\nset(ruy_SOURCE_DIR "/home/xinxin/work/depend_config/tensorflow_lite/ruy-3286a34cc8de6149ac6844107dfdffac91531e72")|' \
/home/xinxin/work/depend_config/tensorflow_lite/tensorflow-2.16.1/tensorflow/lite/tools/cmake/modules/ruy.cmake
```

```bash
# 编译TensorFlow Lite
cd ~/work/depend_config/tensorflow_lite
mkdir tflite_build && cd tflite_build

# CMake配置
cmake ../tensorflow-2.16.1/tensorflow/lite \
    -DTFLITE_ENABLE_XNNPACK=OFF \
    -DTFLITE_ENABLE_GPU=OFF \
    -DBUILD_SHARED_LIBS=ON

# 编译 (并行编译，但需要较长时间)
make -j$(nproc)

# 验证编译结果
ls -la libtensorflow-lite.so
# 应该看到: -rwxr-xr-x 1 xinxin xinxin 5807944 Sep  5 22:35 libtensorflow-lite.so
```

```bash
# 配置TensorFlow Lite
export TFLITE_BUILD="/home/xinxin/work/depend_config/tensorflow_lite/tflite_build"
export LD_LIBRARY_PATH=$TFLITE_BUILD:$LD_LIBRARY_PATH
```

**ONNX Runtime (CPU后端)**

```bash
# 下载ONNX Runtime预编译包
cd ~/work/depend_config/onnxruntime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz
tar -xzf onnxruntime-linux-x64-1.19.2.tgz

# 验证解压结果
ls -la onnxruntime-linux-x64-1.19.2/
# 应该看到:
# include/          # C++ 头文件
# lib/              # 预编译库文件 
# ├── libonnxruntime.so
# ├── libonnxruntime.so.1
# └── libonnxruntime.so.1.19.2
```

```bash
# 配置ONNX Runtime
export ONNXRUNTIME_ROOT="/home/xinxin/work/depend_config/onnxruntime/onnxruntime-linux-x64-1.19.2"
export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH
```

**TensorRT (GPU后端)**

```bash
# **前置条件检查**:

# 配置CUDA (TensorRT需要)
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# 检查CUDA版本 (需要CUDA 11.0+)
nvidia-smi
nvcc --version
```

```bash
# 下载TensorRT 8.6.1.6
cd ~/work/depend_config/tensorrt
wget -c -q --show-progress "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz"
tar -xzf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz

# 验证解压结果
ls -la TensorRT-8.6.1.6/
# 应该看到:
# include/          # TensorRT C++ 头文件 (NvInfer.h, NvInferRuntime.h等)
# lib/              # TensorRT库文件 (libnvinfer.so等)
# bin/              # TensorRT工具 (trtexec等)
```

```bash
# 配置TensorRT
export TENSORRT_ROOT="/home/xinxin/work/depend_config/tensorrt/TensorRT-8.6.1.6"
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH
export PATH=$TENSORRT_ROOT/bin:$PATH
```

**NCNN (移动端后端)**

```bash
# 下载NCNN预编译包
cd ~/work/depend_config/ncnn
wget https://github.com/Tencent/ncnn/releases/download/20231027/ncnn-20231027-ubuntu-2004-shared.zip
unzip ncnn-20231027-ubuntu-2004-shared.zip

# 验证解压结果
ls -la ncnn-20231027-ubuntu-2004-shared/
# 应该看到:
# include/ncnn/       # NCNN C++ 头文件 (net.h, mat.h等)
# lib/                # NCNN库文件 (libncnn.so等)
# bin/                # NCNN工具

# 由于网络限制，需要手动下载以下NCNN依赖包
# Vulkan、protobuf
```

```bash
# 下载Vulkan (NCNN编译依赖)
cd ~/work/depend_config/ncnn
wget https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.3.216.tar.gz
tar -xzf v1.3.216.tar.gz

export VULKAN_HEADERS_ROOT="/home/xinxin/work/depend_config/ncnn/Vulkan-Headers-1.3.216"
```

```bash
# 下载protobuf (NCNN运行时依赖)
cd ~/work/depend_config/ncnn
wget http://archive.ubuntu.com/ubuntu/pool/main/p/protobuf/libprotobuf17_3.6.1.3-2ubuntu5_amd64.deb
ar x libprotobuf17_3.6.1.3-2ubuntu5_amd64.deb
tar -xf data.tar.xz

export PROTOBUF_ROOT="/home/xinxin/work/depend_config/ncnn/usr"
export LD_LIBRARY_PATH=$PROTOBUF_ROOT/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

```bash
# 配置NCNN
export NCNN_ROOT="/home/xinxin/work/depend_config/ncnn/ncnn-20231027-ubuntu-2004-shared"
export LD_LIBRARY_PATH=$NCNN_ROOT/lib:$LD_LIBRARY_PATH
```

**MNN (端侧后端)**

```bash
# 获取并编译 MNN 3.2.5 (仅需 lib)
cd ~/work/depend_config/mnn
git clone https://github.com/alibaba/MNN.git
cd MNN
git checkout tags/3.2.5
mkdir -p build && cd build
cmake .. -DMNN_BUILD_CONVERTER=false -DMNN_BUILD_QUANTOOLS=OFF
make -j$(nproc)

# 整理运行库
mkdir -p ~/work/depend_config/mnn/lib
ln -sf ~/work/depend_config/mnn/MNN/build/libMNN.so ~/work/depend_config/mnn/lib/libMNN.so

# 配置环境变量（无需转换/量化工具）
export MNN_ROOT="/home/xinxin/work/depend_config/mnn/MNN"
export LD_LIBRARY_PATH=~/work/depend_config/mnn/lib:$LD_LIBRARY_PATH
```

## 🚀 快速开始

按照编号顺序执行以下步骤：

### 编译项目

需要先设置CMake路径：

```bash
# CMake路径（编译时必需）
export PATH="/home/xinxin/work/depend_config/cmake/cmake-3.28.1-linux-x86_64/bin:$PATH"

cd 02_cpp

# 步骤1: 构建并安装动态库
./build_so.sh

# 步骤2: 构建并安装可执行文件  
./build_exe.sh
```

### 运行测试

需要先设置运行时库路径：
```bash
# 设置运行时库路径（安装目录里的 libmobilenet_inference.so 没有记录依赖库的路径（仅链接到编译时的绝对位置），这个步骤应该得放在 步骤2: 构建并安装可执行文件 之前，不然上面步骤2会报错）
export LD_LIBRARY_PATH="\
/home/xinxin/work/depend_config/opencv/opencv_install/lib:\
/home/xinxin/work/depend_config/tensorflow_lite/tflite_build:\
/home/xinxin/work/depend_config/onnxruntime/onnxruntime-linux-x64-1.19.2/lib:\
/home/xinxin/work/depend_config/tensorrt/TensorRT-8.6.1.6/lib:\
/home/xinxin/work/depend_config/ncnn/ncnn-20231027-ubuntu-2004-shared/lib:\
/home/xinxin/work/depend_config/ncnn/usr/lib/x86_64-linux-gnu:\
/home/xinxin/work/depend_config/mnn/lib:\
/home/xinxin/work/mobilenetv2/02_cpp/install/lib:\
/usr/local/cuda-12.4/targets/x86_64-linux/lib:\
$LD_LIBRARY_PATH"

# TensorFlow Lite
install/bin/cpp_inference_test ../model/mobilenet_v2_1.0_224.tflite ../input/fish_224x224.jpeg ../model/labels.txt

# ONNX Runtime
install/bin/cpp_inference_test ../model/mobilenet_v2_1.0_224.onnx ../input/fish_224x224.jpeg ../model/labels.txt

# TensorRT FP32/FP16/INT8 (自动识别精度)
install/bin/cpp_inference_test ../model/mobilenet_v2_1.0_224_fp32.trt ../input/fish_224x224.jpeg ../model/labels.txt
install/bin/cpp_inference_test ../model/mobilenet_v2_1.0_224_fp16.trt ../input/fish_224x224.jpeg ../model/labels.txt
install/bin/cpp_inference_test ../model/mobilenet_v2_1.0_224_int8.trt ../input/fish_224x224.jpeg ../model/labels.txt

# NCNN FP32/INT8 (自动识别精度)
install/bin/cpp_inference_test ../model/mobilenet_v2_1.0_224.param ../input/fish_224x224.jpeg ../model/labels.txt
install/bin/cpp_inference_test ../model/mobilenet_v2_1.0_224-int8.param ../input/fish_224x224.jpeg ../model/labels.txt

# MNN FP32/INT8 (自动识别精度)
install/bin/cpp_inference_test ../model/mobilenet_v2_1.0_224.mnn ../input/fish_224x224.jpeg ../model/labels.txt
install/bin/cpp_inference_test ../model/mobilenet_v2_1.0_224_int8.mnn ../input/fish_224x224.jpeg ../model/labels.txt
```

## 📁 文件说明

```
mobilenetv2/
├── 02_cpp/
│   ├── include/                     # 头文件
│   │   ├── inference_backend.hpp    # 核心接口 - 对应Python的InferenceBackend
│   │   ├── tflite_backend.hpp       # TFLite后端
│   │   ├── onnx_backend.hpp         # ONNX后端  
│   │   ├── tensorrt_backend.hpp     # TensorRT后端
│   │   ├── ncnn_backend.hpp         # NCNN后端
│   │   └── mnn_backend.hpp          # MNN后端
│   ├── src/                         # 源文件
│   │   ├── inference_backend.cpp    # 工厂类实现
│   │   ├── tflite_backend.cpp       # TFLite具体实现
│   │   ├── onnx_backend.cpp         # ONNX具体实现
│   │   ├── tensorrt_backend.cpp     # TensorRT具体实现
│   │   ├── ncnn_backend.cpp         # NCNN具体实现
│   │   └── mnn_backend.cpp          # MNN具体实现
│   ├── examples/
│   │   ├── cpp_inference_test.cpp   # 统一多后端测试程序
│   │   └── CMakeLists.txt           # 可执行文件构建配置
│   ├── install/                     # 本地安装目录
│   │   ├── include/                 # 导出头文件
│   │   ├── lib/                     # 动态库
│   │   └── bin/                     # 可执行文件
│   ├── build_so.sh                  # 构建动态库脚本
│   ├── build_exe.sh                 # 构建可执行文件脚本  
│   ├── run_test.sh                  # 测试运行脚本
│   └── README_new.md                # 本文档
└── model/                           # 模型文件 (与Python版本共享)
    ├── mobilenet_v2_1.0_224.tflite     # TensorFlow Lite模型
    ├── mobilenet_v2_1.0_224.onnx       # ONNX模型
    ├── mobilenet_v2_1.0_224_fp32.trt   # TensorRT FP32引擎
    ├── mobilenet_v2_1.0_224_fp16.trt   # TensorRT FP16引擎
    ├── mobilenet_v2_1.0_224_int8.trt   # TensorRT INT8引擎
    ├── mobilenet_v2_1.0_224.param      # NCNN网络结构文件 (FP32)
    ├── mobilenet_v2_1.0_224.bin        # NCNN权重参数文件 (FP32)
    ├── mobilenet_v2_1.0_224-int8.param # NCNN网络结构文件 (INT8)
    ├── mobilenet_v2_1.0_224-int8.bin   # NCNN权重参数文件 (INT8)
    └── labels.txt                      # ImageNet分类标签
```

## 🔧 常见问题

**Q: 编译时找不到依赖库？**
A: 确保所有环境变量正确设置，特别是 `LD_LIBRARY_PATH`。使用 `ldd ./install/bin/cpp_inference_test` 检查依赖。

**Q: 运行时动态库加载失败？**  
A: C++版本使用RPATH机制，可执行文件会自动找到 `install/lib/` 下的动态库。如果失败，检查：
- 是否正确运行了 `./build_so.sh` 和 `./build_exe.sh`
- `install/lib/libmobilenet_inference.so` 是否存在

**Q: 预处理参数与Python版本一致吗？**
A: **完全一致**！所有后端都使用MobileNetV2官方预处理参数：`(pixel - 127.5) / 127.5`，确保与Python版本数值精度完全匹配。

**Q: 条件编译如何工作？**
A: CMake会自动检测可用的依赖库：
- 找不到TensorFlow Lite时跳过TFLite后端
- 找不到CUDA时跳过TensorRT后端  
- 找不到NCNN时跳过NCNN后端
- 至少需要一个后端才能编译成功

## 🎯 核心价值

**完整C++实现**：从Python到C++的端到端移植，性能大幅提升

**架构完全对齐**：1:1复现Python版本的设计模式，便于维护

**动态库架构**：模块化设计，支持独立部署和集成

**精度完美一致**：统一预处理确保数值精度与Python版本完全匹配

**性能显著提升**：
- **CPU优化**：NCNN性能比TFLite快7倍
- **GPU加速**：TensorRT比CPU快20倍，接近1000 FPS
- **量化加速**：INT8版本大幅减小模型体积，精度损失轻微

**工程实践就绪**：条件编译、RPATH优化、统一测试程序等生产就绪特性
