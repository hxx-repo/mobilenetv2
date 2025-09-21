#!/bin/bash

# MobileNetV2 C++ 可执行文件构建脚本

set -e  # 遇到错误立即退出

echo "========================================"
echo "MobileNetV2 C++ 可执行文件构建"
echo "========================================"

# 检查动态库是否存在
if [ ! -f "install/lib/libmobilenet_inference.so" ]; then
    echo "错误: 未找到动态库 install/lib/libmobilenet_inference.so"
    echo "请先运行 ./build_so.sh 构建动态库"
    exit 1
fi

echo "✅ 找到动态库: install/lib/libmobilenet_inference.so"

cd examples

# 创建build目录
if [ -d "build" ]; then
    echo "清理旧的examples/build目录..."
    rm -rf build
fi

mkdir build
cd build

echo "配置CMake (可执行文件)..."
/home/xinxin/work/depend_config/cmake/cmake-3.28.1-linux-x86_64/bin/cmake .. -DCMAKE_BUILD_TYPE=Release

echo "编译可执行文件..."
make -j$(nproc)

echo "安装可执行文件..."
make install

echo "========================================"
echo "可执行文件构建完成！"
echo "========================================"

echo "安装的文件:"
echo "✅ 可执行文件: install/bin/cpp_inference_test"