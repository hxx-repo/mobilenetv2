#!/bin/bash

# MobileNetV2 C++ 动态库构建脚本

set -e  # 遇到错误立即退出

echo "========================================"
echo "MobileNetV2 C++ 动态库构建"
echo "========================================"

# 创建build目录
if [ -d "build" ]; then
    echo "清理旧的build目录..."
    rm -rf build
fi

mkdir build
cd build

echo "配置CMake (动态库)..."
/home/xinxin/work/depend_config/cmake/cmake-3.28.1-linux-x86_64/bin/cmake .. -DCMAKE_BUILD_TYPE=Release

echo "编译动态库..."
make -j$(nproc)

echo "安装动态库到install目录..."
make install

echo "========================================"
echo "动态库构建完成！"
echo "========================================"

echo "安装的文件:"
echo "动态库:"
ls -la ../install/lib/
echo "头文件:"
ls -la ../install/include/