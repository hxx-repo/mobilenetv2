cmake_minimum_required(VERSION 3.16)
project(MobileNetV2_CPP_Inference)

# C++17标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

# 包含目录
include_directories(include)

# 直接指定依赖库路径 - OpenCV
set(OPENCV_ROOT "/home/xinxin/work/depend_config/opencv/opencv_install")
include_directories(${OPENCV_ROOT}/include/opencv4)
link_directories(${OPENCV_ROOT}/lib)
message(STATUS "使用OpenCV: ${OPENCV_ROOT}")

# OpenCV库列表
set(OpenCV_LIBS 
    opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui
)

# 直接指定依赖库路径 - TensorFlow Lite
set(TFLITE_ROOT "/home/xinxin/work/depend_config/tensorflow_lite/tensorflow-2.16.1")
set(TFLITE_BUILD "/home/xinxin/work/depend_config/tensorflow_lite/tflite_build")
set(TFLITE_DEPS "/home/xinxin/work/depend_config/tensorflow_lite")

if(EXISTS ${TFLITE_BUILD})
    # TensorFlow Lite核心包含路径
    include_directories(${TFLITE_ROOT})
    
    # TensorFlow Lite依赖库包含路径
    include_directories(${TFLITE_DEPS}/flatbuffers-23.5.26/include)
    include_directories(${TFLITE_DEPS}/abseil-cpp-fb3621f4f897824c0dbe0615fa94543df6192f30)
    include_directories(${TFLITE_DEPS}/eigen-aa6964bf3a34fd607837dd8123bc42465185c4f8)
    
    link_directories(${TFLITE_BUILD})
    set(TFLITE_LIB tensorflow-lite)
    message(STATUS "使用TensorFlow Lite: ${TFLITE_BUILD}/lib${TFLITE_LIB}.so")
    set(TFLITE_FOUND TRUE)
else()
    message(WARNING "未找到TensorFlow Lite，将跳过TensorFlow Lite后端")
    set(TFLITE_FOUND FALSE)
endif()

# 直接指定依赖库路径 - ONNX Runtime
set(ONNXRUNTIME_ROOT "/home/xinxin/work/depend_config/onnxruntime/onnxruntime-linux-x64-1.19.2")
if(EXISTS ${ONNXRUNTIME_ROOT})
    include_directories(${ONNXRUNTIME_ROOT}/include)
    link_directories(${ONNXRUNTIME_ROOT}/lib)
    set(ONNXRUNTIME_LIB onnxruntime)
    message(STATUS "使用ONNX Runtime: ${ONNXRUNTIME_ROOT}")
    set(ONNXRUNTIME_FOUND TRUE)
else()
    message(WARNING "未找到ONNX Runtime，将跳过ONNX后端")
    set(ONNXRUNTIME_FOUND FALSE)
endif()

# 直接指定依赖库路径 - TensorRT
set(TENSORRT_ROOT "/home/xinxin/work/depend_config/tensorrt/TensorRT-8.6.1.6")
if(EXISTS ${TENSORRT_ROOT})
    include_directories(${TENSORRT_ROOT}/include)
    link_directories(${TENSORRT_ROOT}/lib)
    set(TENSORRT_LIB nvinfer)
    message(STATUS "使用TensorRT: ${TENSORRT_ROOT}")
    set(TENSORRT_FOUND TRUE)
else()
    message(WARNING "未找到TensorRT，将跳过TensorRT后端")
    set(TENSORRT_FOUND FALSE)
endif()

# 直接指定依赖库路径 - NCNN
set(NCNN_ROOT "/home/xinxin/work/depend_config/ncnn/ncnn-20231027-ubuntu-2004-shared")
set(VULKAN_HEADERS_ROOT "/home/xinxin/work/depend_config/ncnn/Vulkan-Headers-1.3.216")
if(EXISTS ${NCNN_ROOT})
    include_directories(${NCNN_ROOT}/include)
    include_directories(${VULKAN_HEADERS_ROOT}/include)  # 添加Vulkan头文件路径
    link_directories(${NCNN_ROOT}/lib) 
    set(NCNN_LIB ncnn)
    message(STATUS "使用NCNN: ${NCNN_ROOT}")
    message(STATUS "使用Vulkan头文件: ${VULKAN_HEADERS_ROOT}/include")
    set(NCNN_FOUND TRUE)
else()
    message(WARNING "未找到NCNN，将跳过NCNN后端")
    set(NCNN_FOUND FALSE)
endif()

# 核心库源文件 - 对齐新的架构
set(CORE_SOURCES
    src/inference_backend.cpp
)

# 根据可用的后端添加源文件
set(BACKEND_SOURCES)
set(BACKEND_LIBS)

if(TFLITE_FOUND)
    list(APPEND BACKEND_SOURCES src/tflite_backend.cpp)
    list(APPEND BACKEND_LIBS ${TFLITE_LIB})
    add_definitions(-DENABLE_TFLITE)
endif()

if(ONNXRUNTIME_FOUND)
    list(APPEND BACKEND_SOURCES src/onnx_backend.cpp)
    list(APPEND BACKEND_LIBS ${ONNXRUNTIME_LIB})
    add_definitions(-DENABLE_ONNXRUNTIME)
endif()

if(TENSORRT_FOUND)
    # 添加CUDA路径
    include_directories(/usr/local/cuda-12.4/targets/x86_64-linux/include)
    link_directories(/usr/local/cuda-12.4/targets/x86_64-linux/lib)
    
    list(APPEND BACKEND_SOURCES src/tensorrt_backend.cpp)
    list(APPEND BACKEND_LIBS ${TENSORRT_LIB} cudart)  # 添加CUDA运行时库
    add_definitions(-DENABLE_TENSORRT)
endif()

if(NCNN_FOUND)
    list(APPEND BACKEND_SOURCES src/ncnn_backend.cpp)
    list(APPEND BACKEND_LIBS ${NCNN_LIB})
    add_definitions(-DENABLE_NCNN)
endif()

# 创建动态库
add_library(mobilenet_inference SHARED
    ${CORE_SOURCES}
    ${BACKEND_SOURCES}
)

# 链接库
target_link_libraries(mobilenet_inference
    ${OpenCV_LIBS}
    ${BACKEND_LIBS}
)

# 设置安装路径到本地install目录
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install")

# 安装规则
install(TARGETS mobilenet_inference
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# 只安装核心接口头文件
install(FILES include/inference_backend.hpp
    DESTINATION include
)