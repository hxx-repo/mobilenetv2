cmake_minimum_required(VERSION 3.16)
project(MobileNetV2_Examples)

# 设置示例程序的编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含主项目的头文件和库
include_directories(../include)
link_directories(../install/lib)

# 直接指定依赖库路径 - OpenCV (与主项目保持一致)
set(OPENCV_ROOT "/home/xinxin/work/depend_config/opencv/opencv_install")
include_directories(${OPENCV_ROOT}/include/opencv4)
link_directories(${OPENCV_ROOT}/lib)

# OpenCV库列表
set(OpenCV_LIBS 
    opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui
)

# 创建示例程序
add_executable(cpp_inference_test cpp_inference_test.cpp)

# 链接库
target_link_libraries(cpp_inference_test
    mobilenet_inference  # 主项目库
    ${OpenCV_LIBS}       # OpenCV库
)

# 设置安装路径到../install目录
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../install")

# 安装可执行文件到bin目录
install(TARGETS cpp_inference_test
    RUNTIME DESTINATION bin
)