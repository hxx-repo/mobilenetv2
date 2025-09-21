#include "ncnn_backend.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <stdexcept>

namespace mobilenet_inference {

NCNNBackend::NCNNBackend(const std::string& model_path) 
    : InferenceBackend(model_path) {
    // 完全对齐Python版本 - 构造函数只调用基类，不需要其他操作
}

bool NCNNBackend::load_model() {
    try {
        // 获取param和bin文件路径 - 从基类model_path_推导
        std::string param_path = model_path_;
        std::string bin_path = model_path_.substr(0, model_path_.find(".param")) + ".bin";
        
        // 检查模型文件是否存在
        if (!std::ifstream(param_path).good()) {
            std::cerr << "无法找到NCNN param文件: " << param_path << std::endl;
            return false;
        }
        
        if (!std::ifstream(bin_path).good()) {
            std::cerr << "无法找到NCNN bin文件: " << bin_path << std::endl;
            return false;
        }
        
        // 配置NCNN - 对应Python的优化设置
        net_.opt.use_vulkan_compute = false;  // 关闭Vulkan GPU加速，使用CPU
        net_.opt.num_threads = 4;             // 设置线程数
        
        // 加载NCNN模型 - 对应Python的net.load_param()和net.load_model()
        std::cout << "加载NCNN模型: " << param_path << std::endl;
        int ret_param = net_.load_param(param_path.c_str());
        int ret_model = net_.load_model(bin_path.c_str());
        
        if (ret_param != 0) {
            std::cerr << "❌ NCNN param文件加载失败: " << param_path << std::endl;
            return false;
        }
        
        if (ret_model != 0) {
            std::cerr << "❌ NCNN bin文件加载失败: " << bin_path << std::endl;
            return false;
        }
        
        // ✅ 通过API获取：输入输出名称 - 完全对齐Python版本
        auto input_names = net_.input_names();
        auto output_names = net_.output_names();
        
        if (input_names.empty() || output_names.empty()) {
            std::cerr << "❌ 获取NCNN输入输出名称失败" << std::endl;
            return false;
        }
        
        // 设置基类成员变量 - 对齐Python版本
        input_name_ = std::string(input_names[0]);     // 第一个输入
        output_name_ = std::string(output_names[0]);   // 第一个输出
        input_shape_ = {1, 3, 224, 224};   // NCHW格式，对应Python的(1, 3, 224, 224)
        output_shape_ = {1, 1001};         // 1001个类别输出（包含背景类）
        input_dtype_ = "float32";
        output_dtype_ = "float32";
        
        // 完全对齐Python的打印格式
        std::cout << "✅ NCNN模型加载成功" << std::endl;
        std::cout << "   输入名称: " << input_name_ << std::endl;
        std::cout << "   输入形状: [";
        for (size_t i = 0; i < input_shape_.size(); ++i) {
            std::cout << input_shape_[i];
            if (i < input_shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "   输入数据类型: " << input_dtype_ << std::endl;
        std::cout << "   输出名称: " << output_name_ << std::endl;
        std::cout << "   输出形状: [";
        for (size_t i = 0; i < output_shape_.size(); ++i) {
            std::cout << output_shape_[i];
            if (i < output_shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "   输出数据类型: " << output_dtype_ << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "NCNN模型加载失败: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> NCNNBackend::preprocess(const std::string& image_path) {
    // 完全对齐Python的预处理逻辑
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        throw std::runtime_error("无法读取图像: " + image_path);
    }
    
    // 转换为RGB格式 (OpenCV默认是BGR)
    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    
    // 调整大小为224x224 - 对应Python的img.resize((224, 224))
    cv::Mat resized;
    cv::resize(rgb_img, resized, cv::Size(224, 224));
    
    // 转换为float32并归一化 - 对应Python的(img_array - 127.5) / 127.5
    resized.convertTo(resized, CV_32F);
    
    // MobileNetV2标准化: (pixel - 127.5) / 127.5, 对齐Python版本
    resized = (resized - 127.5) / 127.5;
    
    // NCNN使用CHW格式 - 对应Python的np.transpose(img_array, (2, 0, 1))
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    
    std::vector<float> input_tensor;
    input_tensor.reserve(1 * 3 * 224 * 224);
    
    // 按CHW顺序排列：C(R,G,B) H W
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < 224; ++h) {
            for (int w = 0; w < 224; ++w) {
                input_tensor.push_back(channels[c].at<float>(h, w));
            }
        }
    }
    
    return input_tensor;
}

std::vector<float> NCNNBackend::inference(const std::vector<float>& input_data) {
    try {
        // 创建NCNN Mat，CHW格式 - 使用基类成员变量 input_shape_
        // input_shape_ = {1, 3, 224, 224} -> NCNN Mat需要 (w=224, h=224, c=3)
        int batch = input_shape_[0];  // N
        int channels = input_shape_[1];  // C  
        int height = input_shape_[2];    // H
        int width = input_shape_[3];     // W
        ncnn::Mat input_mat(width, height, channels);  // w, h, c
        float* mat_data = input_mat;
        for (size_t i = 0; i < input_data.size(); ++i) {
            mat_data[i] = input_data[i];
        }
        
        // 创建NCNN Extractor - 对应Python的net.create_extractor()
        ncnn::Extractor extractor = net_.create_extractor();
        
        // 输入数据到网络 - 使用基类成员变量 input_name_
        int ret_input = extractor.input(input_name_.c_str(), input_mat);
        if (ret_input != 0) {
            throw std::runtime_error("NCNN输入数据设置失败");
        }
        
        // 执行推理并获取输出 - 使用基类成员变量 output_name_
        ncnn::Mat output_mat;
        int ret_output = extractor.extract(output_name_.c_str(), output_mat);
        if (ret_output != 0) {
            throw std::runtime_error("NCNN输出数据提取失败");
        }
        
        // 将输出Mat转换为vector - 内联实现
        std::vector<float> result;
        result.reserve(output_mat.w * output_mat.h * output_mat.c);
        const float* output_data = output_mat;
        for (int i = 0; i < output_mat.total(); ++i) {
            result.push_back(output_data[i]);
        }
        
        return result;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("NCNN推理失败: " + std::string(e.what()));
    }
}

std::vector<float> NCNNBackend::postprocess(const std::vector<float>& output_data) {
    // 对齐Python: """NCNN后处理：压缩多余维度"""
    // return np.squeeze(output_data)
    return output_data; // C++中vector已经是1维的，相当于squeeze后的结果
}

void NCNNBackend::cleanup() {
    // 对齐Python: """NCNN清理资源：释放网络"""
    // NCNN的ncnn::Net析构函数会自动清理资源
    net_.clear();
    std::cout << "NCNN资源已清理" << std::endl;
}

} // namespace mobilenet_inference