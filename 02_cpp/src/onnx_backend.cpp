#include "onnx_backend.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <stdexcept>

namespace mobilenet_inference {

// 构造函数现在是内联的，这里不需要了

bool ONNXBackend::load_model() {
    try {
        // 完全对齐Python的加载逻辑
        std::cout << "加载ONNX模型: " << model_path_ << std::endl;
        std::cout << "💻 使用CPU优化执行" << std::endl;
        
        // 创建env作为成员变量，确保在session生命周期内存在
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ONNXBackend");
        Ort::SessionOptions session_options;
        
        // 配置会话选项 - 对应Python的providers = ['CPUExecutionProvider']
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // 创建推理会话 - 对应Python的ort.InferenceSession(model_path, providers=providers)
        session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(), session_options);
        
        // 获取输入输出信息 - 对应Python的session.get_inputs()和get_outputs()
        Ort::AllocatorWithDefaultOptions allocator;
        
        // 获取模型信息并设置基类成员变量
        auto input_name_ptr = session_->GetInputNameAllocated(0, allocator);
        auto output_name_ptr = session_->GetOutputNameAllocated(0, allocator);
        
        input_name_ = std::string(input_name_ptr.get());
        output_name_ = std::string(output_name_ptr.get());
        
        // 获取输入输出形状和类型信息
        auto input_info = session_->GetInputTypeInfo(0);
        auto output_info = session_->GetOutputTypeInfo(0);
        
        auto input_tensor_info = input_info.GetTensorTypeAndShapeInfo();
        auto output_tensor_info = output_info.GetTensorTypeAndShapeInfo();
        
        auto input_dims = input_tensor_info.GetShape();
        auto output_dims = output_tensor_info.GetShape();
        
        input_shape_.clear();
        output_shape_.clear();
        for (auto dim : input_dims) {
            input_shape_.push_back(static_cast<int>(dim));
        }
        for (auto dim : output_dims) {
            output_shape_.push_back(static_cast<int>(dim));
        }
        
        // 获取数据类型
        auto input_element_type = input_tensor_info.GetElementType();
        input_dtype_ = (input_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ? "float32" : "unknown";
        
        auto output_element_type = output_tensor_info.GetElementType();
        output_dtype_ = (output_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ? "float32" : "unknown";
        
        // 完全对齐Python的打印格式
        std::cout << "✅ ONNX模型加载成功" << std::endl;
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
        std::cout << "   执行提供程序: [\"CPUExecutionProvider\"]" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ ONNX模型加载失败: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> ONNXBackend::preprocess(const std::string& image_path) {
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
    
    // 转换为CHW格式 - 对应Python的np.transpose(img_array, (2, 0, 1))
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

std::vector<float> ONNXBackend::inference(const std::vector<float>& input_data) {
    try {
        // 创建输入张量 - 对应Python的session.run(None, {input_name: input_data})
        // 使用基类成员变量 input_shape_ 而不是硬编码
        std::vector<int64_t> input_shape_int64;
        for (int dim : input_shape_) {
            input_shape_int64.push_back(static_cast<int64_t>(dim));
        }
        
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        // 创建输入张量
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, 
            const_cast<float*>(input_data.data()), 
            input_data.size(), 
            input_shape_int64.data(), 
            input_shape_int64.size()
        ));
        
        // 执行推理 - 使用基类成员变量
        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};
        
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr}, 
            input_names, 
            input_tensors.data(), 
            1, 
            output_names, 
            1
        );
        
        // 获取输出数据
        float* output_data = output_tensors.front().GetTensorMutableData<float>();
        auto output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
        
        size_t output_size = 1;
        for (auto dim : output_shape) {
            output_size *= dim;
        }
        
        std::vector<float> result(output_data, output_data + output_size);
        return result;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("ONNX推理失败: " + std::string(e.what()));
    }
}

std::vector<float> ONNXBackend::postprocess(const std::vector<float>& output_data) {
    // 对齐Python: """ONNX后处理：压缩多余维度"""
    // return np.squeeze(output_data)
    return output_data; // C++中vector已经是1维的，相当于squeeze后的结果
}

void ONNXBackend::cleanup() {
    // ONNX清理资源：释放会话和环境
    if (session_) {
        session_.reset();
    }
    if (env_) {
        env_.reset();
    }
    std::cout << "ONNX资源已清理" << std::endl;
}

} // namespace mobilenet_inference