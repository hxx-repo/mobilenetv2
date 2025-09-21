#include "tflite_backend.hpp"
#include <iostream>

namespace mobilenet_inference {

bool TFLiteBackend::load_model() {
    try {
        std::cout << "加载TFLite模型: " << model_path_ << std::endl;
        
        // 从文件加载模型
        model_ = tflite::FlatBufferModel::BuildFromFile(model_path_.c_str());
        if (!model_) {
            std::cerr << "❌ 加载TFLite模型文件失败: " << model_path_ << std::endl;
            return false;
        }
        
        // 创建操作解析器和解释器
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*model_, resolver);
        builder(&interpreter_);
        
        if (!interpreter_) {
            std::cerr << "❌ 创建TFLite解释器失败" << std::endl;
            return false;
        }
        
        // 分配张量内存
        if (interpreter_->AllocateTensors() != kTfLiteOk) {
            std::cerr << "❌ TFLite张量内存分配失败" << std::endl;
            return false;
        }
        
        // 设置线程数量（CPU优化）
        interpreter_->SetNumThreads(4);
        
        // 获取输入输出details - 对齐Python版本的input_details和output_details
        input_details_ = interpreter_->tensor(interpreter_->inputs()[0]);
        output_details_ = interpreter_->tensor(interpreter_->outputs()[0]);
        
        // 对齐Python的成员变量设置 - 从input_details和output_details获取信息
        input_name_ = input_details_->name ? input_details_->name : "input";
        output_name_ = output_details_->name ? output_details_->name : "output";
        
        input_shape_.clear();
        output_shape_.clear();
        
        for (int i = 0; i < input_details_->dims->size; ++i) {
            input_shape_.push_back(input_details_->dims->data[i]);
        }
        
        for (int i = 0; i < output_details_->dims->size; ++i) {
            output_shape_.push_back(output_details_->dims->data[i]);
        }
        
        // 获取数据类型
        input_dtype_ = (input_details_->type == kTfLiteFloat32) ? "float32" : "unknown";
        output_dtype_ = (output_details_->type == kTfLiteFloat32) ? "float32" : "unknown";
        
        // 完全对齐Python的打印格式
        std::cout << "✅ TFLite模型加载成功" << std::endl;
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
        std::cerr << "❌ TFLite模型加载异常: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> TFLiteBackend::preprocess(const std::string& image_path) {
    // 对齐Python的TFLite预处理逻辑:
    // """TFLite预处理 (HWC格式) - MobileNetV2标准化"""
    // img = Image.open(image_path).resize((224, 224))
    // img = np.array(img).astype(np.float32)
    // img = (img - 127.5) / 127.5  # MobileNetV2标准化: [-1, 1] 范围
    // img = np.expand_dims(img, axis=0)
    
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        throw std::runtime_error("无法加载图像: " + image_path);
    }
    
    // 调整大小到224x224
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(224, 224));
    
    // 确保BGR转RGB（OpenCV默认是BGR）
    cv::Mat rgb_img;
    cv::cvtColor(resized_img, rgb_img, cv::COLOR_BGR2RGB);
    
    // 转换为float32
    rgb_img.convertTo(rgb_img, CV_32F);
    
    // MobileNetV2标准化: (pixel - 127.5) / 127.5
    rgb_img = (rgb_img - 127.5) / 127.5;
    
    // 转换为vector<float> - HWC格式 (TFLite用HWC)
    std::vector<float> input_tensor;
    input_tensor.reserve(1 * 224 * 224 * 3);  // NHWC: 1x224x224x3
    
    // 按HWC顺序存储数据
    for (int h = 0; h < 224; ++h) {
        for (int w = 0; w < 224; ++w) {
            cv::Vec3f pixel = rgb_img.at<cv::Vec3f>(h, w);
            input_tensor.push_back(pixel[0]); // R
            input_tensor.push_back(pixel[1]); // G  
            input_tensor.push_back(pixel[2]); // B
        }
    }
    
    return input_tensor;
}

std::vector<float> TFLiteBackend::inference(const std::vector<float>& input_data) {
    // 对齐Python的TFLite推理逻辑:
    // """TFLite推理"""
    // self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
    // self.interpreter.invoke()
    // output = self.interpreter.get_tensor(self.output_details[0]['index'])
    // return output
    
    // 完全对齐Python的推理逻辑 - 使用input_details_和output_details_的等价方法
    // Python: self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
    float* input_tensor_ptr = interpreter_->typed_input_tensor<float>(0);
    if (!input_tensor_ptr) {
        throw std::runtime_error("获取TFLite输入张量失败");
    }
    
    // 拷贝输入数据 - 对应Python的set_tensor
    std::copy(input_data.begin(), input_data.end(), input_tensor_ptr);
    
    // 执行推理 - 对应 self.interpreter.invoke()
    if (interpreter_->Invoke() != kTfLiteOk) {
        throw std::runtime_error("TFLite推理执行失败");
    }
    
    // 获取输出张量 - 对应 self.interpreter.get_tensor(self.output_details[0]['index'])
    const float* output_tensor_ptr = interpreter_->typed_output_tensor<float>(0);
    if (!output_tensor_ptr) {
        throw std::runtime_error("获取TFLite输出张量失败");
    }
    
    // 拷贝输出数据
    size_t output_size = 1;
    for (int dim : output_shape_) {
        output_size *= static_cast<size_t>(dim);
    }
    
    std::vector<float> output_data(output_tensor_ptr, output_tensor_ptr + output_size);
    return output_data;
}

std::vector<float> TFLiteBackend::postprocess(const std::vector<float>& output_data) {
    // 对齐Python: """TFLite后处理：压缩多余维度"""
    // return np.squeeze(output_data)
    return output_data; // C++中vector已经是1维的，相当于squeeze后的结果
}

void TFLiteBackend::cleanup() {
    // 对齐Python: """TFLite清理资源：释放解释器"""
    if (interpreter_) {
        // TensorFlow Lite解释器会自动管理内存
        interpreter_.reset();
    }
    if (model_) {
        model_.reset();
    }
    input_details_ = nullptr;
    output_details_ = nullptr;
    std::cout << "TFLite资源已清理" << std::endl;
}

} // namespace mobilenet_inference