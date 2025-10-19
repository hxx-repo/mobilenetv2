#include "mnn_backend.hpp"
#include <MNN/Tensor.hpp>
#include <MNN/HalideRuntime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>

namespace mobilenet_inference {

MNNBackend::MNNBackend(const std::string& model_path)
    : InferenceBackend(model_path) {}

MNNBackend::~MNNBackend() {
    cleanup();
}

std::string MNNBackend::halide_type_to_string(const halide_type_t& type) const {
    if (type.code == halide_type_float && type.bits == 32) {
        return "float32";
    }
    if (type.code == halide_type_float && type.bits == 16) {
        return "float16";
    }
    if (type.code == halide_type_uint && type.bits == 8) {
        return "uint8";
    }
    if (type.code == halide_type_int && type.bits == 8) {
        return "int8";
    }
    if (type.code == halide_type_int && type.bits == 32) {
        return "int32";
    }
    return "unknown";
}

bool MNNBackend::load_model() {
    try {
        interpreter_.reset(MNN::Interpreter::createFromFile(model_path_.c_str()));
        if (!interpreter_) {
            std::cerr << "❌ 创建MNN解释器失败: " << model_path_ << std::endl;
            return false;
        }

        schedule_config_.type = MNN_FORWARD_CPU;
        schedule_config_.numThread = 4;
        backend_config_.precision = MNN::BackendConfig::Precision_High;
        backend_config_.power = MNN::BackendConfig::Power_High;
        backend_config_.memory = MNN::BackendConfig::Memory_Normal;
        schedule_config_.backendConfig = &backend_config_;

        session_ = interpreter_->createSession(schedule_config_);
        if (!session_) {
            std::cerr << "❌ 创建MNN会话失败" << std::endl;
            return false;
        }

        auto inputs_map = interpreter_->getSessionInputAll(session_);
        if (inputs_map.empty()) {
            std::cerr << "❌ 无法获取MNN输入Tensor" << std::endl;
            return false;
        }

        input_name_ = inputs_map.begin()->first;
        input_tensor_device_ = inputs_map.begin()->second;

        input_shape_ = input_tensor_device_->shape();
        if (input_shape_.size() == 4 && input_tensor_device_->getDimensionType() == MNN::Tensor::TENSORFLOW) {
            input_shape_ = {input_shape_[0], input_shape_[3], input_shape_[1], input_shape_[2]};
        }

        input_tensor_host_.reset(new MNN::Tensor(input_tensor_device_, input_tensor_device_->getDimensionType()));

        auto outputs_map = interpreter_->getSessionOutputAll(session_);
        if (outputs_map.empty()) {
            throw std::runtime_error("无法获取MNN输出Tensor");
        }
        output_name_ = outputs_map.begin()->first;
        output_tensor_device_ = outputs_map.begin()->second;
        output_shape_ = output_tensor_device_->shape();
        if (output_shape_.empty()) {
            output_shape_ = {1, static_cast<int>(output_tensor_device_->elementSize())};
        }
        output_tensor_host_.reset(new MNN::Tensor(output_tensor_device_, output_tensor_device_->getDimensionType()));

        input_dtype_ = halide_type_to_string(input_tensor_device_->getType());
        output_dtype_ = halide_type_to_string(output_tensor_device_->getType());

        std::cout << "✅ MNN模型加载成功" << std::endl;
        std::cout << "   输入名称: " << input_name_ << std::endl;
        std::cout << "   输入形状: [";
        for (size_t i = 0; i < input_shape_.size(); ++i) {
            std::cout << input_shape_[i] << (i + 1 == input_shape_.size() ? "" : ", ");
        }
        std::cout << "]" << std::endl;
        std::cout << "   输入数据类型: " << input_dtype_ << std::endl;
        std::cout << "   输出名称: " << output_name_ << std::endl;
        std::cout << "   输出形状: [";
        for (size_t i = 0; i < output_shape_.size(); ++i) {
            std::cout << output_shape_[i] << (i + 1 == output_shape_.size() ? "" : ", ");
        }
        std::cout << "]" << std::endl;
        std::cout << "   输出数据类型: " << output_dtype_ << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "❌ MNN模型加载失败: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> MNNBackend::preprocess(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("无法读取图像: " + image_path);
    }

    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(224, 224));

    resized.convertTo(resized, CV_32F);
    resized = (resized - 127.5f) / 127.5f;

    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);

    std::vector<float> input_tensor;
    input_tensor.reserve(1 * 3 * 224 * 224);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < 224; ++h) {
            const float* row_ptr = channels[c].ptr<float>(h);
            input_tensor.insert(input_tensor.end(), row_ptr, row_ptr + 224);
        }
    }

    return input_tensor;
}

std::vector<float> MNNBackend::inference(const std::vector<float>& input_data) {
    try {
        if (!interpreter_ || session_ == nullptr) {
            throw std::runtime_error("MNN会话未初始化");
        }

        std::copy(input_data.begin(), input_data.end(), input_tensor_host_->host<float>());
        input_tensor_device_->copyFromHostTensor(input_tensor_host_.get());
        interpreter_->runSession(session_);
        output_tensor_device_->copyToHostTensor(output_tensor_host_.get());

        const float* output_ptr = output_tensor_host_->host<float>();
        size_t output_size = static_cast<size_t>(output_tensor_host_->elementSize());
        if (output_size == 0) {
            output_size = 1;
            for (int dim : output_shape_) {
                output_size *= static_cast<size_t>(dim);
            }
        }

        return std::vector<float>(output_ptr, output_ptr + output_size);

    } catch (const std::exception& e) {
        throw std::runtime_error("MNN推理失败: " + std::string(e.what()));
    }
}

std::vector<float> MNNBackend::postprocess(const std::vector<float>& output_data) {
    return output_data;
}

void MNNBackend::cleanup() {
    if (interpreter_ && session_) {
        interpreter_->releaseSession(session_);
        session_ = nullptr;
    }
    input_tensor_host_.reset();
    output_tensor_host_.reset();
    interpreter_.reset();
    input_tensor_device_ = nullptr;
    output_tensor_device_ = nullptr;
    std::cout << "MNN资源已清理" << std::endl;
}

} // namespace mobilenet_inference
