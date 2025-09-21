#pragma once

#include "inference_backend.hpp"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <memory>

namespace mobilenet_inference {

// TensorRT后端 - 对应Python的TensorRTBackend
class TensorRTBackend : public InferenceBackend {
public:
    TensorRTBackend(const std::string& model_path) : InferenceBackend(model_path) {}
    ~TensorRTBackend() override;
    
    // 实现Python接口
    bool load_model() override;
    std::vector<float> preprocess(const std::string& image_path) override;
    std::vector<float> inference(const std::vector<float>& input_data) override;
    std::vector<float> postprocess(const std::vector<float>& output_data) override;
    void cleanup() override;

private:
    // 对齐Python版本：engine, context, 和GPU内存缓冲区
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_; // 需要保持runtime生命周期
    std::unique_ptr<nvinfer1::IExecutionContext> context_; // 对应Python的self.context
    
    // GPU内存缓冲区 - 对应Python的self.d_input和self.d_output
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
};

// TensorRT Logger - 必需的日志类
class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

} // namespace mobilenet_inference