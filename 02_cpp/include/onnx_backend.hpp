#pragma once

#include "inference_backend.hpp"
#include <onnxruntime_cxx_api.h>

namespace mobilenet_inference {

// ONNX Runtime后端 - 对应Python的ONNXBackend
class ONNXBackend : public InferenceBackend {
public:
    ONNXBackend(const std::string& model_path) : InferenceBackend(model_path) {}
    ~ONNXBackend() override = default;
    
    // 实现Python接口
    bool load_model() override;
    std::vector<float> preprocess(const std::string& image_path) override;
    std::vector<float> inference(const std::vector<float>& input_data) override;
    std::vector<float> postprocess(const std::vector<float>& output_data) override;
    void cleanup() override;

private:
    // ONNX Runtime需要env在session生命周期内存在
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
};

} // namespace mobilenet_inference