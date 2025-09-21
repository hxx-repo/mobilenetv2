#pragma once

#include "inference_backend.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

namespace mobilenet_inference {

// TensorFlow Lite后端 - 对应Python的TFLiteBackend
class TFLiteBackend : public InferenceBackend {
public:
    TFLiteBackend(const std::string& model_path) : InferenceBackend(model_path) {}
    ~TFLiteBackend() override = default;
    
    // 实现Python接口
    bool load_model() override;
    std::vector<float> preprocess(const std::string& image_path) override;
    std::vector<float> inference(const std::vector<float>& input_data) override;
    std::vector<float> postprocess(const std::vector<float>& output_data) override;
    void cleanup() override;

private:
    // 完全对齐Python版本的成员变量
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    const TfLiteTensor* input_details_ = nullptr;
    const TfLiteTensor* output_details_ = nullptr;
};

} // namespace mobilenet_inference