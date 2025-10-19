#pragma once

#include "inference_backend.hpp"
#include <MNN/Interpreter.hpp>

namespace mobilenet_inference {

class MNNBackend : public InferenceBackend {
public:
    explicit MNNBackend(const std::string& model_path);
    ~MNNBackend() override;
    
    bool load_model() override;
    std::vector<float> preprocess(const std::string& image_path) override;
    std::vector<float> inference(const std::vector<float>& input_data) override;
    std::vector<float> postprocess(const std::vector<float>& output_data) override;
    void cleanup() override;

private:
    std::string halide_type_to_string(const halide_type_t& type) const;
    std::shared_ptr<MNN::Interpreter> interpreter_;
    MNN::Session* session_ = nullptr;
    MNN::Tensor* input_tensor_device_ = nullptr;
    MNN::Tensor* output_tensor_device_ = nullptr;
    std::shared_ptr<MNN::Tensor> input_tensor_host_;
    std::shared_ptr<MNN::Tensor> output_tensor_host_;
    
    MNN::ScheduleConfig schedule_config_;
    MNN::BackendConfig backend_config_;
};

} // namespace mobilenet_inference
