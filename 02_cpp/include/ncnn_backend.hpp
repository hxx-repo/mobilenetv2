#pragma once

#include "inference_backend.hpp"
#include <ncnn/net.h>

namespace mobilenet_inference {

// NCNN后端实现 - 对齐Python的NCNNBackend
class NCNNBackend : public InferenceBackend {
public:
    NCNNBackend(const std::string& model_path);
    ~NCNNBackend() override = default;
    
    bool load_model() override;
    std::vector<float> preprocess(const std::string& image_path) override;
    std::vector<float> inference(const std::vector<float>& input_data) override;
    std::vector<float> postprocess(const std::vector<float>& output_data) override;
    void cleanup() override;
    
private:
    // 完全对齐Python版本：只有net一个成员变量
    ncnn::Net net_;
};

} // namespace mobilenet_inference