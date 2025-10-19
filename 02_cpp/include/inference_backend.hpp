#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace mobilenet_inference {

// 推理后端基类 - 完全对齐Python的InferenceBackend设计
class InferenceBackend {
public:
    InferenceBackend(const std::string& model_path) : model_path_(model_path) {}
    virtual ~InferenceBackend() = default;
    
    // 对齐Python接口的所有方法
    virtual bool load_model() = 0;
    virtual std::vector<float> preprocess(const std::string& image_path) = 0;
    virtual std::vector<float> inference(const std::vector<float>& input_data) = 0;
    
    // 后处理 - 在子类中实现，对应Python的每个子类都有自己的postprocess
    virtual std::vector<float> postprocess(const std::vector<float>& output_data) = 0;
    
    // 资源清理 - 对齐Python的cleanup方法
    virtual void cleanup() = 0;
    
    // 获取模型信息 - 对应Python的属性
    std::vector<int> input_shape() const { return input_shape_; }
    std::vector<int> output_shape() const { return output_shape_; }
    std::string input_name() const { return input_name_; }
    std::string output_name() const { return output_name_; }
    std::string input_dtype() const { return input_dtype_; }
    std::string output_dtype() const { return output_dtype_; }
    
protected:
    // 完全对齐Python版本的成员变量
    std::string model_path_;
    std::string input_name_;
    std::string output_name_;
    std::vector<int> input_shape_;
    std::vector<int> output_shape_;
    std::string input_dtype_;
    std::string output_dtype_;
};

// 工厂模式创建后端 - 按Python顺序
class BackendFactory {
public:
    enum class BackendType {
        TFLITE,           // TensorFlow Lite - 对应TFLiteBackend
        ONNX_RUNTIME,     // ONNX Runtime - 对应ONNXBackend  
        TENSORRT_FP32,    // TensorRT FP32 - 对应TensorRTBackend
        TENSORRT_FP16,    // TensorRT FP16
        TENSORRT_INT8,    // TensorRT INT8
        NCNN,             // NCNN - 对应NCNNBackend
        NCNN_INT8_QUANT,  // NCNN INT8量化版本 (重命名避免宏冲突)
        MNN,              // MNN - 对应MNNBackend
        MNN_INT8          // MNN INT8量化版本
    };
    
    static std::unique_ptr<InferenceBackend> CreateBackend(BackendType backend_type, const std::string& model_path);
    static std::string BackendTypeToString(BackendType backend_type);
};

// 独立的benchmark函数 - 对齐Python的benchmark_model
struct BenchmarkResult {
    std::vector<double> inference_times;  // 所有推理时间
    std::vector<float> predictions;       // 首次推理的结果
    double preprocess_time;               // 预处理时间
    double postprocess_time;              // 后处理时间
};

// 完全对齐Python调用方式: benchmark_model(backend, image_path, test_runs)
BenchmarkResult benchmark_model(InferenceBackend& backend,
                               const std::string& image_path,
                               int test_runs = 20);

} // namespace mobilenet_inference
