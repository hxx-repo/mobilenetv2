#include "inference_backend.hpp"
#ifdef ENABLE_TFLITE
#include "tflite_backend.hpp"
#endif
#ifdef ENABLE_ONNXRUNTIME
#include "onnx_backend.hpp"
#endif
#ifdef ENABLE_TENSORRT
#include "tensorrt_backend.hpp"
#endif
#ifdef ENABLE_NCNN
#include "ncnn_backend.hpp"
#endif
#include <iostream>
#include <chrono>
#include <numeric>
#include <cmath>
#include <iomanip>

namespace mobilenet_inference {

// 工厂方法实现
std::unique_ptr<InferenceBackend> BackendFactory::CreateBackend(BackendType backend_type, const std::string& model_path) {
    switch (backend_type) {
        case BackendType::TFLITE:
#ifdef ENABLE_TFLITE
            return std::make_unique<TFLiteBackend>(model_path);
#else
            std::cerr << "TensorFlow Lite支持未编译进此版本" << std::endl;
            return nullptr;
#endif
            
        case BackendType::ONNX_RUNTIME:
#ifdef ENABLE_ONNXRUNTIME
            return std::make_unique<ONNXBackend>(model_path);
#else
            std::cerr << "ONNX Runtime支持未编译进此版本" << std::endl;
            return nullptr;
#endif
            
        case BackendType::TENSORRT_FP32:
        case BackendType::TENSORRT_FP16:
        case BackendType::TENSORRT_INT8:
#ifdef ENABLE_TENSORRT
            return std::make_unique<TensorRTBackend>(model_path);
#else
            std::cerr << "TensorRT支持未编译进此版本" << std::endl;
            return nullptr;
#endif
            
        case BackendType::NCNN:
        case BackendType::NCNN_INT8_QUANT:
#ifdef ENABLE_NCNN
            return std::make_unique<NCNNBackend>(model_path);
#else
            std::cerr << "NCNN支持未编译进此版本" << std::endl;
            return nullptr;
#endif
            
        default:
            return nullptr;
    }
}

// 后端类型转字符串
std::string BackendFactory::BackendTypeToString(BackendType backend_type) {
    switch (backend_type) {
        case BackendType::TFLITE:       return "TensorFlow Lite";
        case BackendType::ONNX_RUNTIME: return "ONNX Runtime";
        case BackendType::TENSORRT_FP32: return "TensorRT FP32";
        case BackendType::TENSORRT_FP16: return "TensorRT FP16";
        case BackendType::TENSORRT_INT8: return "TensorRT INT8";
        case BackendType::NCNN:         return "NCNN";
        case BackendType::NCNN_INT8_QUANT:    return "NCNN INT8";
        default:                        return "Unknown";
    }
}

// 独立的benchmark_model函数 - 完全对齐Python版本调用方式
BenchmarkResult benchmark_model(InferenceBackend& backend,
                               const std::string& image_path,
                               int test_runs) {
    // 完全对齐Python的benchmark_model函数逻辑
    std::cout << "\n=== 性能测试 ===" << std::endl;
    std::cout << "测试次数: " << test_runs << std::endl;
    
    BenchmarkResult result;
    result.postprocess_time = 0.0; // 初始化
    
    // 加载模型 - 对齐Python: if not backend.load_model(): return None, None
    if (!backend.load_model()) {
        std::cout << "模型加载失败" << std::endl;
        return result;
    }
    
    try {
        // 预处理图像 - 对齐Python: input_data = backend.preprocess(image_path)
        std::cout << "正在预处理图像..." << std::endl;
        auto preprocess_start = std::chrono::high_resolution_clock::now();
        std::vector<float> input_data = backend.preprocess(image_path);
        auto preprocess_end = std::chrono::high_resolution_clock::now();
        
        result.preprocess_time = std::chrono::duration<double>(
            preprocess_end - preprocess_start).count(); // 秒
        
        // 性能测试 - 对齐Python: 没有热身，直接测试
        std::cout << "正在执行性能测试..." << std::endl;
        result.inference_times.reserve(test_runs);
        
        for (int i = 0; i < test_runs; ++i) {
            try {
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<float> output = backend.inference(input_data);
                auto end = std::chrono::high_resolution_clock::now();
                
                double inference_time = std::chrono::duration<double>(
                    end - start).count(); // 秒
                result.inference_times.push_back(inference_time);
                
                // 首次推理时测量后处理时间
                if (i == 0) {
                    auto postprocess_start = std::chrono::high_resolution_clock::now();
                    result.predictions = backend.postprocess(output);
                    auto postprocess_end = std::chrono::high_resolution_clock::now();
                    
                    result.postprocess_time = std::chrono::duration<double>(
                        postprocess_end - postprocess_start).count(); // 秒
                    
                    std::cout << "首次推理完成" << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cout << "推理失败 (第" << (i+1) << "次): " << e.what() << std::endl;
                return result; // 返回部分结果
            }
        }
    } catch (const std::exception& e) {
        std::cout << "预处理失败: " << e.what() << std::endl;
    }
    
    // finally块 - 对齐Python: 清理资源
    backend.cleanup();
    
    return result;
}

} // namespace mobilenet_inference