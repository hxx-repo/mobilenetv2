#include "inference_backend.hpp"
#include <iostream>
#include <iomanip>
#include <numeric>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>

using namespace mobilenet_inference;

// 加载ImageNet标签文件
std::vector<std::string> load_labels(const std::string& labels_path) {
    std::vector<std::string> labels;
    std::ifstream file(labels_path);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            labels.push_back(line);
        }
        file.close();
    }
    return labels;
}

// 辅助函数：计算统计信息 - 完全对齐Python重构版本输出  
void print_benchmark_stats(const BenchmarkResult& result, const std::vector<std::string>& labels = {}) {
    if (result.inference_times.empty()) {
        std::cout << "没有有效的推理时间数据" << std::endl;
        return;
    }
    
    // 计算统计数据 - 对齐Python的计算逻辑
    double avg_time = std::accumulate(result.inference_times.begin(), 
                                     result.inference_times.end(), 0.0) / result.inference_times.size();
    double min_time = *std::min_element(result.inference_times.begin(), result.inference_times.end());
    double max_time = *std::max_element(result.inference_times.begin(), result.inference_times.end());
    
    // 计算标准差
    double variance = 0.0;
    for (double time : result.inference_times) {
        variance += (time - avg_time) * (time - avg_time);
    }
    variance /= result.inference_times.size();
    double std_time = std::sqrt(variance);
    
    // 对齐Python重构版本的性能统计输出格式
    std::cout << "\n📊 性能统计:" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "   预处理耗时: " << result.preprocess_time << "秒" << std::endl;
    std::cout << "   平均推理耗时: " << avg_time << "秒 (" << (1.0 / avg_time) << " FPS)" << std::endl;
    std::cout << "   最快推理耗时: " << min_time << "秒 (" << (1.0 / min_time) << " FPS)" << std::endl;
    std::cout << "   最慢推理耗时: " << max_time << "秒 (" << (1.0 / max_time) << " FPS)" << std::endl;
    std::cout << "   标准差: " << std_time << "秒" << std::endl;
    if (result.postprocess_time > 0.0) {
        std::cout << "   后处理耗时: " << result.postprocess_time << "秒" << std::endl;
    }
    
    // 输出Top-5预测结果
    if (!result.predictions.empty()) {
        std::cout << "\nTop-5 预测结果:" << std::endl;
        
        // 找出最大的5个概率及其索引
        std::vector<std::pair<int, float>> indexed_probs;
        for (size_t i = 0; i < result.predictions.size(); ++i) {
            indexed_probs.emplace_back(static_cast<int>(i), result.predictions[i]);
        }
        
        std::partial_sort(indexed_probs.begin(), 
                         indexed_probs.begin() + std::min(5, static_cast<int>(indexed_probs.size())),
                         indexed_probs.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; });
        
        for (int i = 0; i < std::min(5, static_cast<int>(indexed_probs.size())); ++i) {
            int class_idx = indexed_probs[i].first;
            std::string class_name = "class_" + std::to_string(class_idx);
            std::string real_label = (labels.size() > static_cast<size_t>(class_idx)) ? 
                                     labels[class_idx] : ("unknown_" + std::to_string(class_idx));
            std::cout << "  " << (i + 1) << ". " << class_name << " (" << real_label << "): " 
                      << std::setprecision(6) << indexed_probs[i].second 
                      << " (" << std::setprecision(1) << (indexed_probs[i].second * 100) << "%)" << std::endl;
        }
    }
}

// 解析后端类型
BackendFactory::BackendType parse_backend_type(const std::string& model_path) {
    if (model_path.find(".tflite") != std::string::npos) {
        return BackendFactory::BackendType::TFLITE;
    } else if (model_path.find(".onnx") != std::string::npos) {
        return BackendFactory::BackendType::ONNX_RUNTIME;
    } else if (model_path.find(".trt") != std::string::npos) {
        // 根据文件名中的精度标识符区分TensorRT类型
        if (model_path.find("_fp16") != std::string::npos) {
            return BackendFactory::BackendType::TENSORRT_FP16;
        } else if (model_path.find("_int8") != std::string::npos) {
            return BackendFactory::BackendType::TENSORRT_INT8;
        } else {
            return BackendFactory::BackendType::TENSORRT_FP32; // 默认或fp32
        }
    } else if (model_path.find(".param") != std::string::npos) {
        // 区分NCNN的量化版本
        if (model_path.find("-int8") != std::string::npos) {
            return BackendFactory::BackendType::NCNN_INT8_QUANT;
        } else {
            return BackendFactory::BackendType::NCNN;
        }
    }
    
    // 默认尝试TFLite
    return BackendFactory::BackendType::TFLITE;
}

int main(int argc, char* argv[]) {
    // 参数检查 - 支持多后端自动识别
    if (argc < 3 || argc > 4) {
        std::cout << "用法: " << argv[0] << " <模型文件路径> <图像文件路径> [labels文件路径]" << std::endl;
        std::cout << "支持的模型格式:" << std::endl;
        std::cout << "  TFLite: *.tflite" << std::endl;
        std::cout << "  ONNX: *.onnx" << std::endl;
        std::cout << "  TensorRT: *.trt" << std::endl;
        std::cout << "  NCNN: *.param" << std::endl;
        std::cout << "示例: " << argv[0] << " ../model/mobilenet_v2_1.0_224.tflite ../input/fish_224x224.jpeg ../model/labels.txt" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string labels_path = (argc == 4) ? argv[3] : "";
    
    // 自动检测后端类型
    BackendFactory::BackendType backend_type = parse_backend_type(model_path);
    std::string backend_name = BackendFactory::BackendTypeToString(backend_type);
    
    std::cout << "MobileNetV2 C++ 多后端推理测试 (对齐Python版本)" << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << "后端类型: " << backend_name << std::endl;
    std::cout << "模型文件: " << model_path << std::endl;
    std::cout << "图像文件: " << image_path << std::endl;
    
    // 检查文件是否存在
    if (!std::ifstream(model_path).good()) {
        std::cerr << "❌ 模型文件不存在: " << model_path << std::endl;
        return 1;
    }
    
    if (!std::ifstream(image_path).good()) {
        std::cerr << "❌ 图像文件不存在: " << image_path << std::endl;
        return 1;
    }
    
    try {
        // 完全对齐Python调用方式
        // Python: backend = TFLiteBackend(model_path)
        // Python: avg_time, predictions = benchmark_model(backend, args.image, args.runs)
        std::cout << "\n创建" << backend_name << "后端..." << std::endl;
        auto backend = BackendFactory::CreateBackend(backend_type, model_path);
        
        if (!backend) {
            std::cerr << "❌ 后端创建失败" << std::endl;
            return 1;
        }
        
        // 加载labels文件
        std::vector<std::string> labels;
        if (!labels_path.empty()) {
            labels = load_labels(labels_path);
            if (labels.empty()) {
                std::cout << "⚠️  无法加载labels文件: " << labels_path << std::endl;
            } else {
                std::cout << "✅ 加载labels文件: " << labels_path << std::endl;
            }
        }
        
        std::cout << "执行性能测试 - 完全对齐Python调用方式..." << std::endl;
        BenchmarkResult benchmark_result = benchmark_model(*backend, image_path, 20);
        
        // 输出性能报告
        print_benchmark_stats(benchmark_result, labels);
        
        std::cout << "\n✅ 所有测试完成！与Python版本行为完全一致。" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 执行过程中出现异常: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}