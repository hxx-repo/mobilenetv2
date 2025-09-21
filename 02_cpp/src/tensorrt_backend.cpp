#include "tensorrt_backend.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>

namespace mobilenet_inference {

// TensorRT Logger实现
void TensorRTLogger::log(Severity severity, const char* msg) noexcept {
    // 只输出Warning和Error级别的日志，避免过多输出
    if (severity <= Severity::kWARNING) {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "[TensorRT INTERNAL_ERROR] " << msg << std::endl;
                break;
            case Severity::kERROR:
                std::cerr << "[TensorRT ERROR] " << msg << std::endl;
                break;
            case Severity::kWARNING:
                std::cout << "[TensorRT WARNING] " << msg << std::endl;
                break;
            default:
                break;
        }
    }
}

// TensorRT后端析构函数 - 对齐Python版本简化
TensorRTBackend::~TensorRTBackend() {
    // TensorRT engine会自动管理资源，类似Python版本
}

bool TensorRTBackend::load_model() {
    try {
        std::cout << "加载TensorRT引擎: " << model_path_ << std::endl;
        
        // 读取序列化的引擎文件 - 对齐Python版本
        std::ifstream file(model_path_, std::ios::binary);
        if (!file.good()) {
            std::cerr << "无法打开TensorRT引擎文件: " << model_path_ << std::endl;
            return false;
        }
        
        file.seekg(0, std::ios::end);
        size_t engine_size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> engine_data(engine_size);
        file.read(engine_data.data(), engine_size);
        file.close();
        
        // 创建TensorRT运行时和引擎 - 修复生命周期问题
        static TensorRTLogger logger;
        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
        if (!runtime_) {
            std::cerr << "创建TensorRT运行时失败" << std::endl;
            return false;
        }
        
        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(engine_data.data(), engine_size, nullptr));
        if (!engine_) {
            std::cerr << "反序列化TensorRT引擎失败" << std::endl;
            return false;
        }
        
        // ✅ 通过API获取：输入输出信息 - 完全对齐Python版本
        // Python: self.input_name = self.engine.get_tensor_name(0)
        input_name_ = std::string(engine_->getIOTensorName(0));
        output_name_ = std::string(engine_->getIOTensorName(1));
        
        // Python: self.input_shape = self.engine.get_tensor_shape(self.input_name)
        auto input_dims = engine_->getTensorShape(input_name_.c_str());
        auto output_dims = engine_->getTensorShape(output_name_.c_str());
        
        input_shape_.clear();
        output_shape_.clear();
        for (int i = 0; i < input_dims.nbDims; ++i) {
            input_shape_.push_back(input_dims.d[i]);
        }
        for (int i = 0; i < output_dims.nbDims; ++i) {
            output_shape_.push_back(output_dims.d[i]);
        }
        
        // Python: self.input_dtype = self.engine.get_tensor_dtype(self.input_name)
        auto input_dtype = engine_->getTensorDataType(input_name_.c_str());
        auto output_dtype = engine_->getTensorDataType(output_name_.c_str());
        
        // 转换TensorRT数据类型到字符串
        input_dtype_ = (input_dtype == nvinfer1::DataType::kFLOAT) ? "float32" : "unknown";
        output_dtype_ = (output_dtype == nvinfer1::DataType::kFLOAT) ? "float32" : "unknown";
        
        // 创建执行上下文 - 对应Python的self.context = self.engine.create_execution_context()
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_) {
            std::cerr << "创建TensorRT执行上下文失败" << std::endl;
            return false;
        }
        
        // 分配GPU内存 - 完全对齐Python版本的内存分配逻辑
        size_t input_size = 1;
        size_t output_size = 1;
        for (int dim : input_shape_) {
            input_size *= dim;
        }
        for (int dim : output_shape_) {
            output_size *= dim;
        }
        input_size *= sizeof(float);  // Python: np.dtype(np.float32).itemsize
        output_size *= sizeof(float);
        
        // Python: self.d_input = cuda.mem_alloc(input_size)
        if (cudaMalloc(&d_input_, input_size) != cudaSuccess) {
            std::cerr << "分配GPU输入缓冲区失败" << std::endl;
            return false;
        }
        
        // Python: self.d_output = cuda.mem_alloc(output_size)
        if (cudaMalloc(&d_output_, output_size) != cudaSuccess) {
            cudaFree(d_input_);
            std::cerr << "分配GPU输出缓冲区失败" << std::endl;
            return false;
        }
        
        // 完全对齐Python的打印格式
        std::cout << "✅ TensorRT模型加载成功" << std::endl;
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
        
        // 对齐Python版本的GPU内存信息打印
        std::cout << "   GPU内存: 输入" << (input_size/1024) << "KB, 输出" << (output_size/1024) << "KB" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ TensorRT模型加载失败: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> TensorRTBackend::preprocess(const std::string& image_path) {
    // 完全对齐Python的预处理逻辑
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        throw std::runtime_error("无法读取图像: " + image_path);
    }
    
    // 转换为RGB格式 (OpenCV默认是BGR)
    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    
    // 调整大小为224x224 - 对应Python的img.resize((224, 224))
    cv::Mat resized;
    cv::resize(rgb_img, resized, cv::Size(224, 224));
    
    // 转换为float32并归一化 - 对应Python的(img_array - 127.5) / 127.5
    resized.convertTo(resized, CV_32F);
    
    // MobileNetV2标准化: (pixel - 127.5) / 127.5, 对齐Python版本
    resized = (resized - 127.5) / 127.5;
    
    // TensorRT通常使用CHW格式 - 对应Python的np.transpose(img_array, (2, 0, 1))
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    
    std::vector<float> input_tensor;
    input_tensor.reserve(1 * 3 * 224 * 224);
    
    // 按CHW顺序排列：C(R,G,B) H W
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < 224; ++h) {
            for (int w = 0; w < 224; ++w) {
                input_tensor.push_back(channels[c].at<float>(h, w));
            }
        }
    }
    
    return input_tensor;
}

std::vector<float> TensorRTBackend::inference(const std::vector<float>& input_data) {
    try {
        // 完全对齐Python版本 - 使用预分配的GPU内存和context
        
        // Python: input_host = np.ascontiguousarray(input_data.ravel()).astype(np.float32)
        // Python: cuda.memcpy_htod(self.d_input, input_host)
        size_t input_size = input_data.size() * sizeof(float);
        if (cudaMemcpy(d_input_, input_data.data(), input_size, cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("输入数据拷贝到GPU失败");
        }
        
        // Python: bindings = [int(self.d_input), int(self.d_output)]
        // Python: self.context.execute_v2(bindings)
        void* bindings[] = {d_input_, d_output_};
        
        if (!context_->executeV2(bindings)) {
            throw std::runtime_error("TensorRT推理执行失败");
        }
        
        // Python: output_host = np.empty(self.output_shape, dtype=np.float32)
        // Python: cuda.memcpy_dtoh(output_host, self.d_output)
        size_t output_size = 1;
        for (int dim : output_shape_) {
            output_size *= dim;
        }
        
        std::vector<float> output_data(output_size);
        size_t output_bytes = output_size * sizeof(float);
        if (cudaMemcpy(output_data.data(), d_output_, output_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("输出数据从GPU拷贝失败");
        }
        
        return output_data;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("TensorRT推理失败: " + std::string(e.what()));
    }
}

std::vector<float> TensorRTBackend::postprocess(const std::vector<float>& output_data) {
    // 对齐Python: """TensorRT后处理：压缩多余维度"""
    return output_data; // C++中vector已经是1维的，相当于squeeze后的结果
}

void TensorRTBackend::cleanup() {
    // 完全对齐Python: """TensorRT清理资源：释放GPU内存"""
    // Python cleanup逻辑:
    // if hasattr(self, 'd_input') and self.d_input is not None:
    //     self.d_input.free()
    // if hasattr(self, 'd_output') and self.d_output is not None:
    //     self.d_output.free()
    
    if (d_input_ != nullptr) {
        cudaFree(d_input_);
        d_input_ = nullptr;
    }
    
    if (d_output_ != nullptr) {
        cudaFree(d_output_);
        d_output_ = nullptr;
    }
    
    // 清理context和engine
    context_.reset();
    engine_.reset();
    runtime_.reset();
    std::cout << "TensorRT资源已清理" << std::endl;
}

} // namespace mobilenet_inference