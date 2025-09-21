#!/usr/bin/env python3
"""
03. 多后端性能测试：TFLite、ONNX、TensorRT、NCNN全面对比
第三步执行：对比所有推理后端的性能表现
运行前确保: 01_infer_basic.py 和 02_convert_model.py 成功完成
"""

import argparse
import os
import time
import numpy as np
from PIL import Image

class InferenceBackend:
    """推理后端基类"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        self.input_dtype = None
        self.output_dtype = None
        
    def load_model(self):
        """加载模型"""
        raise NotImplementedError
        
    def preprocess(self, image_path):
        """预处理图像"""
        raise NotImplementedError
        
    def inference(self, input_data):
        """执行推理"""
        raise NotImplementedError
        
    def postprocess(self, output_data):
        """后处理输出"""
        raise NotImplementedError
    
    def cleanup(self):
        """清理资源"""
        raise NotImplementedError

class TFLiteBackend(InferenceBackend):
    """TensorFlow Lite后端"""
    
    def load_model(self):
        try:
            import tensorflow as tf
            
            print(f"加载TFLite模型: {self.model_path}")
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.input_name = self.input_details[0]['name']
            self.output_name = self.output_details[0]['name']
            self.input_shape = self.input_details[0]['shape']
            self.output_shape = self.output_details[0]['shape']
            self.input_dtype = self.input_details[0]['dtype']
            self.output_dtype = self.output_details[0]['dtype']
            
            print(f"✅ TFLite模型加载成功")
            print(f"   输入名称: {self.input_name}")
            print(f"   输入形状: {self.input_shape}")
            print(f"   输入数据类型: {self.input_dtype}")
            print(f"   输出名称: {self.output_name}")
            print(f"   输出形状: {self.output_shape}")
            print(f"   输出数据类型: {self.output_dtype}")
            return True
            
        except Exception as e:
            print(f"❌ TFLite模型加载失败: {e}")
            return False
    
    def preprocess(self, image_path):
        """TFLite预处理 (HWC格式) - MobileNetV2标准化"""
        img = Image.open(image_path).resize((224, 224))
        img = np.array(img).astype(np.float32)
        # MobileNetV2标准化: [-1, 1] 范围
        img = (img - 127.5) / 127.5
        img = np.expand_dims(img, axis=0)
        return img
    
    def inference(self, input_data):
        """TFLite推理"""
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output
    
    def postprocess(self, output_data):
        """TFLite后处理：压缩多余维度"""
        return np.squeeze(output_data)
    
    def cleanup(self):
        """TFLite清理资源：释放解释器"""
        if hasattr(self, 'interpreter') and self.interpreter is not None:
            # TensorFlow Lite解释器会自动管理内存
            self.interpreter = None

class ONNXBackend(InferenceBackend):
    """ONNX Runtime后端"""
    
    def load_model(self):
        try:
            import onnxruntime as ort
            
            print(f"加载ONNX模型: {self.model_path}")
            print("💻 使用CPU优化执行")
            
            # 仅使用CPU执行提供程序
            providers = ['CPUExecutionProvider']
            
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # 获取模型信息
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_shape = self.session.get_outputs()[0].shape
            self.input_dtype = self.session.get_inputs()[0].type
            self.output_dtype = self.session.get_outputs()[0].type
            
            print(f"✅ ONNX模型加载成功")
            print(f"   输入名称: {self.input_name}")
            print(f"   输入形状: {self.input_shape}")
            print(f"   输入数据类型: {self.input_dtype}")
            print(f"   输出名称: {self.output_name}")
            print(f"   输出形状: {self.output_shape}")
            print(f"   输出数据类型: {self.output_dtype}")
            print(f"   执行提供程序: {self.session.get_providers()}")
            return True
            
        except Exception as e:
            print(f"❌ ONNX模型加载失败: {e}")
            return False
    
    def preprocess(self, image_path):
        """ONNX预处理 (CHW格式) - MobileNetV2标准化"""
        img = Image.open(image_path).resize((224, 224))
        img = np.array(img).astype(np.float32)
        # MobileNetV2标准化: [-1, 1] 范围
        img = (img - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)
        return img
    
    def inference(self, input_data):
        """ONNX推理"""
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        return outputs[0]
    
    def postprocess(self, output_data):
        """ONNX后处理：压缩多余维度"""
        return np.squeeze(output_data)
    
    def cleanup(self):
        """ONNX清理资源：释放会话"""
        if hasattr(self, 'session') and self.session is not None:
            # ONNXRuntime会话会自动管理内存
            self.session = None

class TensorRTBackend(InferenceBackend):
    """原生TensorRT后端"""
    
    def load_model(self):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            print(f"加载TensorRT引擎: {self.model_path}")
            
            # 加载引擎
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            with open(self.model_path, "rb") as f:
                runtime = trt.Runtime(TRT_LOGGER)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            if self.engine is None:
                raise RuntimeError("引擎加载失败")
            
            self.context = self.engine.create_execution_context()
            
            # 获取绑定信息
            self.input_name = self.engine.get_tensor_name(0)
            self.output_name = self.engine.get_tensor_name(1)
            
            self.input_shape = self.engine.get_tensor_shape(self.input_name)
            self.output_shape = self.engine.get_tensor_shape(self.output_name)
            self.input_dtype = self.engine.get_tensor_dtype(self.input_name)
            self.output_dtype = self.engine.get_tensor_dtype(self.output_name)
            
            # 分配GPU内存
            input_size = int(np.prod(self.input_shape) * np.dtype(np.float32).itemsize)
            output_size = int(np.prod(self.output_shape) * np.dtype(np.float32).itemsize)
            
            self.d_input = cuda.mem_alloc(input_size)
            self.d_output = cuda.mem_alloc(output_size)
            
            print(f"✅ TensorRT引擎加载成功")
            print(f"   输入名称: {self.input_name}")
            print(f"   输入形状: {self.input_shape}")
            print(f"   输入数据类型: {self.input_dtype}")
            print(f"   输出名称: {self.output_name}")
            print(f"   输出形状: {self.output_shape}")
            print(f"   输出数据类型: {self.output_dtype}")
            print(f"   GPU内存: 输入{input_size//1024}KB, 输出{output_size//1024}KB")
            return True
            
        except Exception as e:
            print(f"❌ TensorRT引擎加载失败: {e}")
            return False
    
    def preprocess(self, image_path):
        """TensorRT预处理 (CHW格式) - MobileNetV2标准化"""
        img = Image.open(image_path).resize((224, 224))
        img = np.array(img).astype(np.float32)
        
        # MobileNetV2标准化: [-1, 1] 范围
        img = (img - 127.5) / 127.5
        
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)
        return img
    
    def inference(self, input_data):
        """TensorRT推理"""
        import pycuda.driver as cuda
        
        # 拷贝输入到GPU
        input_host = np.ascontiguousarray(input_data.ravel()).astype(np.float32)
        cuda.memcpy_htod(self.d_input, input_host)
        
        # 执行推理 (使用传统的binding方式)
        bindings = [int(self.d_input), int(self.d_output)]
        self.context.execute_v2(bindings)
        
        # 拷贝输出回CPU
        output_host = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output_host, self.d_output)
        
        return output_host
    
    def postprocess(self, output_data):
        """TensorRT后处理：压缩多余维度"""
        return np.squeeze(output_data)
    
    def cleanup(self):
        """TensorRT清理资源：释放GPU内存"""
        try:
            if hasattr(self, 'd_input') and self.d_input is not None:
                self.d_input.free()
                self.d_input = None
            
            if hasattr(self, 'd_output') and self.d_output is not None:
                self.d_output.free()
                self.d_output = None
            
            if hasattr(self, 'context') and self.context is not None:
                # TensorRT上下文会自动管理，但显式清理更安全
                self.context = None
            
            if hasattr(self, 'engine') and self.engine is not None:
                # 引擎会自动管理，但显式清理更安全
                self.engine = None
                
        except Exception as e:
            print(f"⚠️ TensorRT资源清理警告: {e}")

class NCNNBackend(InferenceBackend):
    """NCNN后端"""
    
    def load_model(self):
        try:
            import ncnn
            
            print(f"加载NCNN模型: {self.model_path}")
            
            # 检查是否是.param文件，如果是目录则寻找.param文件
            if os.path.isdir(self.model_path):
                param_files = [f for f in os.listdir(self.model_path) if f.endswith('.param')]
                if not param_files:
                    raise FileNotFoundError(f"目录中找不到.param文件: {self.model_path}")
                param_path = os.path.join(self.model_path, param_files[0])
                bin_path = param_path.replace('.param', '.bin')
            elif self.model_path.endswith('.param'):
                param_path = self.model_path
                bin_path = self.model_path.replace('.param', '.bin')
            else:
                # 假设传入的是前缀
                param_path = self.model_path + '.param'
                bin_path = self.model_path + '.bin'
            
            # 检查文件存在性
            if not os.path.exists(param_path):
                raise FileNotFoundError(f"参数文件不存在: {param_path}")
            if not os.path.exists(bin_path):
                raise FileNotFoundError(f"权重文件不存在: {bin_path}")
            
            # 创建网络
            self.net = ncnn.Net()
            
            # 启用优化选项
            self.net.opt.use_vulkan_compute = False  # CPU模式
            self.net.opt.use_fp16_packed = True      # 启用FP16优化
            self.net.opt.use_fp16_storage = True     # 启用FP16存储
            
            # 加载模型文件
            ret_param = self.net.load_param(param_path)
            ret_model = self.net.load_model(bin_path)
            
            if ret_param != 0:
                raise RuntimeError(f"参数文件加载失败: {ret_param}")
            if ret_model != 0:
                raise RuntimeError(f"模型文件加载失败: {ret_model}")
            
            # ✅ 通过API获取：输入输出名称
            input_names = self.net.input_names()
            output_names = self.net.output_names()
            
            self.input_name = input_names[0] if input_names else "input"
            self.output_name = output_names[0] if output_names else "output"
            
            # ✅ 合理假设：其它所有信息
            self.input_shape = (1, 3, 224, 224)  # MobileNetV2标准输入
            self.input_dtype = "float32"          # NCNN几乎总是float32
            self.output_shape = (1, 1001)        # MobileNetV2分类输出（包含背景类）
            self.output_dtype = "float32"         # NCNN几乎总是float32
            
            print(f"✅ NCNN模型加载成功")
            print(f"   输入名称: {self.input_name}")
            print(f"   输入形状: {self.input_shape}")
            print(f"   输入数据类型: {self.input_dtype}")
            print(f"   输出名称: {self.output_name}")
            print(f"   输出形状: {self.output_shape}")
            print(f"   输出数据类型: {self.output_dtype}")
            print(f"   获取方式: API名称 + 合理假设")
            print(f"   模型文件: {os.path.basename(param_path)}, {os.path.basename(bin_path)}")
            return True
            
        except ImportError:
            print("❌ NCNN Python绑定未安装")
            print("   请运行: pip install ncnn")
            return False
        except Exception as e:
            print(f"❌ NCNN模型加载失败: {e}")
            return False
    
    def preprocess(self, image_path):
        """NCNN预处理 (CHW格式) - MobileNetV2标准化"""
        from PIL import Image
        img = Image.open(image_path).convert('RGB').resize((224, 224))
        img = np.array(img).astype(np.float32)
        
        # MobileNetV2标准化: [-1, 1] 范围
        img = (img - 127.5) / 127.5
        
        # NCNN的ncnn.Mat(array)需要CHW格式
        # from_pixels接口会内部做这个转换，但直接用array需要手动转换
        img = np.transpose(img, (2, 0, 1))
        return img
    
    def inference(self, input_data):
        """NCNN推理 - 直接使用numpy预处理好的CHW数据"""
        try:
            import ncnn
            
            # input_data是预处理好的float32 CHW数据 [-1, 1]
            # 创建ncnn.Mat，确保数据是连续的
            input_data = np.ascontiguousarray(input_data)
            mat_in = ncnn.Mat(input_data)
            
            # 创建提取器
            ex = self.net.create_extractor()
            ex.input(self.input_name, mat_in)
            
            # 提取输出（使用动态获取的输出名称）
            mat_out = ncnn.Mat()
            ex.extract(self.output_name, mat_out)
            
            # 转换为 numpy 数组
            output = np.array(mat_out)
            return output
            
        except Exception as e:
            raise RuntimeError(f"NCNN推理失败: {e}")
    
    def postprocess(self, output_data):
        """NCNN后处理：压缩多余维度"""
        return np.squeeze(output_data)
    
    def cleanup(self):
        """NCNN清理资源：释放网络"""
        if hasattr(self, 'net') and self.net is not None:
            # NCNN网络会自动管理内存
            self.net = None

def benchmark_model(backend, image_path, test_runs=20):
    """性能测试"""
    print(f"\n=== 性能测试 ===")
    print(f"测试次数: {test_runs}")
    
    # 加载模型
    if not backend.load_model():
        return None, None
    
    try:
        # 预处理图像
        print("正在预处理图像...")
        preprocess_start = time.time()
        input_data = backend.preprocess(image_path)
        preprocess_time = time.time() - preprocess_start
        
        # 性能测试
        print("正在执行性能测试...")
        inference_times = []
        postprocess_time = None
        predictions = None
        
        for i in range(test_runs):
            try:
                start = time.time()
                output = backend.inference(input_data)
                inference_time = time.time() - start
                inference_times.append(inference_time)
                
                if i == 0:
                    postprocess_start = time.time()
                    predictions = backend.postprocess(output)
                    postprocess_time = time.time() - postprocess_start
                    print("首次推理完成")
                    
            except Exception as e:
                print(f"推理失败 (第{i+1}次): {e}")
                return None, None
                
    finally:
        # 清理资源
        backend.cleanup()
    
    # 计算统计数据
    avg_time = np.mean(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    std_time = np.std(inference_times)
    
    print(f"\n📊 性能统计:")
    print(f"   预处理耗时: {preprocess_time:.4f}秒")
    print(f"   平均推理耗时: {avg_time:.4f}秒 ({1/avg_time:.1f} FPS)")
    print(f"   最快推理耗时: {min_time:.4f}秒 ({1/min_time:.1f} FPS)")
    print(f"   最慢推理耗时: {max_time:.4f}秒 ({1/max_time:.1f} FPS)")
    print(f"   标准差: {std_time:.4f}秒")
    if postprocess_time is not None:
        print(f"   后处理耗时: {postprocess_time:.4f}秒")
    
    return avg_time, predictions

def show_predictions(predictions, labels_path=None, top_k=5):
    """显示预测结果"""
    if predictions is None:
        return
    
    # 加载标签
    try:
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                labels = f.read().splitlines()
        else:
            labels = [f"class_{i}" for i in range(len(predictions))]
    except:
        labels = [f"class_{i}" for i in range(len(predictions))]
    
    # 获取Top-K
    top_indices = predictions.argsort()[-top_k:][::-1]
    
    for i, idx in enumerate(top_indices):
        class_name = f"class_{idx}"
        real_label = labels[idx] if idx < len(labels) else f"unknown_{idx}"
        confidence = predictions[idx]
        print(f"   {i+1}. {class_name} ({real_label}): {confidence:.4f} ({confidence*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="推理性能测试工具")
    parser.add_argument("--tflite", help="TFLite模型路径")
    parser.add_argument("--onnx", help="ONNX模型路径")
    parser.add_argument("--tensorrt-fp32", help="TensorRT FP32引擎路径")
    parser.add_argument("--tensorrt-fp16", help="TensorRT FP16引擎路径") 
    parser.add_argument("--tensorrt-int8", help="TensorRT INT8引擎路径")
    parser.add_argument("--ncnn", help="NCNN模型路径 (.param文件或目录)")
    parser.add_argument("--ncnn-int8", help="NCNN INT8量化模型路径 (.param文件或目录)")
    parser.add_argument("--image", "-i", required=True, help="输入图像路径")
    parser.add_argument("--labels", help="标签文件路径")
    parser.add_argument("--runs", type=int, default=20, help="测试次数 (默认: 20)")
    parser.add_argument("--top-k", type=int, default=5, help="显示Top-K结果 (默认: 5)")
    parser.add_argument("--compare", action="store_true", help="对比所有可用模型")
    parser.add_argument("--ncnn-accuracy-warning", action="store_true", help="显示NCNN精度警告")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ 图像文件不存在: {args.image}")
        return 1
    
    print("=== 推理性能测试工具 ===")
    print(f"输入图像: {args.image}")
    
    results = {}
    all_predictions = {}
    
    # 测试TFLite
    if args.tflite and os.path.exists(args.tflite):
        print(f"\n🔵 === TensorFlow Lite 测试 ===")
        backend = TFLiteBackend(args.tflite)
        avg_time, predictions = benchmark_model(backend, args.image, args.runs)
        if avg_time is not None:
            results['TensorFlow Lite'] = avg_time
            all_predictions['TensorFlow Lite'] = predictions
    
    # 测试ONNX (CPU优化)
    if args.onnx and os.path.exists(args.onnx):
        print(f"\n🟠 === ONNX Runtime 测试 ===")
        backend = ONNXBackend(args.onnx)
        avg_time, predictions = benchmark_model(backend, args.image, args.runs)
        if avg_time is not None:
            results['ONNX (CPU)'] = avg_time
            all_predictions['ONNX (CPU)'] = predictions
    
    # 测试TensorRT各精度模式
    tensorrt_configs = [
        (args.tensorrt_fp32, "TensorRT (FP32)", "🔴"),
        (args.tensorrt_fp16, "TensorRT (FP16)", "🟠"), 
        (args.tensorrt_int8, "TensorRT (INT8)", "🟡")
    ]
    
    for trt_path, name, emoji in tensorrt_configs:
        if trt_path and os.path.exists(trt_path):
            print(f"\n{emoji} === {name} 测试 ===")
            backend = TensorRTBackend(trt_path)
            avg_time, predictions = benchmark_model(backend, args.image, args.runs)
            if avg_time is not None:
                results[name] = avg_time
                all_predictions[name] = predictions
    
    # 测试NCNN各模式
    ncnn_configs = [
        (args.ncnn, "NCNN", "🟢"),
        (args.ncnn_int8, "NCNN (INT8)", "🟫")
    ]
    
    for ncnn_path, name, emoji in ncnn_configs:
        if ncnn_path and os.path.exists(ncnn_path):
            print(f"\n{emoji} === {name} 测试 ===")
            backend = NCNNBackend(ncnn_path)
            avg_time, predictions = benchmark_model(backend, args.image, args.runs)
            if avg_time is not None:
                results[name] = avg_time
                all_predictions[name] = predictions
    
    # 显示对比结果
    if results:
        print(f"\n🏆 === 性能对比结果 ===")
        fastest_time = min(results.values())
        
        for backend, avg_time in sorted(results.items(), key=lambda x: x[1]):
            fps = 1 / avg_time
            if avg_time == fastest_time:
                print(f"{backend:20}: {avg_time:.4f}秒 ({fps:6.1f} FPS) 🥇 [最快]")
            else:
                speedup = avg_time / fastest_time
                print(f"{backend:20}: {avg_time:.4f}秒 ({fps:6.1f} FPS) [{speedup:.1f}x slower]")
        
        # 显示每个后端的预测结果
        print(f"\n🔍 === 各后端预测结果对比 ===")
        for backend_name, predictions in all_predictions.items():
            print(f"\n【{backend_name}】预测结果:")
            if 'NCNN' in backend_name and args.ncnn_accuracy_warning:
                print("⚠️  NCNN预测精度可能与其他框架存在差异，建议仅用于性能测试")
            show_predictions(predictions, args.labels, args.top_k)
    else:
        print("❌ 没有成功的测试结果")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())