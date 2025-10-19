#!/usr/bin/env python3
"""
02. 模型转换脚本：TFLite -> ONNX -> TensorRT/MNN/NCNN
第二步执行：转换模型格式以支持不同推理后端
运行前确保: python 01_check_deps.py 成功
"""

import argparse
import json
import os
import time
import glob
import shutil
import tempfile
import subprocess
import numpy as np
from PIL import Image

def get_calibration_images(dataset_path, max_images=100):
    """
    获取校准图像列表
    
    Args:
        dataset_path (str): 数据集路径（支持以下格式）:
                           - 单个图片文件: path/to/image.jpg
                           - 图片目录: path/to/images/
                           - 图片列表文件: path/to/imagelist.txt
        max_images (int): 最大图片数量，避免校准时间过长
    
    Returns:
        list: 图片路径列表
    """
    if not os.path.exists(dataset_path):
        print(f"❌ 错误: 校准数据集路径不存在: {dataset_path}")
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_paths = []
    
    if os.path.isfile(dataset_path):
        # 检查是否是图片列表文件
        if dataset_path.lower().endswith('.txt'):
            print(f"📝 解析图片列表文件: {dataset_path}")
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue  # 跳过空行和注释行
                    
                    # 支持相对路径（相对于列表文件所在目录）
                    if not os.path.isabs(line):
                        base_dir = os.path.dirname(dataset_path)
                        line = os.path.join(base_dir, line)
                    
                    if os.path.exists(line):
                        if any(line.lower().endswith(ext) for ext in image_extensions):
                            image_paths.append(line)
                        else:
                            print(f"⚠️  跳过非图片文件 (第{line_num}行): {os.path.basename(line)}")
                    else:
                        print(f"⚠️  跳过不存在的文件 (第{line_num}行): {line}")
                        
            except Exception as e:
                print(f"❌ 错误: 无法解析图片列表文件: {e}")
                return []
                
        # 如果是单个图片文件
        elif any(dataset_path.lower().endswith(ext) for ext in image_extensions):
            image_paths = [dataset_path]
        else:
            print(f"❌ 错误: 文件不是支持的格式（图片或.txt列表）: {dataset_path}")
            return []
            
    elif os.path.isdir(dataset_path):
        # 如果是目录，扫描所有图片
        print(f"📁 扫描校准数据集目录: {dataset_path}")
        for ext in image_extensions:
            pattern = os.path.join(dataset_path, f"*{ext}")
            image_paths.extend(glob.glob(pattern))
            pattern = os.path.join(dataset_path, f"*{ext.upper()}")
            image_paths.extend(glob.glob(pattern))
        
        # 按文件名排序保证一致性
        image_paths.sort()
        
        # 限制图片数量
        if len(image_paths) > max_images:
            print(f"⚠️  发现 {len(image_paths)} 张图片，限制使用前 {max_images} 张")
            image_paths = image_paths[:max_images]
    
    if not image_paths:
        print(f"❌ 错误: 在 {dataset_path} 中未找到图片文件")
        return []
    
    print(f"📷 找到 {len(image_paths)} 张校准图片")
    if len(image_paths) <= 5:
        for path in image_paths:
            print(f"   - {os.path.basename(path)}")
    else:
        for i, path in enumerate(image_paths[:3]):
            print(f"   - {os.path.basename(path)}")
        print(f"   ... 还有 {len(image_paths) - 3} 张图片")
    
    return image_paths

def convert_tflite_to_onnx(tflite_path, onnx_path):
    """
    将TFLite模型转换为ONNX格式
    
    Args:
        tflite_path (str): TFLite模型路径
        onnx_path (str): 输出ONNX模型路径
    
    Returns:
        bool: 转换是否成功
    """
    print(f"=== TFLite -> ONNX 转换 ===")
    print(f"输入文件: {tflite_path}")
    print(f"输出文件: {onnx_path}")
    
    if not os.path.exists(tflite_path):
        print(f"❌ 错误: TFLite文件不存在: {tflite_path}")
        return False
    
    try:
        import tflite2onnx
        
        print("正在转换...")
        start_time = time.time()
        tflite2onnx.convert(tflite_path, onnx_path)
        convert_time = time.time() - start_time
        
        if os.path.exists(onnx_path):
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            print(f"✅ 转换成功!")
            print(f"   转换耗时: {convert_time:.2f}秒")
            print(f"   输出文件大小: {file_size:.1f}MB")
            return True
        else:
            print("❌ 转换失败: 输出文件未生成")
            return False
            
    except ImportError:
        print("❌ 错误: tflite2onnx 未安装")
        print("请运行: pip install tflite2onnx tflite")
        return False
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False

# 添加TensorRT校准器基类支持
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    class Int8TRTCalibrator(trt.IInt8EntropyCalibrator2):
        """TensorRT INT8 校准器 - 支持多张图片"""
        
        def __init__(self, calibration_dataset_path, cache_file="calibration.cache", max_images=50):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self.calibration_dataset_path = calibration_dataset_path
            self.cache_file = cache_file
            self.batch_size = 1
            self.current_index = 0
            self.max_images = max_images
            
            # 获取校准图像列表
            self.image_paths = get_calibration_images(calibration_dataset_path, max_images)
            if not self.image_paths:
                raise ValueError(f"未找到校准图像: {calibration_dataset_path}")
            
            print(f"📊 TensorRT校准将使用 {len(self.image_paths)} 张图片")
            
            # 预处理所有校准图像
            self.calibration_data_list = self._preprocess_calibration_images()
            
            # 分配GPU内存 (基于第一张图片的大小)
            if self.calibration_data_list:
                self.device_input = cuda.mem_alloc(self.calibration_data_list[0].nbytes)
            
        def _preprocess_calibration_images(self):
            """预处理多张校准图像 (CHW格式，MobileNetV2标准化)"""
            print(f"📷 正在预处理 {len(self.image_paths)} 张校准图像...")
            
            calibration_data_list = []
            
            for i, image_path in enumerate(self.image_paths):
                try:
                    img = Image.open(image_path).convert('RGB').resize((224, 224))
                    img = np.array(img).astype(np.float32)
                    
                    # 使用MobileNetV2标准化 (保持一致性)
                    img = (img - 127.5) / 127.5  # [-1, 1] 范围
                    
                    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                    img = np.expand_dims(img, axis=0)  # 添加batch维度
                    
                    calibration_data = np.ascontiguousarray(img, dtype=np.float32)
                    calibration_data_list.append(calibration_data)
                    
                    if (i + 1) % 10 == 0 or i == 0:
                        print(f"   已处理: {i + 1}/{len(self.image_paths)}")
                        
                except Exception as e:
                    print(f"⚠️  跳过损坏的图片 {image_path}: {e}")
                    continue
            
            if not calibration_data_list:
                raise ValueError("所有校准图像都无法处理")
                
            print(f"✅ 成功预处理 {len(calibration_data_list)} 张校准图像")
            return calibration_data_list
        
        def get_batch_size(self):
            return self.batch_size
        
        def get_batch(self, names):
            if self.current_index < len(self.calibration_data_list):
                # 将当前图像数据拷贝到GPU
                current_data = self.calibration_data_list[self.current_index]
                cuda.memcpy_htod(self.device_input, current_data)
                
                if (self.current_index + 1) % 10 == 0 or self.current_index == 0:
                    print(f"   校准进度: {self.current_index + 1}/{len(self.calibration_data_list)}")
                    
                self.current_index += 1
                return [int(self.device_input)]
            else:
                print(f"✅ TensorRT校准完成，共使用 {len(self.calibration_data_list)} 张图片")
                return None
        
        def read_calibration_cache(self):
            # 读取缓存文件
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None
        
        def write_calibration_cache(self, cache):
            # 写入缓存文件
            with open(self.cache_file, "wb") as f:
                f.write(cache)
            print(f"✅ 校准缓存已保存: {self.cache_file}")

except ImportError:
    # 如果TensorRT未安装，创建虚拟类
    class Int8TRTCalibrator:
        def __init__(self, *args, **kwargs):
            pass

def convert_onnx_to_tensorrt(onnx_path, trt_path, max_workspace_size=1, precision="fp32", calibration_dataset=None):
    """
    将ONNX模型转换为TensorRT引擎
    
    Args:
        onnx_path (str): ONNX模型路径
        trt_path (str): 输出TensorRT引擎路径
        max_workspace_size (int): 最大工作空间大小(GB)
        precision (str): 精度模式 ("fp32", "fp16", "int8")
        calibration_dataset (str): INT8校准数据集路径（图片文件或目录）
    
    Returns:
        bool: 转换是否成功
    """
    print(f"\n=== ONNX -> TensorRT 转换 ===")
    print(f"输入文件: {onnx_path}")
    print(f"输出文件: {trt_path}")
    print(f"工作空间: {max_workspace_size}GB")
    print(f"精度模式: {precision.upper()}")
    
    if not os.path.exists(onnx_path):
        print(f"❌ 错误: ONNX文件不存在: {onnx_path}")
        return False
    
    try:
        import tensorrt as trt
        
        print("正在创建TensorRT引擎...")
        start_time = time.time()
        
        # 创建builder和配置
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        config = builder.create_builder_config()
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        # 解析ONNX
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        print("正在解析ONNX模型...")
        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                print("❌ ONNX解析失败:")
                for error in range(parser.num_errors):
                    print(f"  {parser.get_error(error)}")
                return False
        
        # 配置builder
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size << 30)
        
        # 设置精度模式
        if precision.lower() == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
            print("✅ 启用FP16混合精度")
        elif precision.lower() == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            print("✅ 启用INT8量化")
            
            if calibration_dataset:
                # 使用提供的校准数据集
                cache_file = os.path.join(os.path.dirname(trt_path), "calibration.cache")
                try:
                    calibrator = Int8TRTCalibrator(calibration_dataset, cache_file)
                    config.int8_calibrator = calibrator
                    print(f"📊 使用校准数据集: {calibration_dataset}")
                except ValueError as e:
                    print(f"❌ 校准数据集错误: {e}")
                    return False
            else:
                print("⚠️  警告：INT8量化未提供校准数据，可能精度下降")
                return False
        else:
            print("✅ 使用FP32精度（默认）")
        
        print("正在构建TensorRT引擎（需要几分钟）...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            print("❌ 引擎构建失败")
            return False
        
        # 保存引擎
        with open(trt_path, "wb") as f:
            f.write(serialized_engine)
        
        convert_time = time.time() - start_time
        file_size = os.path.getsize(trt_path) / (1024 * 1024)  # MB
        
        print(f"✅ TensorRT引擎创建成功!")
        print(f"   构建耗时: {convert_time:.2f}秒")
        print(f"   引擎文件大小: {file_size:.1f}MB")
        return True
        
    except ImportError:
        print("❌ 错误: TensorRT 未安装")
        print("请运行: pip install tensorrt pycuda")
        return False
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False

def convert_onnx_to_ncnn(onnx_path, ncnn_param_path, ncnn_bin_path):
    """
    将ONNX模型转换为NCNN格式
    
    Args:
        onnx_path (str): ONNX模型路径
        ncnn_param_path (str): 输出NCNN参数文件路径
        ncnn_bin_path (str): 输出NCNN权重文件路径
    
    Returns:
        bool: 转换是否成功
    """
    print(f"\n=== ONNX -> NCNN 转换 ===")
    print(f"输入文件: {onnx_path}")
    print(f"参数文件: {ncnn_param_path}")
    print(f"权重文件: {ncnn_bin_path}")
    
    if not os.path.exists(onnx_path):
        print(f"❌ 错误: ONNX文件不存在: {onnx_path}")
        return False
    
    try:
        import subprocess
        
        print("正在转换... (onnx2ncnn)")
        start_time = time.time()
        
        # 执行转换
        cmd = ['onnx2ncnn', onnx_path, ncnn_param_path, ncnn_bin_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        convert_time = time.time() - start_time
        
        if result.returncode == 0:
            if os.path.exists(ncnn_param_path) and os.path.exists(ncnn_bin_path):
                param_size = os.path.getsize(ncnn_param_path) / 1024  # KB
                bin_size = os.path.getsize(ncnn_bin_path) / (1024 * 1024)  # MB
                
                print(f"✅ NCNN转换成功!")
                print(f"   转换耗时: {convert_time:.2f}秒")
                print(f"   参数文件大小: {param_size:.1f}KB")
                print(f"   权重文件大小: {bin_size:.1f}MB")
                return True
            else:
                print("❌ 转换失败: 输出文件未生成")
                return False
        else:
            print(f"❌ 转换失败:")
            if result.stderr:
                print(f"错误信息: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 转换超时")
        return False
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False

def quantize_ncnn_to_int8(ncnn_param_path, ncnn_bin_path, int8_param_path, int8_bin_path, calibration_dataset_path):
    """
    将NCNN模型量化为INT8格式
    
    Args:
        ncnn_param_path (str): NCNN参数文件路径
        ncnn_bin_path (str): NCNN权重文件路径
        int8_param_path (str): 输出INT8参数文件路径
        int8_bin_path (str): 输出INT8权重文件路径
        calibration_dataset_path (str): 校准数据集路径（图片文件或目录）
    
    Returns:
        bool: 量化是否成功
    """
    print(f"\n=== NCNN INT8 量化 ===")
    print(f"输入参数文件: {ncnn_param_path}")
    print(f"输入权重文件: {ncnn_bin_path}")
    print(f"输出参数文件: {int8_param_path}")
    print(f"输出权重文件: {int8_bin_path}")
    print(f"校准数据集: {calibration_dataset_path}")
    
    if not os.path.exists(ncnn_param_path):
        print(f"❌ 错误: NCNN参数文件不存在: {ncnn_param_path}")
        return False
    
    if not os.path.exists(ncnn_bin_path):
        print(f"❌ 错误: NCNN权重文件不存在: {ncnn_bin_path}")
        return False
        
    # 获取校准图像列表
    calibration_images = get_calibration_images(calibration_dataset_path, max_images=50)
    if not calibration_images:
        print(f"❌ 错误: 无法获取校准图像: {calibration_dataset_path}")
        return False
    
    print(f"📊 NCNN量化将使用 {len(calibration_images)} 张图片")
    
    try:
        import subprocess
        import time
        
        print("正在进行NCNN INT8量化...")
        start_time = time.time()
        
        # 1. 先优化模型
        print("🔄 步骤1: 模型优化 (ncnnoptimize)")
        opt_param_path = ncnn_param_path.replace('.param', '-opt.param')
        opt_bin_path = ncnn_bin_path.replace('.bin', '-opt.bin')
        
        cmd = ['ncnnoptimize', ncnn_param_path, ncnn_bin_path, opt_param_path, opt_bin_path, '0']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"❌ 模型优化失败: {result.stderr}")
            return False
        print("✅ 模型优化完成")
        
        # 2. 生成校准表
        print("🔄 步骤2: 生成校准表 (ncnn2table)")
        table_path = ncnn_param_path.replace('.param', '.table')
        imagelist_path = os.path.join(os.path.dirname(ncnn_param_path), 'imagelist_ncnn.txt')
        
        # 创建图像列表文件（包含多张图片）
        print(f"📝 创建图像列表文件: {imagelist_path}")
        with open(imagelist_path, 'w') as f:
            for img_path in calibration_images:
                f.write(img_path + '\n')
        
        cmd = [
            'ncnn2table', opt_param_path, opt_bin_path, imagelist_path, table_path,
            'mean=[127.5,127.5,127.5]', 'norm=[0.007843137,0.007843137,0.007843137]',
            'shape=[224,224,3]', 'pixel=RGB', 'thread=4', 'method=kl'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"❌ 校准表生成失败: {result.stderr}")
            return False
        print("✅ 校准表生成完成")
        
        # 3. 转换INT8模型
        print("🔄 步骤3: INT8转换 (ncnn2int8)")
        cmd = ['ncnn2int8', opt_param_path, opt_bin_path, int8_param_path, int8_bin_path, table_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"❌ INT8转换失败: {result.stderr}")
            return False
        
        # 检查输出文件
        if os.path.exists(int8_param_path) and os.path.exists(int8_bin_path):
            convert_time = time.time() - start_time
            param_size = os.path.getsize(int8_param_path) / 1024  # KB
            bin_size = os.path.getsize(int8_bin_path) / (1024 * 1024)  # MB
            original_bin_size = os.path.getsize(ncnn_bin_path) / (1024 * 1024)  # MB
            compression_ratio = (1 - bin_size / original_bin_size) * 100
            
            print(f"✅ NCNN INT8量化成功!")
            print(f"   转换耗时: {convert_time:.2f}秒")
            print(f"   参数文件大小: {param_size:.1f}KB")
            print(f"   权重文件大小: {bin_size:.1f}MB")
            print(f"   压缩率: {compression_ratio:.1f}% (原始: {original_bin_size:.1f}MB)")
            
            # 清理中间文件
            for temp_file in [opt_param_path, opt_bin_path, table_path, imagelist_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            return True
        else:
            print("❌ 量化失败: 输出文件未生成")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 量化超时")
        return False
    except Exception as e:
        print(f"❌ 量化失败: {e}")
        return False

def convert_onnx_to_mnn(onnx_path, mnn_path):
    """
    将ONNX模型转换为MNN格式
    
    Args:
        onnx_path (str): ONNX模型路径
        mnn_path (str): 输出MNN模型路径
    
    Returns:
        bool: 转换是否成功
    """
    print(f"\n=== ONNX -> MNN 转换 ===")
    print(f"输入文件: {onnx_path}")
    print(f"输出文件: {mnn_path}")
    
    if not os.path.exists(onnx_path):
        print(f"❌ 错误: ONNX文件不存在: {onnx_path}")
        return False
    
    try:
        cmd = [
            "MNNConvert",
            "-f", "ONNX",
            "--modelFile", onnx_path,
            "--MNNModel", mnn_path,
            "--bizCode", "MNN"
        ]
        print("正在转换... (MNNConvert)")
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        convert_time = time.time() - start_time
        
        if result.returncode == 0:
            if os.path.exists(mnn_path):
                file_size = os.path.getsize(mnn_path) / (1024 * 1024)
                print(f"✅ MNN转换成功!")
                print(f"   转换耗时: {convert_time:.2f}秒")
                print(f"   模型大小: {file_size:.1f}MB")
                return True
            print("❌ 转换失败: 输出文件未生成")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            return False
        else:
            print(f"❌ 转换失败 (返回码 {result.returncode})")
            if result.stdout:
                print(f"输出信息: {result.stdout.strip()}")
            if result.stderr:
                print(f"错误信息: {result.stderr.strip()}")
            return False
        
    except subprocess.TimeoutExpired:
        print("❌ 转换超时")
        return False
    except FileNotFoundError:
        print("❌ 错误: 未检测到MNNConvert工具 (请确认已配置PATH)")
        return False
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False

def quantize_mnn_to_int8(mnn_path, int8_mnn_path, calibration_dataset_path):
    """
    使用quantized.out/MNNQuantTool对MNN模型进行INT8量化
    
    Args:
        mnn_path (str): FP32 MNN模型路径
        int8_mnn_path (str): 输出INT8 MNN模型路径
        calibration_dataset_path (str): 校准数据集路径
    
    Returns:
        bool: 量化是否成功
    """
    print(f"\n=== MNN INT8 量化 ===")
    print(f"输入模型: {mnn_path}")
    print(f"输出模型: {int8_mnn_path}")
    print(f"校准数据集: {calibration_dataset_path}")
    
    if not os.path.exists(mnn_path):
        print(f"❌ 错误: MNN模型不存在: {mnn_path}")
        return False
    
    image_paths = get_calibration_images(calibration_dataset_path, max_images=100)
    if not image_paths:
        print("❌ 错误: 未找到用于MNN量化的校准图像")
        return False
    print(f"📊 MNN量化将使用 {len(image_paths)} 张图片")
    
    temp_dir = None
    config_path = None
    try:
        # 步骤1: 复制校准图片到临时目录
        temp_dir = tempfile.mkdtemp(prefix="mnn_calib_")
        print("🔄 步骤1: 准备校准图片 (复制到临时目录)")
        for idx, img_path in enumerate(image_paths):
            ext = os.path.splitext(img_path)[1].lower()
            if not ext:
                ext = ".jpg"
            target_path = os.path.join(temp_dir, f"image_{idx:04d}{ext}")
            shutil.copy(img_path, target_path)
        
        # 步骤2: 生成量化配置文件
        config = {
            "format": "RGB",
            "mean": [127.5, 127.5, 127.5],
            "normal": [0.00784314, 0.00784314, 0.00784314],
            "center_crop_h": 1.0,
            "center_crop_w": 1.0,
            "width": 224,
            "height": 224,
            "path": temp_dir,
            "used_sample_num": len(image_paths),
            "feature_quantize_method": "EMA",
            "weight_quantize_method": "MAX_ABS",
            "feature_clamp_value": 127,
            "weight_clamp_value": 127,
            "batch_size": min(16, len(image_paths)),
            "quant_bits": 8,
            "skip_quant_op_names": [],
            "input_type": "image",
            "debug": False
        }
        config_path = os.path.join(temp_dir, "imageInputConfig.json")
        print(f"🔄 步骤2: 生成量化配置 ({config_path})")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        
        # 步骤3: 执行量化
        quant_tools = ["quantized.out", "MNNQuantTool"]
        last_not_found = []
        for tool in quant_tools:
            cmd = [tool, mnn_path, int8_mnn_path, config_path]
            print(f"🔄 步骤3: 执行量化 ({tool})")
            start = time.time()
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            except FileNotFoundError:
                last_not_found.append(tool)
                continue
            quant_time = time.time() - start
            
            if result.returncode != 0:
                print("❌ 量化失败")
                if result.stdout:
                    print(f"输出信息: {result.stdout.strip()}")
                if result.stderr:
                    print(f"错误信息: {result.stderr.strip()}")
                return False
            
            if os.path.exists(int8_mnn_path):
                file_size = os.path.getsize(int8_mnn_path) / (1024 * 1024)
                original_size = os.path.getsize(mnn_path) / (1024 * 1024)
                compression = (1 - file_size / original_size) * 100 if original_size > 0 else 0
                print(f"✅ MNN INT8量化成功!")
                print(f"   量化耗时: {quant_time:.2f}秒")
                print(f"   模型大小: {file_size:.1f}MB (原始 {original_size:.1f}MB, 压缩 {compression:.1f}%)")
                
                # 清理调试用JSON（quantized.out默认生成）
                extra_json = int8_mnn_path + ".json"
                if os.path.exists(extra_json):
                    os.remove(extra_json)
                
                return True
            print("❌ 量化失败: 未生成量化模型文件")
            if result.stdout:
                print(f"输出信息: {result.stdout.strip()}")
            if result.stderr:
                print(f"错误信息: {result.stderr.strip()}")
            return False
        
        if last_not_found:
            print("❌ 错误: 未检测到MNN量化工具 (quantized.out / MNNQuantTool)")
        return False
        
    except subprocess.TimeoutExpired:
        print("❌ 量化超时")
        return False
    except Exception as e:
        print(f"❌ 量化失败: {e}")
        return False
    finally:
        if config_path and os.path.exists(config_path):
            try:
                os.remove(config_path)
            except:
                pass
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="模型转换工具: TFLite -> ONNX -> TensorRT/NCNN/MNN")
    parser.add_argument("--tflite", "-t", required=True, help="输入TFLite模型路径")
    parser.add_argument("--onnx", action="store_true", help="生成ONNX模型")
    parser.add_argument("--tensorrt-fp32", action="store_true", help="生成TensorRT FP32引擎")
    parser.add_argument("--tensorrt-fp16", action="store_true", help="生成TensorRT FP16引擎")
    parser.add_argument("--tensorrt-int8", action="store_true", help="生成TensorRT INT8引擎")
    parser.add_argument("--ncnn", "-n", action="store_true", help="生成NCNN模型")
    parser.add_argument("--ncnn-int8", action="store_true", help="生成NCNN INT8量化模型")
    parser.add_argument("--mnn", action="store_true", help="生成MNN模型")
    parser.add_argument("--mnn-int8", action="store_true", help="生成MNN INT8量化模型")
    parser.add_argument("--calibration-dataset", "-c", help="INT8量化校准数据集路径（支持：单个图片文件、图片目录、imagelist.txt文件）")
    
    args = parser.parse_args()
    
    # 生成默认输出路径
    base_name = os.path.splitext(args.tflite)[0]
    onnx_path = f"{base_name}.onnx"
    ncnn_param_path = None
    ncnn_bin_path = None
    mnn_path = f"{base_name}.mnn"
    
    # 需求标记
    need_trt = args.tensorrt_fp32 or args.tensorrt_fp16 or args.tensorrt_int8
    need_ncnn = args.ncnn or args.ncnn_int8
    need_mnn = args.mnn or args.mnn_int8
    
    # 检查INT8模式是否提供校准数据集
    if (args.tensorrt_int8 and not args.calibration_dataset) or \
       (args.ncnn_int8 and not args.calibration_dataset) or \
       (args.mnn_int8 and not args.calibration_dataset):
        print("❌ 错误: INT8量化需要校准数据集")
        print("请使用 --calibration-dataset 参数指定校准数据集路径（图片文件或目录）")
        return 1
    
    print("=== 模型转换工具 ===")
    print(f"输入: {args.tflite}")
    if args.onnx: print(f"输出ONNX: {onnx_path}")
    if args.tensorrt_fp32: print(f"输出TensorRT FP32: {base_name}_fp32.trt")
    if args.tensorrt_fp16: print(f"输出TensorRT FP16: {base_name}_fp16.trt")
    if args.tensorrt_int8: print(f"输出TensorRT INT8: {base_name}_int8.trt")
    if need_ncnn: print(f"输出NCNN: {base_name}.param/.bin")
    if args.ncnn_int8: print(f"输出NCNN INT8: {base_name}-int8.param/.bin")
    if need_mnn: print(f"输出MNN: {mnn_path}")
    if args.mnn_int8: print(f"输出MNN INT8: {base_name}_int8.mnn")
    
    
    # 步骤1: TFLite -> ONNX
    if need_trt or need_ncnn or need_mnn or args.onnx:
        print("\n🔄 步骤1: 转换 TFLite -> ONNX")
        if convert_tflite_to_onnx(args.tflite, onnx_path):
            print(f"   ✅ ONNX: {onnx_path}")
        else:
            print(f"   ❌ ONNX 生成失败")
            return 1
    
    # 步骤2: ONNX -> TensorRT
    tensorrt_modes = []
    if args.tensorrt_fp32:
        tensorrt_modes.append(("fp32", f"{base_name}_fp32.trt"))
    if args.tensorrt_fp16:
        tensorrt_modes.append(("fp16", f"{base_name}_fp16.trt"))
    if args.tensorrt_int8:
        tensorrt_modes.append(("int8", f"{base_name}_int8.trt"))
    
    for precision, trt_path in tensorrt_modes:
        print(f"\n🔄 步骤2: 转换 ONNX -> TensorRT ({precision.upper()})")
        calibration_data = args.calibration_dataset if precision == "int8" else None
        if convert_onnx_to_tensorrt(onnx_path, trt_path, 1, precision, calibration_data):
            print(f"   ✅ TensorRT {precision.upper()}: {trt_path}")
        else:
            print(f"   ❌ TensorRT {precision.upper()} 生成失败")
            return 1
    
    # 步骤3: ONNX -> NCNN
    if need_ncnn:
        print("\n🔄 步骤3: 转换 ONNX -> NCNN")
        ncnn_param_path = f"{base_name}.param"
        ncnn_bin_path = f"{base_name}.bin"
        if convert_onnx_to_ncnn(onnx_path, ncnn_param_path, ncnn_bin_path):
            print(f"   ✅ NCNN: {ncnn_param_path}, {ncnn_bin_path}")
        else:
            print(f"   ❌ NCNN 生成失败")
            return 1

    # 步骤4: NCNN INT8 量化
    if args.ncnn_int8:
        if not ncnn_param_path or not ncnn_bin_path:
            print("❌ NCNN INT8 量化失败: 未生成NCNN基础模型")
            return 1
        print("\n🔄 步骤4: NCNN INT8 量化")
        int8_param_path = f"{base_name}-int8.param"
        int8_bin_path = f"{base_name}-int8.bin"
        if quantize_ncnn_to_int8(ncnn_param_path, ncnn_bin_path, int8_param_path, int8_bin_path, args.calibration_dataset):
            print(f"   ✅ NCNN INT8: {int8_param_path}, {int8_bin_path}")
        else:
            print(f"   ❌ NCNN INT8 量化失败")
            return 1
    
    # 步骤5: ONNX -> MNN
    if need_mnn:
        print("\n🔄 步骤5: 转换 ONNX -> MNN")
        if convert_onnx_to_mnn(onnx_path, mnn_path):
            print(f"   ✅ MNN: {mnn_path}")
        else:
            print(f"   ❌ MNN 生成失败")
            return 1
    
    # 步骤6: MNN INT8 量化
    if args.mnn_int8:
        print("\n🔄 步骤6: MNN INT8 量化")
        int8_mnn_path = f"{base_name}_int8.mnn"
        if quantize_mnn_to_int8(mnn_path, int8_mnn_path, args.calibration_dataset):
            print(f"   ✅ MNN INT8: {int8_mnn_path}")
        else:
            print(f"   ❌ MNN INT8 量化失败")
            return 1
    
    # 检查是否指定了任何输出格式
    if not args.onnx and not args.tensorrt_fp32 and not args.tensorrt_fp16 and not args.tensorrt_int8 and \
       not args.ncnn and not args.ncnn_int8 and not args.mnn and not args.mnn_int8:
        print("\n错误: 请指定输出格式: --onnx, --tensorrt-fp32, --tensorrt-fp16, --tensorrt-int8, --ncnn, --ncnn-int8, --mnn, --mnn-int8")
        return 1
    
    print("\n🎉 转换完成!")
    
    return 0

if __name__ == "__main__":
    exit(main())
