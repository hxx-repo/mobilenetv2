#!/usr/bin/env python3
"""
环境依赖检查脚本
用于验证推理环境是否正确配置
"""

def check_dependencies():
    """检查依赖包是否安装"""
    print("=== 检查依赖环境 ===")
    
    missing = []
    
    # 1. numpy检查 - 验证版本兼容性
    try:
        import numpy as np
        version = np.__version__
        print(f"✅ numpy ({version}) - 数值计算")
        if tuple(map(int, version.split('.')[:2])) < (1, 20):
            print(f"    ⚠️  建议升级到1.20+版本，当前{version}可能存在兼容问题")
    except ImportError:
        print(f"❌ numpy - 数值计算 (未安装)")
        missing.append("numpy")
    except Exception as e:
        print(f"⚠️  numpy - 数值计算 (版本检查失败: {e})")
    
    # 2. PIL(Pillow)检查 - 验证图像处理功能
    try:
        from PIL import Image
        import PIL
        version = PIL.__version__
        print(f"✅ PIL/Pillow ({version}) - 图像处理")
        # 测试基本图像功能
        test_img = Image.new('RGB', (10, 10), color='red')
        test_img.resize((5, 5))
    except ImportError:
        print(f"❌ PIL - 图像处理(Pillow) (未安装)")
        missing.append("PIL")
    except Exception as e:
        print(f"⚠️  PIL - 图像处理(Pillow) (功能测试失败: {e})")
    
    # 3. tensorflow检查 - 验证TFLite模块
    try:
        import tensorflow as tf
        version = tf.__version__
        print(f"✅ tensorflow ({version}) - TensorFlow Lite")
        # 验证TFLite模块
        from tensorflow.lite.python.interpreter import Interpreter
        print(f"    ✅ TensorFlow Lite解释器可用")
    except ImportError as e:
        if "tensorflow" in str(e).lower():
            print(f"❌ tensorflow - TensorFlow Lite (未安装)")
            missing.append("tensorflow")
        else:
            print(f"⚠️  tensorflow - TFLite解释器模块缺失: {e}")
    except Exception as e:
        print(f"⚠️  tensorflow - TensorFlow Lite (模块验证失败: {e})")
    
    # 4. onnx检查 - 验证模型格式支持
    try:
        import onnx
        version = onnx.__version__
        print(f"✅ onnx ({version}) - ONNX格式支持")
        # 验证基本功能
        onnx.checker.check_model
    except ImportError:
        print(f"❌ onnx - ONNX格式支持 (未安装)")
        missing.append("onnx")
    except Exception as e:
        print(f"⚠️  onnx - ONNX格式支持 (功能验证失败: {e})")
    
    # 5. onnxruntime检查 - 验证推理后端
    try:
        import onnxruntime as ort
        version = ort.__version__
        print(f"✅ onnxruntime ({version}) - ONNX Runtime")
        # 检查可用的execution providers
        providers = ort.get_available_providers()
        cpu_available = 'CPUExecutionProvider' in providers
        if cpu_available:
            print(f"    ✅ CPU推理后端可用")
        else:
            print(f"    ⚠️  CPU推理后端不可用")
    except ImportError:
        print(f"❌ onnxruntime - ONNX Runtime (未安装)")
        missing.append("onnxruntime")
    except Exception as e:
        print(f"⚠️  onnxruntime - ONNX Runtime (功能验证失败: {e})")
    
    # 6. tflite2onnx检查 - 验证模型转换工具
    try:
        import tflite2onnx
        print(f"✅ tflite2onnx - TFLite转ONNX")
        # 验证核心转换功能
        hasattr(tflite2onnx, 'convert')
    except ImportError:
        print(f"❌ tflite2onnx - TFLite转ONNX (未安装)")
        missing.append("tflite2onnx")
    except Exception as e:
        print(f"⚠️  tflite2onnx - TFLite转ONNX (功能验证失败: {e})")
    
    # 7. TensorRT完整工具链检查 - 验证GPU推理后端
    print(f"\n=== TensorRT工具链检查 ===")
    import subprocess
    import os
    
    # 1) 检查CUDA环境 - TensorRT的基础依赖
    print("🔍 检查CUDA环境:")
    tensorrt_issues = []
    try:
        import pycuda
        import pycuda.driver as cuda
        cuda.init()
        
        # 创建CUDA上下文
        device = cuda.Device(0)
        context = device.make_context()
        
        try:
            # GPU信息
            gpu_name = device.name()
            free, total = cuda.mem_get_info()
            print(f"   ✅ GPU: {gpu_name}")
            print(f"       GPU内存: {free//1024//1024}MB / {total//1024//1024}MB")
        finally:
            # 清理上下文
            context.pop()
        
        # CUDA版本
        cuda_runtime_version = cuda.get_version()
        driver_version = cuda.get_driver_version()
        major = driver_version // 1000
        minor = (driver_version % 1000) // 10
        print(f"   ✅ CUDA Runtime: {cuda_runtime_version[0]}.{cuda_runtime_version[1]}")
        print(f"   ✅ CUDA Driver: {major}.{minor}")
        
        # 兼容性检查
        if cuda_runtime_version[0] >= 12:
            print(f"       ✅ CUDA版本兼容 (支持TensorRT 8.6)")
        else:
            print(f"       ⚠️  CUDA版本可能过低，建议升级到12.x")
            tensorrt_issues.append("CUDA版本过低")
            
    except ImportError:
        print(f"   ❌ pycuda - CUDA Python绑定 (未安装)")
        missing.append("pycuda")
        tensorrt_issues.append("pycuda未安装")
    except Exception as e:
        print(f"   ⚠️  CUDA环境检查失败: {e}")
        tensorrt_issues.append("CUDA环境异常")
    
    # 2) 检查TensorRT系统库路径 - 运行时库依赖
    print("🔍 检查TensorRT系统库:")
    trt_lib_path = os.environ.get('LD_LIBRARY_PATH', '')
    if 'TensorRT' in trt_lib_path or 'tensorrt' in trt_lib_path:
        print(f"   ✅ TensorRT库路径: 已配置")
    else:
        print(f"   ⚠️  TensorRT库路径: 未检测到LD_LIBRARY_PATH中的TensorRT路径")
        print(f"       当前路径: {trt_lib_path[:100]}...")
        tensorrt_issues.append("TensorRT库路径未配置")
    
    # 3) 检查TensorRT Python包 - Python API
    print("🔍 检查TensorRT Python包:")
    try:
        import tensorrt as trt
        version = trt.__version__
        print(f"   ✅ tensorrt ({version}) - TensorRT Python包")
        # 验证基本功能
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        if builder:
            print(f"       ✅ TensorRT引擎构建器可用")
    except ImportError as e:
        # 检查是否是系统库缺失（Python包已安装但找不到.so库）
        if "libnvinfer" in str(e) or "No module named 'tensorrt'" not in str(e):
            print(f"   ⚠️  tensorrt - Python包已安装，但缺少系统级TensorRT库")
            print(f"       错误: {e}")
            tensorrt_issues.append("TensorRT系统库缺失")
        else:
            print(f"   ❌ tensorrt - TensorRT Python包 (未安装)")
            missing.append("tensorrt")
    except Exception as e:
        print(f"   ⚠️  tensorrt - TensorRT功能验证失败: {e}")
        tensorrt_issues.append("TensorRT功能异常")
    
    # 4) 检查TensorRT命令行工具 - 最上层工具
    print("🔍 检查TensorRT工具:")
    tensorrt_tools = [
        ("trtexec", "TensorRT性能测试和转换工具")
    ]
    
    for tool, description in tensorrt_tools:
        try:
            result = subprocess.run(['which', tool], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                tool_path = result.stdout.strip()
                print(f"   ✅ {tool} - {description}")
                print(f"       路径: {tool_path}")
            else:
                print(f"   ❌ {tool} - {description} (未找到)")
                tensorrt_issues.append(f"缺少{tool}工具")
        except Exception as e:
            print(f"   ❌ {tool} - {description} (检查失败: {e})")
            tensorrt_issues.append(f"{tool}工具检查失败")
    
    # 10. NCNN工具链检查 - 验证CPU移动端推理工具
    print(f"\n=== NCNN工具链检查 ===")
    import subprocess
    import os
    
    # 1) 检查protobuf依赖 - NCNN工具的基础依赖
    print("🔍 检查protobuf依赖:")
    ncnn_issues = []
    
    # 检查LD_LIBRARY_PATH中是否存在libprotobuf.so.17
    ld_lib_path = os.environ.get('LD_LIBRARY_PATH', '')
    protobuf_found = False
    
    if ld_lib_path:
        for path in ld_lib_path.split(':'):
            if path.strip():  # 跳过空路径
                protobuf_lib = os.path.join(path.strip(), 'libprotobuf.so.17')
                if os.path.exists(protobuf_lib):
                    print(f"   ✅ protobuf-17: 依赖库可用 (NCNN工具需要)")
                    print(f"       路径: {protobuf_lib}")
                    protobuf_found = True
                    break
    
    if not protobuf_found:
        # 备用检查：系统ldconfig缓存
        try:
            result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True, timeout=10)
            if 'libprotobuf.so.17' in result.stdout:
                print(f"   ✅ protobuf-17: 依赖库可用 (系统缓存)")
                protobuf_found = True
        except Exception as e:
            pass
        
        if not protobuf_found:
            print(f"   ⚠️  protobuf-17: 未找到libprotobuf.so.17 (NCNN工具可能无法运行)")
            ncnn_issues.append("protobuf-17依赖缺失")
    
    # 2) 检查NCNN库路径 - 运行时库路径
    print("🔍 检查NCNN库路径:")
    ld_lib_path = os.environ.get('LD_LIBRARY_PATH', '')
    if 'ncnn' in ld_lib_path.lower():
        print(f"   ✅ NCNN库路径: 已配置")
    else:
        print(f"   ⚠️  NCNN库路径: 未检测到LD_LIBRARY_PATH中的NCNN路径")
        print(f"       当前路径: {ld_lib_path[:100]}...")
        ncnn_issues.append("NCNN库路径未配置")
    
    # 3) 检查NCNN工具可执行性 - 最上层检查
    print("🔍 检查NCNN工具:")
    ncnn_tools = [
        ("onnx2ncnn", "ONNX转NCNN模型转换器"),
        ("ncnnoptimize", "NCNN模型优化器"),
        ("ncnn2table", "NCNN量化校准表生成器"),
        ("ncnn2int8", "NCNN INT8量化转换器")
    ]
    
    for tool, description in ncnn_tools:
        try:
            # 检查工具是否在PATH中
            result = subprocess.run(['which', tool], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                tool_path = result.stdout.strip()
                print(f"   ✅ {tool} - {description}")
                print(f"       路径: {tool_path}")
            else:
                print(f"   ❌ {tool} - {description} (未找到)")
                ncnn_issues.append(f"缺少{tool}工具")
        except Exception as e:
            print(f"   ❌ {tool} - {description} (检查失败: {e})")
            ncnn_issues.append(f"{tool}工具检查失败")
    
    # 检查结果汇总
    print(f"\n=== 检查结果汇总 ===")
    
    # 显示版本信息
    print("📋 已安装版本:")
    try:
        import numpy as np
        print(f"   • numpy: {np.__version__}")
    except:
        print(f"   • numpy: 未安装")
    
    try:
        import PIL
        print(f"   • PIL/Pillow: {PIL.__version__}")
    except:
        print(f"   • PIL/Pillow: 未安装")
    
    try:
        import tensorflow as tf
        print(f"   • TensorFlow: {tf.__version__}")
    except:
        print(f"   • TensorFlow: 未安装")
    
    try:
        import onnx
        print(f"   • ONNX: {onnx.__version__}")
    except:
        print(f"   • ONNX: 未安装")
    
    try:
        import onnxruntime as ort
        print(f"   • ONNX Runtime: {ort.__version__}")
    except:
        print(f"   • ONNX Runtime: 未安装")
    
    try:
        import tflite2onnx
        print(f"   • tflite2onnx: {tflite2onnx.__version__}")
    except:
        print(f"   • tflite2onnx: 未安装")
    
    try:
        import tensorrt as trt
        print(f"   • TensorRT: {trt.__version__}")
    except:
        print(f"   • TensorRT: 未安装")
    
    try:
        import pycuda
        print(f"   • PyCUDA: {pycuda.VERSION_TEXT}")
    except:
        print(f"   • PyCUDA: 未安装")
    
    # 问题汇总
    print("\n🔍 问题汇总:")
    
    if missing:
        print(f"   ❌ 缺少Python依赖: {', '.join(missing)}")
    
    if tensorrt_issues:
        print(f"   ⚠️  TensorRT问题: {', '.join(tensorrt_issues)}")
        print(f"      影响: GPU加速功能受限")
    
    if ncnn_issues:
        print(f"   ⚠️  NCNN问题: {', '.join(ncnn_issues)}")
        print(f"      影响: CPU移动端优化受限")
    
    if not missing and not tensorrt_issues and not ncnn_issues:
        print(f"   ✅ 未发现问题")
    
    # 最终结果
    if missing:
        print("\n❌ 环境检查未完全通过 - 缺少必需的Python依赖")
        return False
    else:
        print("\n✅ 核心Python依赖检查通过")
        if tensorrt_issues:
            print("⚠️  TensorRT功能受限，GPU加速性能可能下降")
        if ncnn_issues:
            print("⚠️  NCNN功能受限，CPU移动端优化可能不可用")
        if not tensorrt_issues and not ncnn_issues:
            print("🎉 所有功能组件完全可用！")
        return True

if __name__ == "__main__":
    success = check_dependencies()
    if not success:
        print("\n💡 修复建议:")
        print("1. 安装Python依赖: pip install -r requirements.txt")
        print("2. 下载并配置TensorRT库:")
        print("   export LD_LIBRARY_PATH=~/work/depend_config/tensorrt/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH")
        print("3. 下载并配置NCNN工具包:")
        print("   export PATH=~/work/depend_config/ncnn/ncnn-20231027-ubuntu-2004-shared/bin:$PATH")
        print("   export LD_LIBRARY_PATH=~/work/depend_config/ncnn/usr/lib/x86_64-linux-gnu:~/work/depend_config/ncnn/ncnn-20231027-ubuntu-2004-shared/lib:$LD_LIBRARY_PATH")
        print("4. 确保CUDA 12.x + 对应cuDNN版本")
        print("5. 重新运行检查: python 00_check_deps.py")
        exit(1)
    else:
        print("\n🚀 环境检查通过，可以开始使用！")
        print("💡 下一步: python 01_infer_basic.py")