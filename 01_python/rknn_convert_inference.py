import numpy as np
import cv2
from rknn.api import RKNN


if __name__ == '__main__':
    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[127.5, 127.5, 127.5]], std_values=[[127.5, 127.5, 127.5]], target_platform='rk3588')
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model='../model/mobilenet_v2_1.0_224.onnx')
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='../input/dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn('../model/mobilenet_v2_1.0_224.rknn')
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(perf_debug=True, eval_mem=True)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread('../input/fish_224x224.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, 0)

    # Inference，输入给u8就行，rknn内部会做归一化和量化，输出也会在rknn内部做反量化
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])

    # Post Process
    print('--> PostProcess')
    with open('../model/labels.txt', 'r') as f:
        labels = [l.rstrip() for l in f]

    # 调试信息
    print(f"原始输出形状: {outputs[0].shape}")
    print(f"原始输出数据类型: {outputs[0].dtype}")
    print(f"原始输出范围: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")

    scores = outputs[0]
    # print the top-5 inferences class
    scores = np.squeeze(scores)
    a = np.argsort(scores)[::-1]
    print('-----TOP 5-----')
    for i in a[0:5]:
        print('[%d] score=%.6f class="%s"' % (i, scores[i], labels[i]))
    print('done')

    # Accuracy analysis
    print('--> Accuracy analysis')
    ret = rknn.accuracy_analysis(inputs=['../input/fish_224x224.jpeg'])
    if ret != 0:
        print('Accuracy analysis failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()
