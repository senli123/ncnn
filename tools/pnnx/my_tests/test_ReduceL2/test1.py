# import torch
# import torchvision.models as models
# import pnnx

# net = models.resnet18(pretrained=True)
# x = torch.rand(1, 3, 224, 224)

# # You could try disabling checking when torch tracing raises error
# # opt_net = pnnx.export(net, "resnet18.pt", x, check_trace=False)
# opt_net = pnnx.export(net, "resnet18.pt", x)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()

#     def forward(self, x):
#         x = F.relu(x)
#         return x
# def run_exe():
#     import pnnx
#     pnnx.convert("test_F_relu.pt", [1, 16], "f32")

# def run_so():
#     import ctypes
#     my_lib = ctypes.CDLL("D:/project/compiler_info/pnnx/pnnx_win/ncnn-master/tools/pnnx/build/lib.win-amd64-cpython-38/pnnx/pnnx.dll")
#     # 定义函数的参数类型和返回值类型
#     # my_lib.main.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)] 
#     # # 设置返回值类型  
#     # my_lib.main.restype = ctypes.c_int 
#     # # 准备参数  
#     # argc = 3  # 假设只有一个参数  
#     # argv = ctypes.POINTER(ctypes.c_char_p)()  
#     # argv_strings = (ctypes.c_char_p * argc)() 
#     # argv_strings[0] = bytes("/0",encoding='utf-8') 
#     # argv_strings[1] = bytes("test_F_relu.pt",encoding='utf-8') 
#     # argv_strings[2] = bytes("inputshape=[1,16]",encoding='utf-8') 
#     # argv.value = argv_strings  
#     # # 调用函数
#     # result = my_lib.main(argc, argv)
#     # return result
#     my_lib.getNvpPnnxModel.argtypes = [ctypes.POINTER(ctypes.c_char),ctypes.POINTER(ctypes.c_char),ctypes.POINTER(ctypes.c_char)]
#     pt_path_str = b'D:/project/compiler_info/pnnx/pnnx_win/ncnn-master/tools/pnnx/python/examples/test/test_F_relu.pt'
#     input_shape_str = b'[1,16]'
#     custom_op_path_str = b'None'
#     result = my_lib.getNvpPnnxModel(pt_path_str, input_shape_str, custom_op_path_str)
#     return result

# def run_extension():
#     import torch
#     pyd_path = "/workspace/trans_onnx/package/ncnn_master/ncnn-master/tools/pnnx/build/temp.linux-x86_64-cpython-311/src/libpnnx.so"
#     torch.ops.load_library(pyd_path)
#     # my_lib = ctypes.CDLL("/workspace/trans_onnx/package/ncnn_master/ncnn-master/tools/pnnx/build/temp.linux-x86_64-cpython-311/src/libpnnx.so")
#     # from pnnx import PnnxGraph
    
#     # model = my_lib.PnnxGraph()
#     pt_path_str = '/workspace/trans_onnx/package/ncnn_master/ncnn-master/tools/pnnx/test_F_relu_/test_F_relu.pt'
#     input_shape_str = '[1,16]'
#     custom_op_path_str = 'None'
#     result = torch.ops.getNvpPnnxModel(pt_path_str, input_shape_str, custom_op_path_str)
#     return result


def run_pybind():
    
    # import torch
    # import ctypes
    # lib_path = "/workspace/trans_onnx/package/ncnn_master/ncnn-master/tools/pnnx/pnnx.cpython-311-x86_64-linux-gnu.so"
 
    # # 加载.pyd文件
    # my_library = ctypes.CDLL(lib_path)
    import sys
    try:
        import torch
        import ptx
    except ImportError as e:
        sys.exit(str(e))
    # pt_path_str = 'test_F_relu.pt'
    # input_shape_str = '[1,16]'
    # custom_op_path_str = 'None'
    pt_path_str = 'ResNet_con_torch21.pt'
    pt_path_str = '/workspace/trans_onnx/project/ncnn/tools/pnnx/my_tests/test_ReduceL2/build/ReduceL2.pt'

    input_shape_str = '[1,3,224,224]'

    custom_op_path_str = 'None'
    custom_op_path_str = '/workspace/trans_onnx/project/ncnn/tools/pnnx/my_tests/test_ReduceL2/build/ReduceL2.so'

    custom_op_py_path_str = 'None'
    custom_op_py_path_str = '/workspace/trans_onnx/project/ncnn/tools/pnnx/my_tests/test_ReduceL2/ReduceL2_infer.py'
    graph = ptx.PnnxGraph()
    result = graph.getNvpPnnxModel(pt_path_str, input_shape_str, custom_op_path_str, custom_op_py_path_str)
    if(result):
        # a = graph.loadModel('ResNet_con_torch21.pnnx.param',\
        #                     'ResNet_con_torch21.pnnx.bin')
        a = graph.loadModel('/workspace/trans_onnx/project/ncnn/tools/pnnx/my_tests/test_ReduceL2/build/ReduceL2.pnnx.param',\
                            '/workspace/trans_onnx/project/ncnn/tools/pnnx/my_tests/test_ReduceL2/build/ReduceL2.pnnx.bin')
        operators = graph.getOperators()
        operands = graph.getOperands()
        input_ops = graph.getInputOps()
        output_ops = graph.getOutputOps()
        return a
    return False
import importlib
def load_module(module_path):
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

import torch
import numpy as np
import scipy.spatial.distance as dist
def ReduceL2(input_tensor, axis, keepdims):
    output_array = np.sqrt(np.sum(np.square(input_tensor.numpy()), axis= axis, keepdims=keepdims))
    return torch.from_numpy(output_array)
def simlarity(out, golden):
    return 1 - dist.cosine(out.ravel().astype(np.float32), golden.ravel().astype(np.float32))

if __name__ == "__main__":
    # net = Model()
    # net.eval()

    # torch.manual_seed(0)
    # x = torch.rand(1, 16)

    # a0 = net(x)

    # mod = torch.jit.trace(net, x)
    # mod.save("test_F_relu.pt")
    # run_so()
    print(run_pybind())
    py_path = '/workspace/trans_onnx/project/ncnn/tools/pnnx/my_tests/test_ReduceL2/build/ReduceL2_pnnx_infer.py'
    bin_path = '/workspace/trans_onnx/project/ncnn/tools/pnnx/my_tests/test_ReduceL2/build/ReduceL2.pnnx.bin'
    module = load_module(py_path)
    function = getattr(module, 'test_inference')
    random_array = np.random.rand(1, 3, 224, 224) 
    input = torch.from_numpy(random_array)
    pnnx_out = function(bin_path, True, input)
    pnnx_out = pnnx_out.detach().numpy()
    python_out = ReduceL2(input, (1), False)
    python_out = python_out.detach().numpy()
    result = simlarity(pnnx_out, python_out)
    print("simlarity:{}".format(result))