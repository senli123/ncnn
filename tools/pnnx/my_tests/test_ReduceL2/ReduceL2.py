import os
import torch.nn as nn
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load, _import_module_from_library
import numpy as np
def build_op(define_path_list, build_output_path):
    # module_path = os.path.dirname(__file__)
    upfirdn2d_op = load(
        'ReduceL2',
        sources=define_path_list,
        build_directory = build_output_path,
        is_python_module=False
    )

def load_op(pyd_path):
    torch.ops.load_library(pyd_path)

def test_op(pyd_path):
    load_op(pyd_path)
    input = torch.rand(2,3)
    a = np.sqrt(np.sum(np.square(input.numpy()), axis= (1), keepdims=False))
    b = torch.ops.ReduceL2_op.ReduceL2(_0 = input, _1 =[1], _2 = False)
    print(a,b)
  
    

class example(nn.Module):
    def __init__(self):
          super().__init__()
    def forward(self, x1):
        x2 = torch.ops.ReduceL2_op.ReduceL2(x1, [1], False)
        return x2

def export_torchscript(output_dir):
    net = example()
    net.eval()
    torch.manual_seed(0)
    v_1 = torch.rand((1, 3, 224, 224), dtype=torch.float)
    mod = torch.jit.trace(net, v_1)
    output_path = os.path.join(output_dir, 'ReduceL2.pt')
    mod.save(output_path)

def ref():
    torch.ops.load_library('D:/project/history/20240221/compiler-v3/plugin/torch/ReduceL2/build/ReduceL2.pyd')
    model = torch.load("D:/project/history/20240221/compiler-v3/plugin/torch/ReduceL2/build/ReduceL2.pt")
    model.eval()
    v_1 = torch.rand((1, 3, 224, 224), dtype=torch.float)
    print(model(v_1))
   






#-------------------------------------select mode------------------------------------------------------
# 1: build op 2:load and test op [2,3]: load -> test op -> export_torchscripts  5: export pnnx
mode_list = [2,3]
for mode in mode_list:
    if mode == 1:
        define_path_list = ['/workspace/trans_onnx/project/ncnn/tools/pnnx/my_tests/test_ReduceL2/ReduceL2.cpp']
        build_output_path = '/workspace/trans_onnx/project/ncnn/tools/pnnx/my_tests/test_ReduceL2/build'
        build_op(define_path_list, build_output_path)
    elif mode == 2:
        pyd_path ='/workspace/trans_onnx/project/ncnn/tools/pnnx/my_tests/test_ReduceL2/build/ReduceL2.so'
        test_op(pyd_path)
    elif mode == 3:
        output_dir = '/workspace/trans_onnx/project/ncnn/tools/pnnx/my_tests/test_ReduceL2/build'
        export_torchscript(output_dir)
    elif mode == 5:
        os.system("D:/project/compiler_info/pnnx/pnnx_win/ncnn-master/tools/pnnx/build/lib.win-amd64-cpython-38/pnnx/pnnx.exe ReduceL1.pt inputshape=[1,3,224,224] customop='D:/project/history/20240221/compiler-v3/custom_op/ReduceL1/pnnx_build/ReduceL1.pyd'")
    elif mode == 6:
        ref()

#D:/project/compiler_info/pnnx/pnnx_win/ncnn-master/tools/pnnx/build/lib.win-amd64-cpython-38/pnnx/pnnx.exe custom.pt inputshape=[1,3,224,224]
        

       