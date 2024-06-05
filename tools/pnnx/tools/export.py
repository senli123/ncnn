import sys
from typing import List, Union, Optional
import argparse
import os
import shutil
import json
import importlib
import numpy as np
import platform 
try:
    import torch
    import torchvision.models as models
    import torch.nn as nn
    import torch.nn.functional as F
    if platform.system() == "Windows":  
        sys.path.append('D:/project/programs/ncnn_project/nvppnnx/python/build/lib.win-amd64-cpython-38/pnnx')
    elif platform.system() == "Linux": 
        # sys.path.append('/workspace/trans_onnx/project/new_project/nvppnnx/python/build/temp.linux-x86_64-cpython-311/src')
        sys.path.append('/workspace/trans_onnx/project/new_project/ncnn/tools/pnnx/python/build/temp.linux-x86_64-cpython-311/src')
    else:    
        assert False, "noly support win and linux"
    import ptx
    graph = ptx.PnnxGraph()
except ImportError as e:
	sys.exit(str(e))
import onnx
from onnxsim import simplify
 

if platform.system() == "Windows":  
    save_path = 'D:/project/programs/ncnn_project/ncnn/tools/pnnx/model_zoo'
elif platform.system() == "Linux":  
    save_path = '/workspace/trans_onnx/project/new_project/ncnn/tools/pnnx/model_zoo'
else:  
    assert False, "noly support win and linux"


def input_torch_type_to_str(tensor):
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float:
        return "f32"
    if tensor.dtype == torch.float64 or tensor.dtype == torch.double:
        return "f64"
    if tensor.dtype == torch.float16 or tensor.dtype == torch.half:
        return "f16"
    if tensor.dtype == torch.uint8:
        return "u8"
    if tensor.dtype == torch.int8:
        return "i8"
    if tensor.dtype == torch.int16 or tensor.dtype == torch.short:
        return "i16"
    if tensor.dtype == torch.int32 or tensor.dtype == torch.int:
        return "i32"
    if tensor.dtype == torch.int64 or tensor.dtype == torch.long:
        return "i64"
    if tensor.dtype == torch.complex32:
        return "c32"
    if tensor.dtype == torch.complex64:
        return "c64"
    if tensor.dtype == torch.complex128:
        return "c128"

    return "f32"


#-------------------------------------------
# def models 
#-------------------------------------------

class IndexModel(nn.Module):
    def __init__(self,):
        super(IndexModel, self).__init__()
        self.indices= torch.tensor([
            [0, 1],
            [1, 2],
        ], dtype=torch.long) 
    def forward(self, v_0):
        # indices = torch.tensor([0,2], dtype=torch.long) 
        
        # gathered = torch.gather(v_0, dim=3, index=indices)  
        gathered = v_0[:,self.indices,self.indices,:]
        # gathered = v_0[self.indices,:]
        return gathered
    

class stackModel(nn.Module):
    def __init__(self,):
        super(stackModel, self).__init__()
        self.v_3 = torch.rand(1, 2, 3 ,224, dtype=torch.float)
        
    def forward(self, v_0, v_1): #1,3,224
        v_2 = torch.stack([v_0,v_1],dim = 1) #,1,2,3,224
        v_4 = torch.stack([v_2,self.v_3],dim = 1) #,1,2,2,3,224
        return v_4

class oneHotModel(nn.Module):
        def __init__(self,):
            super(oneHotModel, self).__init__()
                 
        def forward(self, v_0):
            one_hot = F.one_hot(v_0, num_classes=4) 
            return one_hot

class reshape_as_Model(nn.Module):
    def __init__(self,):
        super(reshape_as_Model, self).__init__()
        
    def forward(self, x, y):
        output = x.reshape_as(y)
        return output
    
class unfold_Model(nn.Module):
    def __init__(self,):
        super(unfold_Model, self).__init__()
        self.unfold = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(0,1),dilation=(2,2))
    def forward(self, x):
        output = self.unfold(x)
        return output
 

def export(model_name: str, net: Union[nn.Module, str], input_shape, export_onnx: bool):
    if isinstance(input_shape, list):
        input_tensor_list = []
        input_shape_str_list = []
        for shape in input_shape:
            if isinstance(shape, list):
                input_shape_str_list.append('[' + ', '.join(str(item) for item in shape) + ']' )
                input_tensor_list.append(torch.rand(shape, dtype=torch.float))
            elif isinstance(shape, torch.Tensor):
                input_tensor_list.append(shape)
                tensor_shape = shape.shape
                input_shape_str_list.append('[' + ', '.join(str(item) for item in tensor_shape) + ']' + input_torch_type_to_str(shape) )
            else:
                assert False, 'the type in input_shape must be torch.Tensor of list'  
        input_shape_str = ','.join(input_shape_str_list)

    else:
         assert False, 'the type of input_shape must be str or List[torch.Tensor]'
    
    
    save_dir = os.path.join(save_path, model_name)
    os.makedirs(save_dir, exist_ok= True)
    #  export pt
    if len(input_tensor_list) == 1:
        input_tensor_list = input_tensor_list[0]
        
    else:
        input_tensor_list = tuple(input_tensor_list)
    if isinstance(net, str):
        pt_path = net
    else:
        mod = torch.jit.trace(net, input_tensor_list)
        pt_path = os.path.join(save_dir, model_name + '.pt').replace('\\','/')
        mod.save(pt_path)
    # export pnnx
    result = graph.getNvpPnnxModel(pt_path, input_shape_str, 'None', 'None')
    assert result == 1, 'failed to  export pnnx'
    if export_onnx:
        torch.onnx.export(net, input_tensor_list, os.path.join(save_dir, model_name + '.onnx'), verbose=True, input_names=['input0'],
                        output_names=['output0'],opset_version=17)
        model = onnx.load(os.path.join(save_dir, model_name + '.onnx'))
        # convert model
        model_simp, check = simplify(model)

        onnx.save(model_simp, '{}_sim.onnx'.format(os.path.join(save_dir, model_name)))
    

if __name__ == "__main__":

    net_map = {  
    "index2": IndexModel,
    "stack":stackModel,
    "oneHot":oneHotModel,
    "reshape_as": reshape_as_Model,
    "unfold":unfold_Model
} 
    
    model_name = 'unfold'
    if model_name in net_map:  
        net = net_map[model_name]()  
    else:  
        assert False, 'not found model_name: {} in net_map'.format(model_name)

    # input_shape = [[1,3, 224],[1,3,224]]
    # v_0 = torch.tensor( [
	# 		[1.0, 1.2],
	# 		[2.3, 3.4],
	# 		[4.5, 5.7],
	# 	])
    # v_0 = torch.tensor( [
	# 		[1.0, 1.2, 1.3],
	# 		[2.3, 3.4, 1.4],
	# 		[4.5, 5.7, 1.8],
	# 	])
    # v_0 = torch.rand([1,3,4,4], dtype= float)
    # v_0 = torch.tensor([0, 2, 1, 3])
    # input_shape = [v_0]
    # input_shape = [[1,3, 224]]

    # v_0 = torch.randn(2, 3)  # 第一个输入张量  
    # v_1 = torch.randn(3, 2)  # 第二个输入张量  
       
    # input_shape = [v_0,v_1]
    export_onnx = True
    #-------------------------------------------------------
    # model_name = 'pvig'  
    # net = '/workspace/trans_onnx/project/new_project/ncnn/tools/pnnx/model_zoo/pvig/model.pt'
    # input_shape = [[1,3,224,224]]
    # export_onnx = False
    # ----------------------------

    # unfold
    input_shape = [[1,3,9,9]]

    export(model_name, net, input_shape, export_onnx)
    # import pnnx
    # pnnx.export


