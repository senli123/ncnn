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

 

def export(model_name: str, net: nn.Module, input_shape, export_onnx: bool):
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
                input_shape_str_list.append('[' + ', '.join(str(item) for item in tensor_shape) + ']' )
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
    "stack":stackModel
} 
    
    model_name = 'index2'
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
    v_0 = torch.tensor( [
			[1.0, 1.2, 1.3],
			[2.3, 3.4, 1.4],
			[4.5, 5.7, 1.8],
		])
    v_0 = torch.rand([1,3,4,4], dtype= float)
    input_shape = [v_0]
    export_onnx = True
    export(model_name, net, input_shape,export_onnx)
