import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('D:/project/programs/ncnn_project/ncnn/tools/pnnx')
op_type = 'torch.stack'
 
class Model(nn.Module):
	def __init__(self, dim, attr_data):
		super(Model, self).__init__()
		self.dim = dim
		self.attr_data = attr_data
		

	def forward(self, *v_0):
		tensor_list = []
		if len(self.attr_data) != 0:
	
			for i in range(len(self.attr_data)):
				tensor_list.append(torch.unsqueeze(self.attr_data[i], self.dim))
			
		for vv in v_0:
			tensor_list.append(torch.unsqueeze(vv, self.dim))
		v_1 = torch.cat(tensor_list, self.dim)
		return v_1
	
def export_torchscript(dim, v_0, save_dir, op_name, attr_data = [], input_shapes = None):
	net = Model(dim, attr_data)
	net.eval()
	mod = torch.jit.trace(net, v_0)
	pt_path = os.path.join(save_dir, op_name + '.pt').replace('\\','/')
	mod.save(pt_path)

def check_pass():
	v_0 = torch.rand(1,3,224,224, dtype = torch.float)
	v_1 = torch.rand(1,3,224,224, dtype = torch.float)
	v_2 = torch.rand(1,3,224,224, dtype = torch.float)
	v = [v_0, v_1, v_2]
	#finish your check pass code
# from tools.serializer import *
# def export_pnnx(pt_path_str, input_shape_str):
#     parser = PnnxParser()
    
#     operators, operands, input_ops, output_ops = parser.getNvpPnnxModel(pt_path_str, input_shape_str)
	
if __name__ == "__main__":
    export = 'pt'
    if export == 'pt':
        v_0 = torch.rand(1,3,224,224, dtype = torch.float)
        v_1 = torch.rand(1,3,224,224, dtype = torch.float)
        v_2 = torch.rand(1,3,224,224, dtype = torch.float)
        v_3 = torch.rand(1,3,224,224, dtype = torch.float)
        dim = 1
        v = [v_0,v_1]
        save_dir = r'D:\project\programs\ncnn_project\ncnn\tools\pnnx\model_zoo'
        op_name = 'stack4'
        export_torchscript(dim,v,save_dir,op_name,[v_2,v_3])
    elif export == 'pnnx':
        pt_path_str = 'D:/project/programs/ncnn_project/ncnn/tools/pnnx/model_zoo/stack4.pt' 
        input_shape_str = '[1,3,224,224],[1,3,224,224]'
        export_pnnx(pt_path_str,input_shape_str)
