import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

op_type = 'nn.Unfold'
 
class Model(nn.Module):
	def __init__(self,dilation, kernel_size, padding, stride, input_shapes):
		super(Model, self).__init__()
		assert len(input_shapes) == 1, 'the num of nn.Unfold input must be equal 1'
		input_shape = input_shapes[0]
		assert len(input_shape) == 4, 'the dim of nn.Unfold input must be equal 4'
		self.b, c, ih, iw = input_shape
		ih += 2 * padding[0]
		iw += 2 * padding[1]
		nh = kernel_size[0] + dilation[0]
		nw = kernel_size[1] + dilation[1] 
		# to get indices
		self.indices1 = []
		for i in range(0, kernel_size[0]):
			s = i * dilation[0]
			sub_indices = []
			for j in range(s, ih - nh + s + 1, stride[0]):
				sub_indices.append(j)
			self.indices1.append(sub_indices)
		self.indices2 = []
		for i in range(0, kernel_size[1]):
			s = i * dilation[1]
			sub_indices = []
			for j in range(s, iw - nw + s + 1, stride[1]):
				sub_indices.append(j)
			self.indices2.append(sub_indices)
		self.padding = padding
		self.out_shape = kernel_size[0] * kernel_size[1] * c


	def forward(self, *v_0):
		v_0 = v_0[0]
		if self.padding != (0,0):
			v_0 = F.pad(v_0,(self.padding[1],self.padding[1],self.padding[0],self.padding[0]))
		v_1 = v_0[:,:,self.indices1,:]
		v_2 = v_1[:,:,:,:,self.indices2]
		v_3 = v_2.permute(0,1,2,4,3,5)
		v_4 = v_3.reshape(self.b, self.out_shape,-1)
		return v_4


def export_torchscript(dilation, kernel_size, padding, stride, v_0, save_dir, op_name, attr_data = None):
	net = Model(dilation, kernel_size, padding, stride)
	net.eval()
	mod = torch.jit.trace(net, v_0)
	pt_path = os.path.join(save_dir, op_name + '.pt').replace('\\','/')
	mod.save(pt_path)


if __name__ == "__main__":
	
	v_0 = torch.rand(1, 3, 9, 9, dtype=torch.float)
	kernel_size=(3, 3)
	stride=(1, 1)
	padding=(0,1)
	dilation=(2,2)
	unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation)
	net = Model(dilation, kernel_size, padding, stride, input_shapes = [[1,3,9,9]])
	net.eval()
	v_1 = unfold(v_0)
	v_2 = net(v_0)
	print(v_1 == v_2)
	print(v_1.shape == v_2.shape)
	print(torch.equal(v_1, v_2))
	pass

