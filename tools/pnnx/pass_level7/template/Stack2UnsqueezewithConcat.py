import os
import torch
import torch.nn as nn
import torch.nn.functional as F

op_type = 'torch.stack'
 
class Model(nn.Module):
	def __init__(self,dim):
		super(Model, self).__init__()
		# please finish params init 
		pass

	def forward(self, *v_0):
		# please finish forwad 
		pass

def export_torchscript(dim, *v_0, save_dir, op_name):
	net = Model(dim)
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

if __name__ == "__main__":
	check_pass()
