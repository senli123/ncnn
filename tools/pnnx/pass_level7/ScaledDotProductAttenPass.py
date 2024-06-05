import os
import torch
import torch.nn as nn
import torch.nn.functional as F

op_type = 'F.scaled_dot_product_attention'
 
class Model(nn.Module):
	def __init__(self,attn_mask, dropout_p, is_causal):
		super(Model, self).__init__()
		# please finish params init 
		pass

	def forward(self, *v_0):
		v_1, v_2, v_3 = v_0
		v_4 = torch.transpose(input=v_2, dim0=-2, dim1=-1)
		v_5 = torch.matmul(input=v_1, other=v_4)
		v_6 = (v_5 * 1.250000e-01)
		v_7 = F.softmax(input=v_6, dim=-1)
		v_8 = torch.matmul(input=v_7, other=v_3)
		return v_8

def export_torchscript(attn_mask, dropout_p, is_causal, v_0, save_dir, op_name, attr_data = None, input_shapes = None):
	net = Model(attn_mask, dropout_p, is_causal)
	net.eval()
	mod = torch.jit.trace(net, v_0)
	pt_path = os.path.join(save_dir, op_name + '.pt').replace('\\','/')
	mod.save(pt_path)

def check_pass():
	v_0 = torch.rand(1,197,9,64, dtype = torch.float)
	v_1 = torch.rand(1,197,9,64, dtype = torch.float)
	v_2 = torch.rand(1,197,9,64, dtype = torch.float)
	v = [v_0, v_1, v_2]
	#finish your check pass code

if __name__ == "__main__":
	check_pass()
