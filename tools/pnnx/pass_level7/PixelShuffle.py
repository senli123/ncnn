import os
import torch
import torch.nn as nn
import torch.nn.functional as F

op_type = 'nn.PixelShuffle'
 
class Model(nn.Module):
	def __init__(self, upscale_factor):
		super(Model, self).__init__()
		self.upscale_factor = upscale_factor
		pass

	def forward(self, *v_0):
		v_1 = v_0[0]
		batch_size, channels, in_height, in_width = v_1.size()  
		channels //= (self.upscale_factor ** 2)  
		out_height = in_height * self.upscale_factor  
		out_width = in_width * self.upscale_factor  

		shuffled = v_1.view(batch_size, channels, self.upscale_factor, self.upscale_factor, in_height, in_width)  
	
		shuffled = shuffled.permute(0, 1, 4, 2, 5, 3)  

		output = shuffled.reshape(batch_size, channels, out_height, out_width)  
		return output

  

def export_torchscript(upscale_factor, v_0, save_dir, op_name, attr_data = None, input_shapes = None):
	net = Model(upscale_factor)
	net.eval()
	mod = torch.jit.trace(net, v_0)
	pt_path = os.path.join(save_dir, op_name + '.pt').replace('\\','/')
	mod.save(pt_path)

def check_pass():
	v_0 = torch.rand(1,64,8,8, dtype = torch.float)
	#finish your check pass code
	model = Model(2)
	model.eval()
	o1 = model(v_0)
	p = nn.PixelShuffle(2)
	o2 = p(v_0)
	print(o1.shape)
	print(o1==o2)

if __name__ == "__main__":
	check_pass()
