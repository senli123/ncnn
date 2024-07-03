import os
import torch
import torch.nn as nn
import torch.nn.functional as F

op_type = 'nn.MultiheadAttention'
 
class Model(nn.Module):
	def __init__(self, embed_dim, num_heads, in_proj_bias, in_proj_weight, out_proj_bias, out_proj_weight):
		super(Model, self).__init__()
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.head_dim = embed_dim // num_heads
		assert self.head_dim * num_heads == self.embed_dim, "Embedding dimension must be divisible by number of heads"
		self.q_proj = nn.Linear(embed_dim, embed_dim)
		self.k_proj = nn.Linear(embed_dim, embed_dim)
		self.v_proj = nn.Linear(embed_dim, embed_dim)
		self.out_proj = nn.Linear(embed_dim, embed_dim)
		# 复制权重
		qkv_weight = in_proj_weight.chunk(3, dim=0)
		qkv_bias = in_proj_bias.chunk(3, dim=0)
		self.q_proj.weight.data.copy_(qkv_weight[0])
		self.k_proj.weight.data.copy_(qkv_weight[1])
		self.v_proj.weight.data.copy_(qkv_weight[2])

		self.q_proj.bias.data.copy_(qkv_bias[0])
		self.k_proj.bias.data.copy_(qkv_bias[1])
		self.v_proj.bias.data.copy_(qkv_bias[2])

		self.out_proj.weight.data.copy_(out_proj_weight)
		self.out_proj.bias.data.copy_(out_proj_bias)
		
	def forward(self, *v_0, attn_mask=None):
		if len(v_0) == 1:
			query = key = value = v_0[0]
		elif len(v_0) == 3:
			query, key, value = v_0
		else:
			assert False, "the length of input tensor must equl to 1 or 3"
		query = query.permute(1, 0, 2)
		key = key.permute(1, 0, 2)
		value = value.permute(1, 0, 2)
		batch_size, seq_len, _ = query.size()

		q = self.q_proj(query)
		k = self.k_proj(key)
		v = self.v_proj(value)

		q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
		k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
		v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

		attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
		if attn_mask is not None:
			attn_weights += attn_mask

		attn_weights = torch.softmax(attn_weights, dim=-1)
		# attn_weights = self.dropout(attn_weights)  # 应用dropout

		attn_output = torch.matmul(attn_weights, v)
		attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.embed_dim)
		attn_output = self.out_proj(attn_output)
		attn_output = attn_output.permute(1, 0, 2)
		return attn_output, attn_weights

def export_torchscript(add_bias_kv, add_zero_attn, batch_first, bias, embed_dim, kdim, num_heads, vdim, in_proj_bias, in_proj_weight, out_proj_bias, out_proj_weight, v_0, save_dir, op_name, attr_data = None, input_shapes = None):
	net = Model(add_bias_kv, add_zero_attn, batch_first, bias, embed_dim, kdim, num_heads, vdim, in_proj_bias, in_proj_weight, out_proj_bias, out_proj_weight)
	net.eval()
	mod = torch.jit.trace(net, v_0)
	pt_path = os.path.join(save_dir, op_name + '.pt').replace('\\','/')
	mod.save(pt_path)

def check_pass():
	original_module = nn.MultiheadAttention(embed_dim=512, num_heads=8)
	in_proj_weight = original_module.in_proj_weight
	in_proj_bias = original_module.in_proj_bias
	out_proj_weight = original_module.out_proj.weight
	out_proj_bias = original_module.out_proj.bias

	custom_module = Model(512, 8, in_proj_bias, in_proj_weight,  out_proj_bias, out_proj_weight)
	custom_module.eval()
	v_0 = torch.rand(256, 1, 512, dtype = torch.float)
	# 运行两个函数
	original_output, _ = original_module(v_0, v_0, v_0)
	custom_output, _ = custom_module(v_0)

	# 比较输出
	atol = 1e-4  
	rtol = 1e-5  
	if torch.allclose(original_output, custom_output, atol=atol, rtol=rtol):
		print("The outputs of the two models are numerically equivalent.")
	else:
		print("The outputs of the two models are not numerically equivalent.")
	#finish your check pass code

if __name__ == "__main__":
	check_pass()
