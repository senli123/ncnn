import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.spatial.distance as dist
def simlarity(out, golden):
    return 1 - dist.cosine(out.ravel().astype(np.float32), golden.ravel().astype(np.float32))
op_type = 'nn.Unfold'
 
class Model(nn.Module):
	def __init__(self,dilation, kernel_size, padding, stride):
		super(Model, self).__init__()
		self.dilation
		pass

	def forward(self, *v_0):
		# please finish forwad 
		pass

def export_torchscript(dilation, kernel_size, padding, stride, v_0, save_dir, op_name, attr_data = None):
	net = Model(dilation, kernel_size, padding, stride)
	net.eval()
	mod = torch.jit.trace(net, v_0)
	pt_path = os.path.join(save_dir, op_name + '.pt').replace('\\','/')
	mod.save(pt_path)

def check_pass():
	v_0 = torch.rand(1,1,4,4, dtype = torch.float)
	v = [v_0]
	#finish your check pass code

def my_unfold(input, kernel_size, stride=1):  
    # 假设输入是一个四维张量：(batch_size, channels, height, width)  
    # kernel_size是块（或滤波器）的大小  
    # stride是块在输入张量上滑动的步长  
      
    batch_size, channels, height, width = input.size()  
      
    # 计算块在高度和宽度方向上的数量  
    height_blocks = (height - kernel_size[0]) // stride + 1  
    width_blocks = (width - kernel_size[1]) // stride + 1  
      
    # 首先，我们需要重塑输入张量以便在每个通道上展开  
    # 注意：这里并没有直接使用gather，但我们可以认为reshape和transpose在某种程度上模拟了gather的功能  
    # 假设我们已经在通道维度上展开了块  
    unfolded_shape = (batch_size, channels, height_blocks, stride, width_blocks, stride, kernel_size[0], kernel_size[1])  
    unfolded = input.unsqueeze(3).unsqueeze(5).repeat(1, 1, 1, height_blocks, 1, width_blocks, 1, 1)  
      
    # 使用arange来创建索引，这类似于gather中的索引操作  
    # 注意：这只是一个简化的索引创建方法，它假设步长为1且没有填充  
    indices_h = torch.arange(0, kernel_size[0]) * stride  
    indices_w = torch.arange(0, kernel_size[1]) * stride  
    indices_h = indices_h.unsqueeze(0).unsqueeze(2).repeat(1, height_blocks, width_blocks, 1)  
    indices_w = indices_w.unsqueeze(0).unsqueeze(1).repeat(1, height_blocks, height_blocks, 1)  
    indices_h = indices_h.view(-1)  
    indices_w = indices_w.view(-1)  
      
    # 这里我们可以使用gather来收集数据，但由于我们使用了reshape和索引，我们可以直接索引数据  
    # 但请注意，直接索引通常不如PyTorch内部使用的gather操作高效  
    # unfolded = ... (使用indices_h和indices_w来索引input的对应位置，但这在这里不是直接的gather调用)  
      
    # 由于直接索引操作在这里不适用（因为它需要复杂的索引张量），我们将使用reshape和transpose来模拟结果  
    # 我们假设已经通过某种方式获得了块数据（在真实场景中，你需要使用索引或循环来获取这些数据）  
    # 这里我们只是模拟块的形状  
    unfolded_simulated = input.new_zeros(batch_size, channels, height_blocks, width_blocks, kernel_size[0], kernel_size[1])  
      
    # 注意：这里我们没有实际填充unfolded_simulated，因为它只是为了展示形状  
      
    # 最终的unfolded张量应该有形状：(batch_size, channels, height_blocks * width_blocks, kernel_size[0] * kernel_size[1])  
    # 我们可以使用reshape和transpose来实现这个效果  
    unfolded_final = unfolded_simulated.permute(0, 1, 2, 4, 3, 5).contiguous().view(batch_size, channels, height_blocks * width_blocks, -1)  
      
    return unfolded_final  
  
def gather(data, indices,axis):
	"""onnx的gather算子模拟，类似于numpy中的take
		data = [
			[1.0, 1.2],
			[2.3, 3.4],
			[4.5, 5.7],
		]
		--------------------------
		axis = 0
		indices = [
			[0, 1],
			[1, 2],
		]
		
		y = [	[[1.  1.2]
				[2.3 3.4]]

				[[2.3 3.4]
				[4.5 5.7]]]
		--------------------------
		data = [ [1.0, 1.2, 1.9],
				[2.3, 3.4, 3.9],
				[4.5, 5.7, 5.9]]
		indices = [
			[0, 2],
		]
		axis = 1,
		y =  [
			[[1.0, 1.9]],
			[[2.3, 3.9]],
			[[4.5, 5.9]],
			]
	"""
	y = np.take(data, indices, axis= axis)
	tensor = torch.from_numpy(data)
	# a = tensor[:,indices]
	a = tensor[indices,:]
	z = a.numpy()
	print(simlarity(y,z))
	print(z)
	return y
	  
if __name__ == "__main__":
	# # 示例用法  
	# input = torch.randn(1, 1, 4, 4)  # 假设的输入张量  
	# input = torch.arange(0, 16,dtype=torch.float).view(4,4)
	# kernel_size = (2, 2)  # 块大小  
	# stride = 1  # 步长  
	
	# unfolded = my_unfold(input, kernel_size, stride)  
	# print(unfolded)
	# indices = torch.tensor([[0, 1,2,3], [0, 1,2,3],[0, 1,2,3],[0, 1,2,3]], dtype=torch.long) 
	# gathered = torch.gather(input, dim=0, index=indices)
	# print(gathered)
	# import torch  
  
	# # 假设我们有一个形状为[2, 4]的二维张量  
	# data = torch.tensor([[1, 2, 3, 4],  
	# 					[5, 6, 7, 8]])  
	
	# # 我们有一个一维张量，包含了要收集的列的索引（沿着dim=1）  
	# indices = torch.tensor([1, 3])  
	
	# # 使用torch.gather沿着第二个维度收集元素  
	# # 因为indices的形状是[2]，它会被广播以匹配data的第二维度的大小  
	# result = torch.gather(data, dim=1, index=indices.unsqueeze(1))  
	
	# print(result)  
	# # 输出将是：  
	# # tensor([[ 2,  4],  
	# #         [ 6,  8]])


	#---------------测试np.take(onnx gather) 等价于tensor.index
	# data = np.array([ [1.0, 1.2, 1.9],
    # [2.3, 3.4, 3.9],
    # [4.5, 5.7, 5.9]])
	# indices = [
	# 			[0, 2],
	# 		]
	# axis = 1
# 	data = np.array([
# 	[1.0, 1.2],
# 	[2.3, 3.4],
# 	[4.5, 5.7],
# ])
	
# 	axis = 0
# 	indices =np.array( [
# 		[0, 1],
# 		[1, 2],
# 	])
	

	
	# output1 = gather(data, indices, axis)
	# # print(data.shape)
	# # print(output1.shape)	
	# print(output1)

	#-----------------------------------------------
	tensor_0 = torch.arange(0, 16,dtype=torch.float).view(1,1,4,4)
	print('input:',tensor_0)
	kernel_size = 2
	stride = 2
	fold = nn.Unfold(kernel_size=(kernel_size, kernel_size), stride=(stride, stride))
	out1 = fold(tensor_0)
	print("out1_shape:",out1.shape)
	print("out1:",out1)

	# -----------torch实现----------------
	indices1 = torch.tensor([[0,2],[1,3]])
	gather1 = tensor_0[:,:,indices1,:]
	print("gather1_shape:",gather1.shape)
	print(gather1)

	indices2 = torch.tensor([[0,2],[1,3]])
	gather2 = gather1[:,:,:,:,indices2]
	print("gather2_shape:",gather2.shape)
	print(gather2)


	# array0 = tensor_0.numpy()
	# indices1 = np.array([[0,2],[1,3]])
	# axis1 = 2
	# y = np.take(array0, indices1, axis= axis1)
	# print(y)
	# print(y.shape)

	# z = torch.take_along_dim(tensor_0,indices=indices1_tensor,dim=axis1)
	# print(z)
	# print(z.shape)
	# import torch
	# import numpy as np
	
	# # 示例数据
	# input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
	# indices = torch.tensor([0, 2, 1])
	
	# # 使用广播机制和条件索引实现np.take功能
	# result = input_tensor[torch.arange(input_tensor.shape[0]).repeat(indices.shape[0], 1), indices]
	
	# # 将结果转换为NumPy数组
	# result_np = result.numpy()
	
	# # 验证结果
	# print(np.take(input_tensor.numpy(), indices.numpy(), axis=1))
	# print(result_np)
	
	# # 确保两者结果相同
	# assert np.array_equal(np.take(input_tensor.numpy(), indices.numpy(), axis=1), result_np)

