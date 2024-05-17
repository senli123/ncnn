
import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
except:
    pass
import scipy.spatial.distance as dist
def simlarity(out, golden):
    return 1 - dist.cosine(out.ravel().astype(np.float32), golden.ravel().astype(np.float32))

def layer_range():
    path = 'D:/project/compiler_info/pnnx/test/pudi/EDAM-master/output/resnet38_EDAM_cls_sim/nvpncc/output/ref/Tensor_706.bin'
    ref_res = np.fromfile(path, np.float32).reshape((1,21,32,32))
    print(np.min(ref_res), np.max(ref_res))

def compare(cpp,ref):
    cpp_res = np.fromfile(cpp, np.float16).reshape((1,256,256,3))
    ref_res = np.fromfile(ref, np.float32).reshape((1,3,256,256))
    cpp_res = np.transpose(cpp_res,(0,3,1,2))
    # cpp_res = cpp_res[:,0,:,:]
    print(cpp_res.shape)
    print(ref_res.shape)
    print(np.min(cpp_res), np.max(cpp_res))
    print(np.min(ref_res), np.max(ref_res))
    print(simlarity(cpp_res,ref_res))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.mask = None
        self.dropout = 0.0
    
    # def forward(self, v_0, v_1, v_2):
    #     v_3 = torch.transpose(input=v_1, dim0=-2, dim1=-1)
    #     v_4 = torch.matmul(input=v_0, other=v_3)
    #     v_5 = (v_4 * 1.250000e-01)
    #     v_6 = F.softmax(input=v_5, dim=-1)
    #     v_7 = torch.matmul(input=v_6, other=v_2)
    #     v_8 = (v_7, )
    #     return v_8

    def forward(self, *v_0):
        v_1, v_2, v_3 = v_0
        v_4 = torch.transpose(input=v_2, dim0=-2, dim1=-1)
        v_5 = torch.matmul(input=v_1, other=v_4)
        v_6 = (v_5 * 1.250000e-01)
        v_7 = F.softmax(input=v_6, dim=-1)
        v_8 = torch.matmul(input=v_7, other=v_3)
        return v_8
    
    # def forward(self, query, key, value):  
    #     d_k = query.size(-1)  
    #     scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  
    #     if self.mask is not None:  
    #         scores = scores.masked_fill(self.mask == 0, -1e9)  # Add the mask to the scaled tensor.  
    #     attention_weights = F.softmax(scores, dim=-1)  
    #     # if dropout is not None:  
    #     # attention_weights = dropout(attention_weights)  
    #     output = torch.matmul(attention_weights, value)  
    #     return output, attention_weights

def export_torchscript():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 197, 9, 64, dtype=torch.float)
    v_1 = torch.rand(1, 197, 9, 64, dtype=torch.float)
    v_2 = torch.rand(1, 197, 9, 64, dtype=torch.float)

    mod = torch.jit.trace(net, (v_0, v_1, v_2))
    mod.save("D:\nextvpu\models\001_pnnx\F_scaled_dot_product_attention\scaled_dot_product_atten_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 197, 9, 64, dtype=torch.float)
    v_1 = torch.rand(1, 197, 9, 64, dtype=torch.float)
    v_2 = torch.rand(1, 197, 9, 64, dtype=torch.float)

    torch.onnx._export(net, (v_0, v_1, v_2), "D:\nextvpu\models\001_pnnx\F_scaled_dot_product_attention\scaled_dot_product_atten_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0', 'in1', 'in2'], output_names=['out0'])

def test_inference():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 197, 9, 64, dtype=torch.float)
    v_1 = torch.rand(1, 197, 9, 64, dtype=torch.float)
    v_2 = torch.rand(1, 197, 9, 64, dtype=torch.float)

    return net(v_0, v_1, v_2)

if __name__ == "__main__":
    v_0 = torch.rand(1, 197, 9, 64, dtype=torch.float)
    v_1 = torch.rand(1, 197, 9, 64, dtype=torch.float)
    v_2 = torch.rand(1, 197, 9, 64, dtype=torch.float)
    v_3 = F.scaled_dot_product_attention(v_0,v_1,v_2)
    model = Model()
    model.eval()
    v_4 = model(v_0,v_1,v_2)
    # print(v_3 == v_4[0])
    # result1 = np.array_equal(v_3.numpy(), v_4[0].numpy()) 
    result1 = simlarity(v_3.numpy(), v_4.numpy())
    print(result1)