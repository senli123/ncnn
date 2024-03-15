
import torch
import numpy as np

def ReduceL2(_0 , _1 , _2 ):
    input_tensor = _0
    axis = _1
    keepdims = _2
    output_array = np.sqrt(np.sum(np.square(input_tensor.numpy()), axis= axis, keepdims=keepdims))
    return torch.from_numpy(output_array)