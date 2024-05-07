#include <torch/torch.h>
torch::Tensor ReduceL2(const torch::Tensor& input_tensor, std::vector<int64_t> axis, bool keepdims)
{
    torch::Tensor output;
    if (axis.size() == 1 && axis[0] == -1)
    {
        output = torch::sqrt(torch::sum(torch::square(input_tensor)));
        if (keepdims)
        {
            std::vector<int64_t> shape = input_tensor.sizes().vec();
            for (int64_t dim : shape)
            {
                output = output.unsqueeze(0);
            }
        }
    }
    else
    {
        output = torch::sqrt(torch::sum(torch::square(input_tensor), axis, keepdims));
    }
    return output;
}

TORCH_LIBRARY(ReduceL2_op, m)
{
    m.def("ReduceL2", ReduceL2);
}