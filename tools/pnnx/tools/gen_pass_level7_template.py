from serializer import *

def gen_pass_level7_template(ops, output_path, pass_name):
    """ 
     Operator:
        inputs: list(Operand)
        outputs: list(Operand)
        type: str
        name: str
        inputnames: list(str)
        params: dict{str:Parameter}
        attrs: dict{str:Attribute}
    """
    assert os.path.isdir(output_path), 'template output path: {} is not exist'.format(output_path)
    # only support single op template gen
    op_list = []
    for op in ops:
        if op.type == 'pnnx.Input' or op.type == 'pnnx.Output' or op.type == 'pnnx.Attribute':
            continue
        op_list.append(op)
    
    assert len(op_list) == 1, 'only support single op template gen'
    cur_op = op_list[0]
    #parse params and attribute
    params = cur_op.params
    attribute = cur_op.attrs
    init_params_name = list(params.keys()) + list(attribute.keys())
    init_params_name.append('input_shapes')
    op_type = cur_op.type

    output_py_path = os.path.join(output_path, pass_name + '.py')
    with open(output_py_path, "w", encoding="utf-8") as f:
         
        f.write("import os\n")
        f.write("import torch\n") 
        f.write("import torch.nn as nn\n")
        f.write("import torch.nn.functional as F\n")
        f.write("\n")

        #op_type
        f.write("op_type = '{}'\n ".format(op_type))
        f.write("\n")

        #define model
        f.write("class Model(nn.Module):\n")
        init_params_name_str = ', '.join(init_params_name)
        f.write("\tdef __init__(self," + init_params_name_str + "):\n")
        f.write("\t\tsuper(Model, self).__init__()\n") 
        f.write("\t\t# please finish params init \n")
        f.write("\t\tpass\n")
        f.write("\n")

        f.write("\tdef forward(self, *v_0):\n")
        f.write("\t\t# please finish forwad \n")
        f.write("\t\tpass\n")
        f.write("\n")

        #define export_torchscript()
        export_name = init_params_name.copy()
        export_name.append('v_0')
        export_name.append('save_dir')
        export_name.append('op_name')
        export_name.append('attr_data = None')
        export_name.append('input_shapes = None')
        export_params_name_str = ', '.join(export_name)
        f.write("def export_torchscript(" + export_params_name_str + "):\n")
        f.write("\tnet = Model(" + init_params_name_str + ")\n")
        f.write("\tnet.eval()\n")
        f.write("\tmod = torch.jit.trace(net, v_0)\n")
        f.write("\tpt_path = os.path.join(save_dir, op_name + '.pt').replace('\\\\','/')\n")
        f.write("\tmod.save(pt_path)\n")
        f.write("\n")

        #def check pass 
        f.write("def check_pass():\n")
        # rand input
        Operands = cur_op.inputs
        input_num = len(Operands)
        #[todo] cur only support float
        for i in range(input_num):
            f.write("\tv_{} = torch.rand({}, dtype = torch.float)\n".format(i, ','.join([str(dim)for dim in Operands[i].shape])))
        f.write("\tv = [{}]\n".format(', '.join(['v_' + str(j) for j in range(input_num)])))
        f.write("\t#finish your check pass code\n")
        f.write("\n")

        f.write('if __name__ == "__main__":\n')
        f.write("\tcheck_pass()\n")
       
    f.close()


if __name__ == "__main__":

    parser = PnnxParser()
    # stack
    # pt_path_str = 'D:/project/programs/my_project/tests/test_python/test_op/model_zoo2/stack_16/stack_16.pt' 
    # input_shape_str = '[1,3,224,224],[1,3,224,224]'
    # pass_name = 'Stack2UnsqueezewithConcat'
    
    # # stack
    # pt_path_str = 'D:/project/programs/my_project/tests/test_python/test_op/model_zoo2/scaled_dot_product_attention/scaled_dot_product_attention.pt' 
    # input_shape_str = '[1,197,9,64],[1,197,9,64],[1,197,9,64]'
    # pass_name = 'ScaledDotProductAttenPass'

     # unfold
    pt_path_str = '/workspace/trans_onnx/project/new_project/ncnn/tools/pnnx/model_zoo/unfold/unfold.pt' 
    input_shape_str = '[1,3,9,9]'
    pass_name = 'UnfoldPass_new'

    # custom_op_path_str = 
    # infer_py_path = 
    # gen pnnx model
    if platform.system() == "Windows":  
        output_path = 'D:/project/programs/ncnn_project/ncnn/tools/pnnx/pass_level7/template'
    elif platform.system() == "Linux":  
        output_path = '/workspace/trans_onnx/project/new_project/ncnn/tools/pnnx/pass_level7/template'
    else:  
        assert False, "noly support win and linux"

    
    
    operators, operands, input_ops, output_ops = parser.getNvpPnnxModel(pt_path_str, input_shape_str)

    # gen pass level7 template
    gen_pass_level7_template(operators, output_path, pass_name)

