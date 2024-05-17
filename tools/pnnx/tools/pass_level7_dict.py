from serializer import *
import numpy as np
import struct
import copy
def ParseParams(op, customOp_attrs = None):
    """Convert a list of AttributeProto to a dict, with names as keys."""
    params_data = {}
    params = op.params
    
    #parse parms
    #0=null 1=b 2=i 3=f 4=s 5=ai 6=af 7=as 8=others
    for name, param in params.items():
        param_type = param.type
        if param_type == 0:
            params_data[name] = None
        elif param_type == 1:
            params_data[name] = param.b
        elif param_type == 2:
            params_data[name] = param.i
        elif param_type == 3:
            params_data[name] = param.f   
        elif param_type == 4:
            params_data[name] = param.s  
        elif param_type == 5:
            params_data[name] = param.a_i  
        elif param_type == 6:
            params_data[name] = param.a_f  
        elif param_type == 7:
            params_data[name] = param.a_s
        else:  
            raise Exception("params type [{}] do not supported!".format(param_type))
    if 'padding' in params_data and params_data['padding'] == 'same':
        params_data['padding'] = (np.array(params_data['kernel_size']) // 2).tolist()
    if customOp_attrs == None:
        return params_data
    else:
        update_params_data = {}
        for op_name, custom_op_name in customOp_attrs.items():
            if custom_op_name not in params_data:
                raise Exception("please check customOp_attrs {}:{}!".format(op_name, custom_op_name))
            update_params_data[op_name] = params_data[custom_op_name]
        return update_params_data
    

def ParseAttrs(op):
    attrs_data = {}
    #parse attrs
    attrs = op.attrs
    for name,attr in attrs.items():
        sub_dict = {}
        sub_dict['shape'] = attr.shape
        if attr.type == 1:
            dtype = 'float32'
        elif attr.type == 2:
            dtype = 'float64'
        elif attr.type == 3:
            dtype = 'float16'
        elif attr.type == 4:
            dtype = 'int32'
        elif attr.type == 5:
            dtype = 'int64'
        elif attr.type == 6:
            dtype = 'int16'
        elif attr.type == 7:
            dtype = 'int8'
        elif attr.type == 8:
            dtype = 'uint8'
        elif attr.type == 9:
            dtype = 'bool'
        else:
             raise Exception("attr.type [{}] do not supported!".format(attr.type))
        if hasattr(attr,'b_data'):
            sub_dict['data'] = np.frombuffer(attr.b_data, dtype=dtype)
        else:
            sub_dict['data'] = attr.data
        attrs_data[name] = sub_dict
    return attrs_data

def load_module(module_path):
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def debug_op(operators):
    for op in operators:
        print("----------cur op name :{}-------------".format(op.name))
        op_in = [ i.name for i in op.inputs] 
        op_out = [ o.name for o in op.outputs] 
        print("inputs :{}".format(op_in))
        print("outputs :{}".format(op_out))
        

def debug_operand(operands):
    for tensor in operands:
        print("----------cur tensor name :{}-------------".format(tensor.name))
        producer = tensor.producer.name
        consumers = [ o.name for o in tensor.consumers] 
        print("producer :{}".format(producer))
        print("consumers :{}".format(consumers))


def get_src_node_info(op):
    input_names, input_shapes, input_datas  = [], [], []
    attr_input_names, attr_input_datas = [], []
    inOperands = op.inputs
    for operand in inOperands:
        if operand.producer.type == 'pnnx.Attribute':
            attrs_params = ParseAttrs(operand.producer)
            operand_dict = attrs_params["data"]
            attr_input_names.append(operand.name)
            attr_input_datas.append(operand_dict['data'].reshape(operand_dict['shape']))
        else:
            input_names.append(operand.name)
            input_shapes.append(operand.shape)
            input_datas.append(torch.rand(operand.shape, dtype = torch.float))
            
    
    outOperands = op.outputs
    output_names = [out_operand.name for out_operand in outOperands]
    return input_names, input_shapes, input_datas, attr_input_names, attr_input_datas, output_names

def trans_list_to_dict(operator, operand, update_name = False, cur_op_name = ''):

    def update_operand_name(operands,operator_update_name_dict,operand_update_name_dict):
        for operand in operands:
            if operand.name in operand_update_name_dict and operand_update_name_dict[operand.name] == False:
                operand_update_name_dict[operand.name] = True
                operand.name = cur_op_name + '_tensor_' + operand.name
                # get producer consumers
                producer = operand.producer
                consumers = operand.consumers
                update_operator_name([producer],operator_update_name_dict,operand_update_name_dict)
                update_operator_name(consumers,operator_update_name_dict,operand_update_name_dict)
    
    def update_operator_name(operator, operator_update_name_dict, operand_update_name_dict):
         for index, op in enumerate(operator):
            if op.name in operator_update_name_dict and operator_update_name_dict[op.name] == False:
                operator_update_name_dict[op.name] = True
                operator[index].name = cur_op_name + '_expand_' + operator[index].name
                # get inputs outputs
                inputs_operand = op.inputs
                outputs_operand = op.outputs
                update_operand_name(inputs_operand,operator_update_name_dict,operand_update_name_dict)
                update_operand_name(outputs_operand,operator_update_name_dict,operand_update_name_dict)

    if update_name:
        operator_update_name_dict = {}
        for op in operator:
            operator_update_name_dict[op.name] =False

        operand_update_name_dict = {}
        for tensor in operand:
            operand_update_name_dict[tensor.name] = False
        new_operator_update_name_dict = operator_update_name_dict.copy()
        new_operand_update_name_dict = operand_update_name_dict.copy()
        update_operator_name(operator,operator_update_name_dict,operand_update_name_dict)
        for op in operator:
            op.name =  cur_op_name + '_expand_' + op.name
        update_operand_name(operand,new_operator_update_name_dict,new_operand_update_name_dict)

    operator_dict = {op.name: op for op in operator}
    operand_dict = {tensor.name: tensor for tensor in operand}
    return operator_dict, operand_dict





def trans_dict_to_list(operator_dict, operand_dict):
    operator = list(operator_dict.values())
    operand = list(operand_dict.values())
    return operator, operand


def get_pre_node_name(operand_dict, operand_names):
    pre_node_name = []
    for input_name in operand_names:
        pre_node_name.append(operand_dict[input_name].producer.name)

    return pre_node_name


if __name__ == "__main__":
   
   
    parser = PnnxParser()
    # stack
    # example_name = 'stack'
    # pt_path_str = 'D:/project/programs/my_project/tests/test_python/test_op/model_zoo2/stack_16/stack_16.pt' 
    # input_shape_str = '[1,3,224,224],[1,3,224,224]'

    example_name = 'scaled_dot_product_attention'
    pt_path_str = 'D:/project/programs/my_project/tests/test_python/test_op/model_zoo2/scaled_dot_product_attention/scaled_dot_product_attention.pt' 
    input_shape_str = '[1,197,9,64],[1,197,9,64],[1,197,9,64]'

    # custom_op_path_str = 
    # infer_py_path = 
    pass_level7_path = 'D:/project/programs/ncnn_project/ncnn/tools/pnnx/pass_level7'
    # gen pnnx model
    operators, operands, input_ops, output_ops = parser.getNvpPnnxModel(pt_path_str, input_shape_str)
    # trans list to dict for pass 
    operator_dict, operand_dict = trans_list_to_dict(operators, operands)
    
    
    pass_level7_tmp_output_path = 'D:/project/programs/ncnn_project/ncnn/tools/pnnx/output/tmp'
    pass_level7_tmp_output_path = os.path.join(pass_level7_tmp_output_path, example_name)
    os.makedirs(pass_level7_tmp_output_path, exist_ok= True)
    # loop all pass level7
    all_pass_files = os.listdir(pass_level7_path)
    all_pass_files = [pass_file for pass_file in all_pass_files if pass_file not in ['__init__.py'] and not os.path.isdir(os.path.join(pass_level7_path, pass_file)) ]
    for pass_file in all_pass_files:
        pass_name, extension = os.path.splitext(pass_file)
        if extension != '.py':
            continue
        print("run pass:{}".format(pass_name))
        passMod = load_module(os.path.join(pass_level7_path, pass_file))
        if hasattr(passMod, 'op_type'):
            op_type = getattr(passMod, 'op_type')
        else:
            assert False, 'There are not op_type attr in pass: {}'.format(pass_name)
        if hasattr(passMod, 'export_torchscript'):
            export_pt = getattr(passMod, 'export_torchscript')
        else:
            assert False, 'There are not export_torchscript function in pass: {}'.format(pass_name)
        # loop all op
        pass_or_not = False
        while True:
            matched = False
            for op_name, op in operator_dict.items():
                if op.type == op_type:
                    pass_or_not = True
                    matched = True
                    
                    # -------run pass------
                    
                    # 1. export pt
                    
                    # get params and attr_dict
                    params_dict = ParseParams(op)
                    attrs_dict = ParseAttrs(op)
                    # update attrs_dict
                    for attrs_key, attrs_value in attrs_dict.items():
                        attrs_data = attrs_value['data']
                        attrs_shape = attrs_value['shape']
                        params_dict[attrs_key] = attrs_data.reshape(attrs_shape)
                    

                    # get src node info
                    input_names, input_shapes, input_datas, \
                        attr_input_names, attr_input_datas,\
                            output_names = get_src_node_info(op)

                    # export pt
                    all_params_dict = params_dict.copy()
                    all_params_dict['v_0'] = input_datas
                    all_params_dict['save_dir'] = pass_level7_tmp_output_path
                    all_params_dict['op_name'] = op_name
                    all_params_dict['attr_data'] = [torch.from_numpy(attr_input_data) for attr_input_data in attr_input_datas]
                    export_pt(**all_params_dict)

                    pass_pt_path = os.path.join(pass_level7_tmp_output_path, op_name + '.pt').replace('\\','/')
                    pass_input_shape_str = ','.join([str(inner_list) for inner_list in input_shapes]) 
                    pass_input_shape_str.replace(' ','')
                    # 2. export pnnx
                    cur_parser = PnnxParser()
                    pass_operators, pass_operands, pass_input_ops, pass_output_ops = cur_parser.getNvpPnnxModel(pass_pt_path, pass_input_shape_str)
                    pass_operators_dict, pass_operands_dict = trans_list_to_dict(pass_operators, pass_operands, True, op_name)
                    
                    attr_input_node_name = get_pre_node_name(operand_dict, attr_input_names)
                    del_op_names = [op.name] +  attr_input_node_name
                    for del_op_name in del_op_names:
                        operator_dict.pop(del_op_name) 

                    # insert pass op
                    input_index = 0
                    output_index = 0
                    for cur_pass_op_name, cur_pass_op in pass_operators_dict.items():
                    # for cur_pass_op in pass_operators:
                        if cur_pass_op.type == 'pnnx.Input':
                            # get src input operand
                            src_input_operand_name = input_names[input_index]
                            src_input_operand = operand_dict[src_input_operand_name]
                            # get src input node name
                            src_input_node_name = src_input_operand.producer.name
                            # get dst ops
                            cur_pass_input_operand = cur_pass_op.outputs[0]
                            cur_dst_ops = cur_pass_input_operand.consumers
                             
                            # src_input_operand connect new node
                            src_input_operand.consumers = [ consumers  for consumers in src_input_operand.consumers if consumers.name != op_name ] + cur_dst_ops
                            for dst_op in cur_dst_ops:
                                dsp_op_name = dst_op.name
                                # pass_operators_dict[dsp_op_name].inputs = 
                                pass_operators_dict[dsp_op_name].inputs = [ src_input_operand if d_input.name == cur_pass_input_operand.name else d_input for d_input in pass_operators_dict[dsp_op_name].inputs]
                            
                            # src_input node connect new node 
                            src_input_node = operator_dict[src_input_node_name]
                            for src_input_node_out in src_input_node.outputs:
                                src_input_node_out.consumers = [ out_cons for out_cons in src_input_node_out.consumers if out_cons.name != op_name] + cur_dst_ops
                            #
                            input_index += 1

                            
                        elif cur_pass_op.type == 'pnnx.Output':
                            src_output_name = output_names[output_index]
                            src_output_operand = operand_dict[src_output_name]
                            
                            dst_output_op = cur_pass_op.inputs[0].producer
                            src_output_operand.producer = dst_output_op
                            dst_output_op_name = dst_output_op.name
                            pass_operators_dict[dst_output_op_name].outputs = [src_output_operand]

                            # sink node connect new node 
                            src_output_node_names = [ con.name for con in src_output_operand.consumers]
                            for src_output_node_name in src_output_node_names:
                                for input_operand in operator_dict[src_output_node_name].inputs:
                                    if input_operand.producer.name == op_name:
                                        input_operand.producer.name = dst_output_op.name
                            output_index += 1
                        else:

                            operator_dict[cur_pass_op.name] = cur_pass_op
                    
                    #delect src attr operands
                    for attr_input_name in attr_input_names:
                        operand_dict.pop(attr_input_name)
                                
                    # insert pass operand
                    for cur_pass_operand in pass_operands:
                        if cur_pass_operand.producer.type != 'pnnx.Input' and cur_pass_operand.consumers[0].type != 'pnnx.Output':
                            
                            operand_dict[cur_pass_operand.name] = cur_pass_operand
                    # debug info
                    print('finish pass {} in {}'.format(pass_name, op_name))
                    break     

            if not matched:
                break
        
        if pass_or_not:
            # if pass valid visual
            visual_operators, visual_operands = trans_dict_to_list(operator_dict, operand_dict)
            # debug_op(operators)
            # debug_operand(operands)
            param_path = os.path.join(pass_level7_tmp_output_path, pass_name + '.pnnx.param')
            parser.SaveModel(param_path, visual_operators, visual_operands)

    # get finial 
    operators, operands = trans_dict_to_list(operator_dict, operand_dict)


                

                
       






     