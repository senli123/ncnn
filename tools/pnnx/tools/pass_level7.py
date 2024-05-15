from serializer import *
import numpy as np
import struct
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
if __name__ == "__main__":
   
   
    parser = PnnxParser()
    pt_path_str = 'D:/project/programs/my_project/tests/test_python/test_op/model_zoo2/stack_16/stack_16.pt' 
    input_shape_str = '[1,3,224,224],[1,3,224,224]'
    # custom_op_path_str = 
    # infer_py_path = 
    pass_level7_path = 'D:/project/programs/ncnn_project/ncnn/tools/pnnx/pass_level7'
    # gen pnnx model
    operators, operands, input_ops, output_ops = parser.getNvpPnnxModel(pt_path_str, input_shape_str)
    pass_level7_tmp_output_path = 'D:/project/programs/ncnn_project/ncnn/tools/pnnx/output/tmp'
    # loop all pass level7
    all_pass_files = os.listdir(pass_level7_path)
    all_pass_files = [pass_file for pass_file in all_pass_files if pass_file not in ['__init__.py'] and not os.path.isdir(os.path.join(pass_level7_path, pass_file))]
    for pass_file in all_pass_files:
        pass_name, _ = os.path.splitext(pass_file)
        print("run pass:{}".format(pass_name))
        passMod = load_module(os.path.join(pass_level7_path, pass_file))
        op_type = getattr(passMod, 'op_type')
        export_pt = getattr(passMod, 'export_torchscript')
        # loop all op
        for index, op in enumerate(operators):
            if op.type == op_type:
                op_name = op.name
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
                
                # gen input 
                input_names, input_shapes, input_datas,src_input_Operands = [], [], [],[]
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
                        src_input_Operands.append(operand)
                
                outOperands = op.outputs
                output_names = [operand.name for operand in outOperands]
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
                pass_operators, pass_operands, pass_input_ops, pass_output_ops = parser.getNvpPnnxModel(pass_pt_path, pass_input_shape_str)
                
                
                # delect cur op
                # delete cur input attribute
                del_op_indexs = []
                for cur_index, cur_op in enumerate(operators):
                    if len(cur_op.outputs) == 0:
                        continue
                    if cur_op.outputs[0].name in attr_input_names: 
                        del_op_indexs.append(cur_index)
                del_op_indexs.append(index)
                del_op_indexs.sort(reverse=True)  
                for cur_attr_op_index in del_op_indexs:   
                    del operators[cur_attr_op_index]

                # insert pass op
                input_index = 0
                output_index = 0
                for cur_pass_op in pass_operators:
                    if cur_pass_op.type == 'pnnx.Input':
                        # get src input operand
                        src_input_operand_name = input_names[input_index]
                        src_input_operand = src_input_Operands[input_index]
                        # get dst ops
                        cur_pass_input_operand = cur_pass_op.outputs[0]
                        cur_dst_ops = cur_pass_input_operand.consumers
                        #update cur_dst_ops name
                        for cur_dst_op in cur_dst_ops:
                            cur_dst_op.name = op_name + '_' + cur_dst_op.name 
                        # src_input_operand connect new node
                        src_input_operand.consumers = [ consumers  for consumers in src_input_operand.consumers if consumers.name != op_name ] + cur_dst_ops
                        for dst_op in cur_dst_ops:
                            dst_op.inputs = [ src_input_operand if d_input.name == cur_pass_input_operand else d_input for d_input in dst_op.inputs]
                        input_index += 1
                    elif cur_pass_op.type == 'pnnx.Output':
                        src_output_name = output_names[output_index]
                        src_output_operand = outOperands[output_index]
                        src_output_operand_consumers = src_output_operand.consumers
                        dst_output_op = cur_pass_op.inputs[0].producer
                        src_output_operand.producer = dst_output_op
                        dst_output_op.outputs = [src_output_operand]
                        output_index += 1
                    else:
                        #update name
                        cur_pass_op.name = op_name + '_' + cur_pass_op.name
                        #update inputs outputs name
                        for i in range(len(cur_pass_op.inputs)):
                            cur_pass_op.inputs[i].name = op_name + '_' + cur_pass_op.inputs[i].name
                        for j in range(len(cur_pass_op.outputs)):
                            cur_pass_op.outputs[j].name = op_name + '_' + cur_pass_op.outputs[j].name
                       
                        operators.append(cur_pass_op)
                
                #delect src attr operands
                del_operand_indexs = []
                for cur_operand_index, cur_operand in enumerate(operands):
                    if cur_operand.name in attr_input_names: 
                        del_operand_indexs.append(cur_operand_index)

                del_operand_indexs.sort(reverse=True)  
  
                for attr_input_operand_index in del_operand_indexs:  
                    del operands[attr_input_operand_index]  
                            
                # insert pass operand
                for cur_pass_operand in pass_operands:
                    if cur_pass_operand.producer.type != 'pnnx.Input' and cur_pass_operand.consumers[0].type != 'pnnx.Output':
                        operands.append(cur_pass_operand)

                # operand.name = op_name + '_' + operand.name

                
                # # update all_operands name
                # for operand in pass_operands:
                #     operand.name = op_name + '_' + operand.name
                #     # operand.producer.name =  op_name + '_' + operand.producer.name
                #     # for consumer in operand.consumers:
                #     #     consumer.name = op_name + '_' + consumer.name

                # # update all_opers name
                # for operator in pass_operators:
                #     operator.name = op_name + '_' + operator.name
                #     # for ii in operator.inputs:
                #     #     ii.name = op_name + '_' + ii.name
                #     # for oo in operator.outputs:
                #     #     oo.name = op_name + '_' + oo.name

                # for pass_input_index, operator in enumerate(pass_input_ops):
                #     pass_input_ops[pass_input_index].name =  op_name + '_' + pass_input_ops[pass_input_index].name
                #     for oo_index, oo in enumerate(operator.outputs):
                #         pass_input_ops[pass_input_index].outputs[oo_index].name = op_name + '_' + pass_input_ops[pass_input_index].outputs[oo_index].name
                       

                # for pass_output_index, operator in enumerate(pass_output_ops):
                #     pass_output_ops[pass_output_index].name =  op_name + '_' + pass_output_ops[pass_output_index].name
                #     for ii_index, ii  in enumerate(operator.inputs):
                #         pass_output_ops[pass_output_index].inputs[ii_index].name = op_name + '_' + pass_output_ops[pass_output_index].inputs[ii_index].name

                # # connect pass_operators to operator

                # input_index  = 0
                # for pass_input_op in pass_input_ops:
                #     cur_input_op_name = pass_input_op.name
                #     cur_input_op_type = pass_input_op.type
                #     if cur_input_op_name == '' or cur_input_op_type == 'pnnx.Attribute':
                #         continue
                #     input_operand = pass_input_op.outputs[0]
                #     input_operand_name = input_operand.name
                #     # dst ops
                #     dst_ops = input_operand.consumers
                #     src_input_operand_name = input_names[input_index]
                #     input_index += 1
                #     # src_input_operand connect new node
                #     src_input_operand = operands_dict[src_input_operand_name]
                #     src_input_operand.consumers = [ consumers  for consumers in src_input_operand.consumers if consumers.name != op_name ] + dst_ops
                #     for dst_op in dst_ops:
                #         dst_op.inputs = [ src_input_operand if d_input.name == input_operand_name else d_input for d_input in dst_op.inputs]

                    
                # # src_output_operad connect new node
                # pass_output_ops_list = [] 
                # for cur_output_op in pass_output_ops:
                #     cur_output_op_name = cur_output_op.name
                #     if cur_output_op_name == '':
                #         continue
                #     pass_output_ops_list.append(cur_output_op)
                
                # for src_output_name, pass_output_op in zip(output_names, pass_output_ops_list):
                #     src_output_operand = operands_dict[src_output_name]
                #     src_output_operand_consumers = src_output_operand.consumers
                #     dst_output_op = pass_output_op.inputs[0].producer
                #     src_output_operand.producer = dst_output_op
                #     dst_output_op.outputs = [src_output_operand]

                
                # #delect src op
                # del operators[index]

                # #delect src attribute
                # for attr_input_name in attr_input_names:
                #     for operand_index, operand in enumerate(       ):
                #         if operand.name == attr_input_name:
                #             attribute_obj = operators_dict[operand.producer.name]
                #             attribute_index = attribute_obj['index']
                #             del operators[attribute_index]
                #             del operands[operand_index]
                #             operators_dict
                #             break
                
                
                # #add new op
                # for pass_op in pass_operators:
                #     if pass_op.type not in ['pnnx.Input','pnnx.Output']:
                #         operators.append(pass_op)

                # # add new operands
                # for pass_operand in pass_operands:
                #     if pass_operand.producer.type not in ['pnnx.Input'] and pass_operand.consumers[0].type not in['pnnx.Output']:
                #         operands.append(pass_op)

                # debug info
                print('finish pass {} in {}'.format(pass_name, op_name))

        # finish pass to visual 
        # print(operators)
        # print(operands)
        # print(input_ops)
        # print(output_ops)
        debug_op(operators)
        debug_operand(operands)



                

                
       






     