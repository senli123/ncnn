import sys
from typing import List, Union, Optional
import argparse
import os
import shutil
import json
import importlib
import re
try:
	import torch
	import torchvision.models as models
	import torch.nn as nn
	import torch.nn.functional as F
	sys.path.append('D:/project/programs/ncnn_project/ncnn/tools/pnnx/python/build/lib.win-amd64-cpython-38/pnnx')
	# sys.path.append('/workspace/trans_onnx/project/new_project/ncnn/tools/pnnx/python/build/temp.linux-x86_64-cpython-311/src')
	import ptx
	graph = ptx.PnnxGraph()
except ImportError as e:
	sys.exit(str(e))

def extract_content_between_parentheses(text):
    pattern = r'\((.*?)\)'
    matches = re.findall(pattern, text)
    return matches

class PnnxParser():
    
    def __init__(self,):
        """pnnx ir description

            Operator:
                intputs: list(Operand)
                outputs: list(Operand)
                type: str
                name: str
                inputnames: list(str)
                params: dict{str:Parameter}
                attrs: dict{str:Attribute}

            Operand:
                producer: Operator
                consumers: list(Operator)
                type: int  // 0=null 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8 9=bool 10=cp64 11=cp128 12=cp32
                shape: list(int)
                name: str
                params: dict{str:Parameter}
                attrs: dict{str:Attribute}
            
            Attribute:
                type: int // 0=null 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8 9=bool
                shape: list(int)
                data: list(char)
            Parameter:
                type: int  //0=null 1=b 2=i 3=f 4=s 5=ai 6=af 7=as 8=others
                b: bool type==1
                i: int type==2
                f: float type==3
                ai: list(int) type==4
                af: list(float) type==5
                s: str type==6
                as: list(str) type==7

        """
    
    def LoadModel(self, params_path: str, bin_path: str):
        """

        Args:
            params_path (str): the path of pnnx.params
            bin_path (str): the path of pnnx.bin

        Returns:
            operators list(Operator)
            operands list(Operand)
            input_ops list(Operator)
            output_ops list(Operator)
        """
        a = graph.loadModel(params_path,bin_path)
        assert(a is True, "please check your you input path")
        operators = graph.getOperators()
        operands = graph.getOperands()
        input_ops = graph.getInputOps()
        output_ops = graph.getOutputOps()
        return operators, operands, input_ops, output_ops
    
    def getNvpPnnxModel(self, pt_path_str: str, input_shape_str: str, custom_op_path_str: str = 'None', infer_py_path: str = 'None'):
        """_summary_

        Args:
            pt_path_str (str): the path of pt
            input_shape_str (str): the shape of input
            custom_op_path_str (str, optional): the path of custom op. Defaults to 'None'.
            infer_py_path (str, optional): the path of exeutor. Defaults to 'None'.

        Returns:
            operators list(Operator)
            operands list(Operand)
            input_ops list(Operator)
            output_ops list(Operator)
        """
        result = graph.getNvpPnnxModel(pt_path_str, input_shape_str, custom_op_path_str, infer_py_path)
        assert(result, "get pnnx model failed")
        params_path = pt_path_str.replace('.pt','.pnnx.param')
        bin_path = pt_path_str.replace('.pt','.pnnx.bin')
        return self.LoadModel(params_path, bin_path)
    
    def extract(self, params_path: str, infer_path: str, output_path :str, start_tensor_name: str, end_tensor_name: str = 'None'):
        """T he first version of the subgraph extraction function \
            mainly records the information of the starting row and the ending row to construct the subgraph.  

        Args:
            params_path (str): the path of params file
            infer_path (str): the path of infer py 
            output_path (str): the output path of new params file and new infer py
            start_tensor_name (str): the name of input tensor in sub graph  
            end_tensor_name (str): the name of output tensor in sub graph. Defaults to 'None'.
        """
        assert os.path.exists(params_path), 'the path of params: {} is not exist'.format(params_path)
        assert os.path.exists(infer_path), 'the path of infer py: {} is not exist'.format(infer_path)
        assert os.path.isdir(output_path), 'the path of output: {} is not exist'.format(output_path)
        with open(params_path, mode= 'r') as f:
            params_lines = f.readlines()
        input_shape = ''
        output_shape = '' 
        start_index = -1
        end_index = -1
        adjacency_matrix_dict = {}
        for index, line in enumerate(params_lines):
            if index < 2:
                 continue
            print(line.split())
            op_info_list = line.split()
            op_name = op_info_list[1]
            input_nums = op_info_list[2]
            output_nums = op_info_list[3]
            input_name_list = []
            for i in range(int(input_nums)):
                input_name_list.append(op_info_list[4 + i])
            output_name_list = []
            for j in range(int(output_nums)):
                output_name_list.append(op_info_list[4 + int(input_nums) + j]) 

            if len(input_name_list) == 0:
                # input node or attribute
                if start_index != -1:
                    adjacency_matrix_dict[output_name] = {'input_names':input_name_list, 'index_info':[index]}
                continue
            if start_tensor_name == input_name_list[0] and len(input_name_list) == 1:
                start_index = index
                #get input shape
                tensor_shape = []
                for op_info in op_info_list:
                    if op_info.startswith('#'):
                        tensor_shape.append(op_info)
                input_shape = tensor_shape[0]
            
            if end_tensor_name == output_name_list[0] and len(output_nums) == 1:
                end_index = index
                for output_name in output_name_list:
                    adjacency_matrix_dict[output_name] = {'input_names':input_name_list, 'index_info':[index]}
                #get output shape
                tensor_shape = []
                for op_info in op_info_list:
                    if op_info.startswith('#'):
                        tensor_shape.append(op_info)
                output_shape = tensor_shape[-1]
                
                
                break
            if start_index !=-1:
                for output_name in output_name_list:
                    if output_name in adjacency_matrix_dict:
                        adjacency_matrix_dict[output_name]['input_names'].extend(input_name_list)
                        adjacency_matrix_dict[output_name]['index_info'].append(index)
                    else:
                        adjacency_matrix_dict[output_name] = {'input_names':input_name_list, 'index_info':[index]}
                
        if start_index > end_index or start_index == -1 or end_index == -1:
            assert False, 'please check your start_tensor_name and end_tensor_name!'
       
        # from end node to start node to get real sub graph
        cur_input_name_list = adjacency_matrix_dict[end_tensor_name]['input_names']
        cur_index_info_list = adjacency_matrix_dict[end_tensor_name]['index_info']
 
        tmp_index_list = []

        def backtrack(cur_input_name_list, cur_index_info_list):
            if start_tensor_name in cur_input_name_list:
                tmp_index_list.extend(cur_index_info_list)
                return
            for cur_input in cur_input_name_list:
                if cur_input in adjacency_matrix_dict:
                    new_cur_input_name_list = adjacency_matrix_dict[cur_input]['input_names']
                    index_info_list = adjacency_matrix_dict[cur_input]['index_info']
                    new_cur_index_info_list =cur_index_info_list.copy()
                    new_cur_index_info_list.extend(index_info_list)
                    backtrack(new_cur_input_name_list,new_cur_index_info_list)

        backtrack(cur_input_name_list,cur_index_info_list)
                
        #sorted
        unique_list = list(set(tmp_index_list))  
        sorted_list = sorted(unique_list)  
        new_params_lines = [params_lines[i] for i in sorted_list]
        #get sub_graph all tensor
        sub_graph_all_tensor = []
        for new_param_line in new_params_lines:
            new_op_info_list = new_param_line.split()
            new_op_name = new_op_info_list[1]
            new_op_input_nums = new_op_info_list[2]
            new_op_output_nums = new_op_info_list[3]
            new_op_output_name_list = []
            for j in range(int(new_op_output_nums)):
                new_op_output_name_list.append(new_op_info_list[4 + int(new_op_input_nums) + j]) 
                sub_graph_all_tensor.extend(new_op_output_name_list)
        sub_graph_all_tensor = list(set(sub_graph_all_tensor))   
        #insert output node
        input_node = 'pnnx.Input               pnnx_input_0             0 1 {} {}\n'.format(start_tensor_name, input_shape)
        output_node = 'pnnx.Output              pnnx_output_0            1 0 {} {}\n'.format(end_tensor_name, output_shape)
        new_params_lines.insert(0,input_node)
        new_params_lines.append(output_node)
        ops_nums = len(new_params_lines)
        tensor_nums = ops_nums - 1
        new_params_lines.insert(0,'{} {}\n'.format(ops_nums,tensor_nums))
        new_params_lines.insert(0,'7767517\n')
        new_params_file_path = os.path.join(output_path,'{}-{}_model.pnnx.param'.format(start_tensor_name, end_tensor_name))
        with open(new_params_file_path, 'w') as file:  
            file.writelines(new_params_lines)
        new_infer_file_path = os.path.join(output_path,'{}-{}_model_pnnx_infer.py'.format(start_tensor_name, end_tensor_name))
        self.Extract_forward(infer_path, start_tensor_name, end_tensor_name, input_shape, new_infer_file_path,sub_graph_all_tensor)

    def Extract_forward(self, infer_path, start_tensor_name, end_tensor_name, input_shape, new_infer_file_path, sub_graph_all_tensor):
        
        with open(infer_path, mode= 'r') as f:
            infer_lines = f.readlines()
        #get input shape line index
        #get forwrad start_index
        #get forward end_index
        #get start_tensor_index
        #get end_tensor_index
        #get output_tensor_index
        for infer_line_index, infer_line_info in enumerate(infer_lines):
            infer_line_info_list = infer_line_info.split()
            if infer_line_info_list == ['def','getInput(self,):']:
                input_shape_line_index = infer_line_index + 1
            elif infer_line_info_list == ['def', 'forward(self,', 'v_0):']:
                forwrad_start_index = infer_line_index
            elif infer_line_info_list == ['return', 'intermediate']:
                forwrad_end_index = infer_line_index
            elif len(infer_line_info_list) > 1 and infer_line_info_list[0] == 'v_' + start_tensor_name: 
                start_tensor_index = infer_line_index + 1
            elif len(infer_line_info_list) > 1 and infer_line_info_list[0] == 'v_' + end_tensor_name: 
                end_tensor_index = infer_line_index
            elif infer_line_info_list == ['if', 'self.infer_flag:']:
                output_tensor_index = infer_line_index + 1
        
        #update input shape
        infer_lines[input_shape_line_index] = '        return [[' + extract_content_between_parentheses(input_shape)[0] + ']]\n'
        #update forward tensor name
        infer_lines[forwrad_start_index] ='    def forward(self, v_{}):\n'.format(start_tensor_name)
        #update output_tensor_index
        infer_lines[output_tensor_index] ='            return v_{}\n'.format(end_tensor_name)
        #get sub graph forward
        sub_graph_line_indexs = []
        for forward_index in range(start_tensor_index,end_tensor_index + 1):
            o_tensor = infer_lines[forward_index].split()[0].split('_')[1]
            if o_tensor in sub_graph_all_tensor:
                sub_graph_line_indexs.append(forward_index)
        sub_graph_line_indexs = list(set(sub_graph_line_indexs))  
        sub_graph_line_indexs = sorted(sub_graph_line_indexs)  
        #get new infer_lines
        sub_graph_lines = [infer_lines[i] for i in sub_graph_line_indexs]
        new_infer_lines = infer_lines[:forwrad_start_index + 1] + sub_graph_lines + \
        infer_lines[output_tensor_index - 1: ]
        with open(new_infer_file_path, 'w') as file:  
            file.writelines(new_infer_lines)


if __name__ == "__main__":

    # line = ' def forward(self, v_0):\n'
    # print(line.split())
    
    parser = PnnxParser()
    params_path = "D:/project/model_zoo/icnet/model.pnnx.param"
    infer_path = "D:/project/model_zoo/icnet/model_pnnx_infer.py"
    output_path = "D:/project/model_zoo/icnet/extract"
    start_tensor_name  = '144'
    end_tensor_name = '145'
    parser.extract(params_path, infer_path, output_path, start_tensor_name, end_tensor_name)


    # with open('conv.pnnx.param', mode = 'w') as f:
    #     fprintf(paramfp, "%-24s %-24s %d %d", op->type.c_str(), op->name.c_str(), (int)op->inputs.size(), (int)op->outputs.size());
    #             with open(template_output_path, 'w') as file:  
    #         file.writelines(original_lines)
    # with open('conv.pnnx.param', "w", encoding="utf-8") as f:
    #       f.write("import os")
    # a = [1,3,4,5]
    # b = [23]
    # c = b.copy()
    # c.extend(a)
    # print(a)
    # print(b)
    # print(c)
    pass