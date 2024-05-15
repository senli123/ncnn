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
    
    # def pass_level7(self, pass_level7_path):