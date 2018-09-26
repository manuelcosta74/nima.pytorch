# -*- coding: utf-8 -*-
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import torch
import torch.onnx
from torch.autograd import Variable

from nima.inference.inference_model import InferenceModel


def convert_pth2onnx(path_to_src_pth, path_to_dest_onnx, vb = False):
    
        
    nimaModel = InferenceModel(path_to_model=path_to_src_pth)

    
    dummy_input = Variable(torch.randn(1, 3, 224, 224))
    
    torch.onnx.export(nimaModel.model, dummy_input, path_to_dest_onnx, verbose = vb)
