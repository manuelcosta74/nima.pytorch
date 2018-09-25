# -*- coding: utf-8 -*-

import torch
import torch.onnx
from torchvision.datasets.folder import default_loader
from nima.common import Transform

from nima.inference.inference_model import InferenceModel


def convert_pth2onnx(path_to_src_pth, path_to_dest_onnx, verbose = False):
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
        
    nimaModel = InferenceModel(path_to_model=path_to_src_pth)

    transform = Transform().val_transform

    image = default_loader("Dummy.jpg")
    image = transform(image)
    image = image.unsqueeze_(0)
    image = image.to(device)

    with torch.no_grad():
        image = torch.autograd.Variable(image)

    torch.onnx.export(nimaModel.model, image, verbose=True)
