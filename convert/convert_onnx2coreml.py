# -*- coding: utf-8 -*-

import copy
import coremltools

from onnx_coreml import convert
from onnx import onnx_pb
from coremltools.models import MLModel

"""
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

"""

def convert_onnx2coreml(path_to_src_onnx, path_to_dest_mlmodel):
    
    red_b = -(0.485 * 255.0)
    green_b = -(0.456 * 255.0)
    blue_b = -(0.406 * 255.0)
    
    red_scale = 1.0 / (0.229 * 255.0)
    green_scale = 1.0 / (0.224 * 255.0)
    blue_scale = 1.0 / (0.225 * 255.0)

    args = dict(is_bgr=False, red_bias = red_b, green_bias = green_b, blue_bias = blue_b)

            
    model_file = open(path_to_src_onnx, 'rb')
    model_proto = onnx_pb.ModelProto()
    model_proto.ParseFromString(model_file.read())
    coreml_model = convert(model_proto, image_input_names=['0'], preprocessing_args=args)

    spec = coreml_model.get_spec()

    # get NN portion of the spec
    nn_spec = spec.neuralNetwork
    layers = nn_spec.layers # this is a list of all the layers
    layers_copy = copy.deepcopy(layers) # make a copy of the layers, these will be added back later
    del nn_spec.layers[:] # delete all the layers

    # add a scale layer now
    # since mlmodel is in protobuf format, we can add proto messages directly
    # To look at more examples on how to add other layers: see "builder.py" file in coremltools repo
    scale_layer = nn_spec.layers.add()
    scale_layer.name = 'scale_layer'
    scale_layer.input.append('0')
    scale_layer.output.append('0_scaled')

    params = scale_layer.scale
    params.scale.floatValue.extend([red_scale, green_scale, blue_scale]) # scale values for RGB
    params.shapeScale.extend([3,1,1]) # shape of the scale vector 
    #params.bias.floatValue.extend([-(0.485 * 255.0), -(0.456 * 255.0), -(0.406 * 255.0)])
    #params.shapeBias.extend([3,1,1])

    # now add back the rest of the layers (which happens to be just one in this case: the crop layer)
    nn_spec.layers.extend(layers_copy)

    # need to also change the input of the crop layer to match the output of the scale layer
    nn_spec.layers[1].input[0] = '0_scaled'


    #os.chdir("/Users/Storyo/Documents/Code/python/convert_onnx2coreml/")

    coreml_model = coremltools.models.MLModel(spec)

    #coreml_model.save(path_to_dest_mlmodel)
    
    coremltools.utils.rename_feature(spec, '0', 'input')
    coremltools.utils.rename_feature(spec, '487', 'output')
    
    coremltools.utils.save_spec(spec, path_to_dest_mlmodel)