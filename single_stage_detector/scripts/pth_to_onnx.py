
#!/usr/bin/env python3
import argparse

import torch
import torch.onnx
import torchvision
from torch.autograd import Variable

from model.retinanet import retinanet_from_backbone

def parse_args(add_help=True):
    parser = argparse.ArgumentParser(description='Convert PyTorch detection file to onnx format', add_help=add_help)

    parser.add_argument('--input', required=True, help='input pth file')
    parser.add_argument('--output', default=None, help='output onnx file')

    parser.add_argument('--backbone', default='resnext50_32x4d',
                        choices=['resnet50', 'resnext50_32x4d', 'resnet101', 'resnext101_32x8d'],
                        help='The model backbone')
    parser.add_argument('--num-classes', default=264, type=int,
                        help='Number of detection classes')
    parser.add_argument('--trainable-backbone-layers', default=3, type=int,
                        help='number of trainable layers of backbone')

    parser.add_argument('--image-size', default=None, nargs=2, type=int,
                        help='Image size for training. If not set then will be dynamic')
    parser.add_argument('--batch-size', default=None, type=int,
                        help='input batch size. if not set then will be dynamic')
    parser.add_argument('--data-layout', default="channels_first", choices=['channels_first', 'channels_last'],
                        help="Model data layout")
    parser.add_argument('--device', default='cuda', help='device')

    args = parser.parse_args()

    args.output = args.output or ('retinanet_'+args.backbone+'.onnx')
    return args

def main(args):
    batch_size = args.batch_size or 1
    image_size = args.image_size or [800, 800]

    print("Creating model")
    model = retinanet_from_backbone(backbone=args.backbone,
                                    num_classes=args.num_classes,
                                    image_size=image_size,
                                    data_layout=args.data_layout,
                                    pretrained=False,
                                    trainable_backbone_layers=args.trainable_backbone_layers)
    device = torch.device(args.device)
    model.to(device)

    print("Loading model")
    checkpoint = torch.load(args.input)
    model.load_state_dict(checkpoint['model'])

    print("Creating input tensor")
    rand = torch.randn(batch_size, 3, image_size[0], image_size[1],
                       device=device,
                       requires_grad=False,
                       dtype=torch.float)
    inputs = torch.autograd.Variable(rand)
    # Output dynamic axes
    dynamic_axes = {
        'boxes': {0 : 'num_detections'},
        'scores': {0 : 'num_detections'},
        'labels': {0 : 'num_detections'},
    }
    # Input dynamic axes
    if (args.batch_size is None) or (args.image_size is None):
        dynamic_axes['images'] = {}
        if args.batch_size is None:
            dynamic_axes['images'][0]: 'batch_size'
        if args.image_size is None:
            dynamic_axes['images'][2] = 'width'
            dynamic_axes['images'][3] = 'height'


    print("Exporting the model")
    model.eval()
    torch_outs = model(inputs) # For testing
    torch.onnx.export(model,
                      inputs,
                      args.output,
                      export_params=True,
                      opset_version=13,
                      do_constant_folding=False,
                      input_names=['images'],
                      output_names=['boxes', 'scores', 'labels'],
                      dynamic_axes=dynamic_axes)

    check_onnx_model(inputs, torch_outs, args.output)


def check_onnx_model(inputs, torch_outs, onnx_model_path):
    # For testing, may need to install onnx, onnxruntime and update protobuf version for it to work
    # apt-get update
    # pip install onnx onnxruntime
    # pip install protobuf==3.20.*
    import onnx, onnxruntime
    import numpy as np
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    print("Pytorch Results \n ", torch_outs)
    print("Onnx Results \n ", ort_outs)
    np.testing.assert_allclose(to_numpy(torch_outs[0]['labels']), ort_outs[2], rtol=1e-02, atol=1e-03)
    np.testing.assert_allclose(to_numpy(torch_outs[0]['scores']), ort_outs[1], rtol=1e-02, atol=1e-03)
    np.testing.assert_allclose(to_numpy(torch_outs[0]['boxes']), ort_outs[0], rtol=1e-01, atol=1e-01)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

if __name__ == "__main__":
    args = parse_args()
    main(args)
