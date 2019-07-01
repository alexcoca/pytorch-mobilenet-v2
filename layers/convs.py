import torch.nn as nn

def conv_bn(inp, oup, stride):
    r""" This is the layer that needs to be changed """
  
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_bn_combined(inp, oup, stride):
    r""" Layer that has been changed to include batch normalisation in the weights"""
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1),
        nn.ReLU6(inplace=True)
)

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )