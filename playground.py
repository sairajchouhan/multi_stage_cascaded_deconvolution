import torch
from torch import nn
from collections import OrderedDict


DenseNet161 = torch.hub.load("pytorch/vision:v0.10.0", "densenet161", pretrained=True)

# newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))


# remove last layer
model = nn.Sequential(*list(DenseNet161.children())[:-1])


print(model)
# newmodel = torch.nn.Sequential(*(list(model.children())[0][:-1]))
