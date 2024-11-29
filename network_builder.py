import json
from utils.LayerData import Layers, Loss, Optimizer
template = '''
import torch.nn as nn
from torch import optim
{user_info}
class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
{layers}
    def forward(self,x):
{forward}
        return x
'''
with open(r"C:\Users\prata\code\misc\NNBuilder\config.json","r") as f:
    data = json.load(f)


layers = []

for idx,layer in enumerate(data.get('layers')):
    type = layer.get('type')
    input = layer.get('in')
    output = layer.get('out')
    kernel = layer.get('kernel')
    stride = layer.get('stride')
    if type == Layers.CONV2D:
        line = f"\t\tself.layer{idx} = nn.Conv2D({input},{output},{kernel},{stride})"
    elif type == Layers.LIN:
        line = f"\t\tself.layer{idx} = nn.Linear({input}, {output})"
    elif type == Layers.MAXPOOL2D:
        line = f"\t\tself.layer{idx} = nn.MaxPool2D({kernel},{stride})"
    elif type == Layers.RELU:
        line = f"\t\tself.layer{idx} = nn.ReLU()"
    else:
        line=""
    layers.append(line)





formatted_string = template.format(user_info=f"#Designed by {data.get('user')}",layers="\n".join(layers),forward="")
print(formatted_string)

