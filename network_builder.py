import json
from utils.LayerData import Layers, Loss, Optimizer
template = '''
import torch.nn as nn
from torch import optim
{user_info}
class Model(nn.Module):
\tdef __init__(self, *args, **kwargs) -> None:
\t\tsuper().__init__(*args, **kwargs)
{layers}
\tdef forward(self,x):
{forward}
\t\treturn x
'''
preprocessor_template = '''
from torchvision import transforms
pre_processor = transforms.Compose([\n {layers} \n])
'''
FILE_PATH=r"C:\Users\prata\code\misc\NNBuilder\config.json"

class NetworkBuilder:
    def __init__(self, file_path:str)->None:
        self.file_path = file_path
    def read(self):
        with open(self.file_path,"r") as f:
            self.data = json.load(f)
    def build(self)->None:
        self.layers = []
        self.forward = [] 
        for idx,layer in enumerate(self.data.get('layers')):
            type = layer.get('type')
            input = layer.get('in')
            output = layer.get('out')
            kernel = layer.get('kernel')
            stride = layer.get('stride')
            if type == Layers.CONV2D:
                line = f"\t\tself.layer{idx} = nn.Conv2d({input},{output},{kernel},{stride})"
            elif type == Layers.LIN:
                line = f"\t\tself.layer{idx} = nn.Linear({input}, {output})"
            elif type == Layers.MAXPOOL2D:
                line = f"\t\tself.layer{idx} = nn.MaxPool2d({kernel},{stride})"
            elif type == Layers.RELU:
                line = f"\t\tself.layer{idx} = nn.ReLU()"
            else:
                line=""
            self.layers.append(line)

        for idx, layer in enumerate(self.data.get('layers')):
            line = f"\t\tx = self.layer{idx}(x)"
            self.forward.append(line)
    def save(self):
        formatted_string = template.format(user_info=f"#Designed by {self.data.get('user')}",layers="\n".join(self.layers),forward="\n".join(self.forward))
        with open("network.py","w") as f:
            f.write(formatted_string)

class ImagePreProcessorBuilder:
    def __init__(self, file_path:str)->None:
        self.file_path = file_path
    def read(self):
        with open(self.file_path,"r") as f:
            self.data = json.load(f)
    def build(self):
        self.pre_processors = []
        statement_template = "transforms.{layer},\n"
        for idx,layer in enumerate(self.data.get('data')):

        

if __name__=="__main__":
    builder = NetworkBuilder(FILE_PATH)
    builder.read()
    builder.build()
    builder.save()

