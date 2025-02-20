import json
from utils.LayerData import Layers, Loss, Optimizer, Transforms
from utils.Templates import Blocks
template = '''
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
{user_info}
{dependency}
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
FILE_PATH=r"C:\Users\prata\code\misc\NNBuilder\classfier_config.json"

class NetworkBuilder:
    def __init__(self, file_path:str)->None:
        self.file_path = file_path
    def read(self):
        with open(self.file_path,"r") as f:
            self.data = json.load(f)
    def build(self)->None:
        self.layers = []
        self.forward = [] 
        self.dependency = False
        for idx,layer in enumerate(self.data.get('layers')):
            type = layer.get('type')
            if type == Layers.CONV2D:
                line = f"\t\tself.layer{idx} = nn.Conv2d({layer.get('in')},{layer.get('out')},{layer.get('kernel')},{layer.get('stride')})"
            elif type == Layers.LIN:
                line = f"\t\tself.layer{idx} = nn.Linear({layer.get('in')}, {layer.get('out')})"
            elif type == Layers.MAXPOOL2D:
                line = f"\t\tself.layer{idx} = nn.MaxPool2d({layer.get('kernel')},{layer.get('stride')})"
            elif type == Layers.RELU:
                line = f"\t\tself.layer{idx} = nn.ReLU()"
            elif type == Layers.FLATTEN:
                line = f"\t\tself.layer{idx} = nn.Flatten(start_dim = {layer.get('start')}, end_dim = {layer.get('end')})"
            elif type == Layers.TRANSFORMER:
                line = f"""\t\tself.layer{idx} = nn.Transformer(d_model = {layer.get('d_model')},
                nhead = {layer.get('nhead')},
                num_encoder_layers = {layer.get('num_encoder_layers')},
                num_decoder_layers = {layer.get('num_decoder_layers')},
                dim_feedforward = {layer.get('dim_feedforward')},
                dropout = {layer.get('dropout')},
                batch_first = {True})"""
            elif type == Layers.EMBEDDING:
                line = f"\t\tself.layer{idx} = nn.Embedding(num_embeddings = {layer.get('num_embeddings')}, embedding_dim = {layer.get('embedding_dim')})"
            elif type == Layers.RESBLOCK:
                self.dependency = True
                line = f"\t\tself.layer{idx} = ResidualBlock(in_channels={layer.get('in')},out_channels={layer.get('out')},stride = {layer.get('stride')})"
            elif type == Layers.DROPOUT:
                line = f"\t\tself.layer{idx} = nn.Dropout(p = {layer.get('prob')})"
            elif type == Layers.BATCHNORM1D:
                line = f"\t\tself.layer{idx} = nn.BatchNorm1d(num_features = {layer.get('num_features')})"
            elif type == Layers.BATCHNORM2D:
                line = f"\t\tself.layer{idx} = nn.BatchNorm2d(num_features = {layer.get('num_features')})"
            elif type == Layers.GAPOOL2D:
                line = f"\t\tself.layer{idx} = nn.AdaptiveAvgPool2d({layer.get('out_shape')})"
            else:
                line=""
            self.layers.append(line)

        for idx, layer in enumerate(self.data.get('layers')):
            line = f"\t\tx = self.layer{idx}(x)"
            self.forward.append(line)
    def save(self):
        formatted_string = template.format(
            user_info=f"#Designed by {self.data.get('user')}",
            dependency = "#No dependency" if not self.dependency else Blocks.residual_block,
            layers="\n".join(self.layers),
            forward="\n".join(self.forward))
        with open("network.py","w") as f:
            f.write(formatted_string)

class ImagePreProcessorBuilder:
    def __init__(self, file_path:str, output_path:str)->None:
        self.file_path = file_path
        self.output_path = output_path
    def read(self):
        with open(self.file_path,"r") as f:
            self.data = json.load(f)
    def build(self):
        self.pre_processors = []
        for idx,layer in enumerate(self.data.get('data')):
            # print(layer.get('param1') != None and layer.get('param2') != None and layer.get("type")=="Resize")
            if layer.get('param1') == None and layer.get('param2') == None:
                params = " "
            elif layer.get('param1') == None and layer.get('param2') != None:
                params = f"{layer.get('param2')}"
            elif layer.get('param1') != None and layer.get('param2') == None:
                params = f"{layer.get('param1')}"
            elif layer.get('param1') != None and layer.get('param2') != None:
                params = f"{layer.get('param1')}, {layer.get('param2')}"
            if layer.get('param1') != None and layer.get('param2') != None and layer.get("type")=="Resize":
                params = f"({layer.get('param1')}, {layer.get('param2')})"
                
            self.pre_processors.append(
                f"\ttransforms.{Transforms.pre_processors[layer.get('type')]}({params}),"
                )
    def save(self):
        formatted_string = preprocessor_template.format(layers = "\n".join(self.pre_processors))
        with open(self.output_path,"w") as f:
            f.write(formatted_string)

        

if __name__=="__main__":
    builder = NetworkBuilder(FILE_PATH)
    builder.read()
    builder.build()
    builder.save()
    # builder = ImagePreProcessorBuilder(r"C:\Users\prata\code\misc\NNBuilder\PreProcessorConfig.json","preprocessor.py")
    # builder.read()
    # builder.build()
    # builder.save()



