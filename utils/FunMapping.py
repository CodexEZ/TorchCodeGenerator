from LayerData import Layers
def conv2d(idx, in_channels,out_channels,kernel,stride):
    return f"\t\tself.layer{idx} = nn.Conv2d({in_channels},{out_channels},{kernel},{stride})"

def linear(idx, inpt, out):
    return f"\t\tself.layer{idx} = nn.Linear({inpt}, {out})"

def maxPool2D(idx, kernel, stride):
    return f"\t\tself.layer{idx} = nn.MaxPool2d({kernel},{stride})"

def relu(idx):
    return f"\t\tself.layer{idx} = nn.ReLU()"

def flatten(idx, start_dim, end_dim):
    return f"\t\tself.layer{idx} = nn.Flatten(start_dim = {start_dim}, end_dim = {end_dim})"
def transformer(idx,
                d_model:int,
                nhead:int,
                num_encoder_layers:int, 
                num_decoder_layers:int,
                dim_feedforward:int,
                dropout:float):
    return f"""\t\tself.layer{idx} = nn.Transformer(d_model = {d_model},
                nhead = {nhead},
                num_encoder_layers = {num_encoder_layers},
                num_decoder_layers = {num_decoder_layers},
                dim_feedforward = {dim_feedforward},
                dropout = {dropout},
                batch_first = {True})"""

def embedding(idx,num_embeddings,embedding_dim):
    return f"\t\tself.layer{idx} = nn.Embedding(num_embeddings = {num_embeddings}, embedding_dim = {embedding_dim})"

def resBlock(idx,inpt, out, stride):
    return f"\t\tself.layer{idx} = ResidualBlock(in_channels={inpt},out_channels={out},stride = {stride})"

def dropout(idx, prob):
    return f"\t\tself.layer{idx} = nn.Dropout(p = {prob})"

def batchNorm1D(idx, num_features):
    return f"\t\tself.layer{idx} = nn.BatchNorm1d(num_features = {num_features})"

def batchNorm2D(idx, num_features):
    return f"\t\tself.layer{idx} = nn.BatchNorm2d(num_features = {num_features})"

def gaPool2D(idx, out_shape):
    return  f"\t\tself.layer{idx} = nn.AdaptiveAvgPool2d({out_shape})"