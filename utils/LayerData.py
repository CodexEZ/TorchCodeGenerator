from FunMapping import *
class Layers():
    CONV2D = "CONV2D"
    MAXPOOL2D = "MAXPOOL2D"
    LIN = "LIN"
    RELU = "RELU"
    FLATTEN = "FLATTEN"
    TRANSFORMER = "TRANSFORMER"
    EMBEDDING = "EMBEDDING"
    RESBLOCK = "RESBLOCK"
    DROPOUT = "DROPOUT"
    BATCHNORM1D = "BATCHNORM1D"
    BATCHNORM2D = "BATCHNORM2D"
    GAPOOL2D = "GAPOOL2D"

class Loss():
    CROSSENTROPY = "CROSSENTROPY"

class Optimizer():
    ADAMW = "ADAMW"

class Transforms:
    pre_processors = {
        "Tensor":"ToTensor",
        "Normalize":"Normalize",
        "Resize":"Resize",
        "Image":"ToPILImage",
        "CenterCrop":"CenterCrop",
        "RandomCrop":"RandomCrop",
        "RandomFlip":"RandomHorizontalFlip",
        "RandomRotate":"RandomRotation",
        "GrayScale":"Grayscale",
        " ":" ",
        "":" ",
        None:""
    }

class LayerMappings:
    layers = {
        Layers.LIN : linear,
        Layers.CONV2D : conv2d,
        Layers.FLATTEN : flatten,
        Layers.MAXPOOL2D : maxPool2D,
        Layers.DROPOUT : dropout,
        Layers.RELU : relu,
        Layers.
    }