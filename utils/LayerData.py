class Layers():
    CONV2D = "CONV2D"
    MAXPOOL2D = "MAXPOOL2D"
    LIN = "LIN"
    RELU = "RELU"

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
        "RandomFlip":"RandomHorziontalFlip",
        "RandomRotate":"RandomRotation",
        "GrayScale":"GrayScale"
    }