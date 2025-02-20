# NNBuilder

NNBuilder is a Python utility that automates the generation of PyTorch neural network code based on JSON configuration files. It helps you quickly prototype models and image preprocessing pipelines by converting simple configuration specifications into fully functional Python scripts.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Building a Neural Network Model](#building-a-neural-network-model)
  - [Building an Image Preprocessing Pipeline](#building-an-image-preprocessing-pipeline)
- [Configuration File Details](#configuration-file-details)
  - [Sample Network Configuration (`classfier_config.json`)](#sample-network-configuration)
  - [Sample Preprocessor Configuration (`PreProcessorConfig.json`)](#sample-preprocessor-configuration)
- [Generated Output Examples](#generated-output-examples)
  - [Example: Generated `network.py`](#example-generated-networkpy)
  - [Example: Generated `preprocessor.py`](#example-generated-preprocessorpy)
- [Customization](#customization)
- [License](#license)

---

## Overview

NNBuilder reads a JSON configuration file that specifies the layers and their parameters for your neural network architecture. It then generates a Python file (`network.py`) that defines a PyTorch `Model` class, complete with a customizable forward pass. Optionally, you can also build an image preprocessor by providing another JSON file (`PreProcessorConfig.json`), which will output a `preprocessor.py` file containing a torchvision transforms pipeline.

---

## Features

- **Flexible Layer Definitions:** Supports various layer types including convolutional layers, linear layers, max pooling, residual blocks, and more.
- **Automatic Residual Block Insertion:** If a residual block is specified, its dependency code is automatically included.
- **Image Preprocessing Pipeline:** Easily generate a transformation pipeline using torchvision’s transforms.
- **Easy Customization:** Modify the JSON configuration to rapidly prototype different network architectures and preprocessing steps.

---

## Directory Structure

```
NNBuilder/
├── main.py                     # Main script containing the network and preprocessor builders
├── classfier_config.json       # JSON configuration file for the neural network
├── PreProcessorConfig.json     # JSON configuration file for image preprocessing (optional)
├── network.py                  # Generated neural network model code
├── preprocessor.py             # Generated image preprocessing pipeline code
└── utils/
    ├── LayerData.py            # Definitions for Layers, Loss, Optimizer, and Transforms
    └── Templates.py            # Code templates (e.g., ResidualBlock definition)
```

---

## Requirements

- Python 3.x
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/)

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository_url>
   cd NNBuilder
   ```

2. **Install Required Packages:**

   ```bash
   pip install torch torchvision
   ```

---

## Usage

### Building a Neural Network Model

1. **Configure Your Network:**

   Update the `classfier_config.json` file with your desired architecture. See the [Sample Network Configuration](#sample-network-configuration) section for an example.

2. **Run the Builder:**

   Execute the main script to generate `network.py`:

   ```bash
   python main.py
   ```

   This script reads the JSON configuration, constructs the model layers and forward pass, and writes the output to `network.py`.

### Building an Image Preprocessing Pipeline

1. **Configure Your Preprocessor:**

   Update the `PreProcessorConfig.json` file with your desired transformations. See the [Sample Preprocessor Configuration](#sample-preprocessor-configuration) section for an example.

2. **Enable Preprocessor Building:**

   In `main.py`, uncomment the code block that instantiates and runs the `ImagePreProcessorBuilder`.

3. **Run the Builder:**

   Execute the script to generate `preprocessor.py`:

   ```bash
   python main.py
   ```

   This creates a file containing a torchvision transforms pipeline based on your configuration.

---

## Configuration File Details

### Sample Network Configuration

**File:** `classfier_config.json`

```json
{
    "user": "John Doe",
    "layers": [
        {"type": "CONV2D", "in": 3, "out": 16, "kernel": 3, "stride": 1},
        {"type": "BATCHNORM2D", "num_features": 16},
        {"type": "RELU"},
        {"type": "MAXPOOL2D", "kernel": 2, "stride": 2},
        {"type": "RESBLOCK", "in": 16, "out": 16, "stride": 1},
        {"type": "FLATTEN", "start": 1, "end": -1},
        {"type": "LIN", "in": 256, "out": 10}
    ]
}
```

This configuration specifies:
- A convolutional layer converting 3 input channels to 16 output channels.
- A batch normalization layer for 16 features.
- A ReLU activation.
- A max pooling layer with kernel size 2 and stride 2.
- A residual block (which automatically includes the required dependency code).
- A flattening layer.
- A linear layer mapping 256 features to 10 output classes.

### Sample Preprocessor Configuration

**File:** `PreProcessorConfig.json`

```json
{
    "data": [
        {"type": "Resize", "param1": 224, "param2": 224},
        {"type": "ToTensor", "param1": null, "param2": null},
        {"type": "Normalize", "param1": [0.485, 0.456, 0.406], "param2": [0.229, 0.224, 0.225]}
    ]
}
```

This configuration defines a preprocessing pipeline that:
- Resizes the image to 224x224 pixels.
- Converts the image to a PyTorch tensor.
- Normalizes the image using specified mean and standard deviation values.

---

## Generated Output Examples

### Example: Generated `network.py`

```python
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
#Designed by John Doe
# Residual Block dependency included below since a RESBLOCK is specified:

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.relu(out)
        
        return out

class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer0 = nn.Conv2d(3,16,3,1)
        self.layer1 = nn.BatchNorm2d(num_features = 16)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.MaxPool2d(2,2)
        self.layer4 = ResidualBlock(in_channels=16,out_channels=16,stride = 1)
        self.layer5 = nn.Flatten(start_dim = 1, end_dim = -1)
        self.layer6 = nn.Linear(256, 10)
    def forward(self,x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
```

### Example: Generated `preprocessor.py`

```python
from torchvision import transforms
pre_processor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

---

## Customization

- **Adding New Layer Types:** Extend the `build` method in the `NetworkBuilder` class to support additional PyTorch layers.
- **Modifying Forward Pass:** Customize the sequence of operations in the forward pass by adjusting the generation logic in the builder.
- **Extending Preprocessing:** Add more transformations in the `Transforms.pre_processors` mapping or update the JSON preprocessor configuration to suit your needs.

---

## License

This project is licensed under the MIT License.

---

With NNBuilder, generating and experimenting with different neural network architectures and preprocessing pipelines becomes a streamlined process. Simply adjust your JSON configuration files, run the builder, and start experimenting with the generated PyTorch code!