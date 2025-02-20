
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
#Designed by Aswin


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Automatically create a downsample operation if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x  # Store original input
        
        if self.downsample is not None:
            identity = self.downsample(x)  # Adjust input shape if needed
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # Residual connection (skip connection)
        out = F.relu(out)
        
        return out
        
class Model(nn.Module):
	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		self.layer0 = nn.Conv2d(3,64,7,2)
		self.layer1 = nn.BatchNorm2d(num_features = 64)
		self.layer2 = nn.ReLU()
		self.layer3 = nn.MaxPool2d(3,2)
		self.layer4 = ResidualBlock(in_channels=64,out_channels=64,stride = 1)
		self.layer5 = ResidualBlock(in_channels=64,out_channels=64,stride = 1)
		self.layer6 = ResidualBlock(in_channels=64,out_channels=128,stride = 2)
		self.layer7 = ResidualBlock(in_channels=128,out_channels=128,stride = 1)
		self.layer8 = ResidualBlock(in_channels=128,out_channels=256,stride = 2)
		self.layer9 = ResidualBlock(in_channels=256,out_channels=256,stride = 1)
		self.layer10 = ResidualBlock(in_channels=256,out_channels=512,stride = 2)
		self.layer11 = ResidualBlock(in_channels=512,out_channels=512,stride = 1)
		self.layer12 = nn.AdaptiveAvgPool2d((1,1))
		self.layer13 = nn.Flatten(start_dim = 1, end_dim = -1)
		self.layer14 = nn.Linear(512, 1000)
	def forward(self,x):
		x = self.layer0(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = self.layer6(x)
		x = self.layer7(x)
		x = self.layer8(x)
		x = self.layer9(x)
		x = self.layer10(x)
		x = self.layer11(x)
		x = self.layer12(x)
		x = self.layer13(x)
		x = self.layer14(x)
		return x
