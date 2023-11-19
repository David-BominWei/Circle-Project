import torch.nn as nn
from circle.circle_detection_data import show_circle

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 3):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 1)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x.unsqueeze(1))
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class Network(nn.Module):

    def __init__(self, image_size=100):
        super().__init__()
        
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1 = nn.Conv2d(1,16,3,1,1)
        self.conv2 = nn.Conv2d(16,32,3,1,1)
        
        self.fc = nn.Sequential(nn.Linear(32*25*25, 128),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(128, 128),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(128, 3),
                                nn.ReLU())
        
    def forward(self, x):
        
        x = x.unsqueeze(1) # add another dimention to fit the cnn input
        x = self.pooling(self.conv1(x))
        x = self.pooling(self.conv2(x))
        x = x.flatten(1) # flatten the model to fit into the fc layer
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    from circleDataset import FixCircleDataset
    from torch.utils.data import DataLoader
    import numpy as np
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    
    test_loader = DataLoader(FixCircleDataset(data_size=128), batch_size=16, shuffle=True)
    test_data = iter(test_loader)
    model = Network()
    print(model(next(test_data)[0]))
    print(next(test_data)[1])