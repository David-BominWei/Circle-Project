import torch.nn as nn
from circle.circle_detection_data import show_circle


class ResNetBlock(nn.Module):
    """ Residual block for the ResNet Architecture
    Reference from: https://arxiv.org/abs/1512.03385
    
    Args:
        in_channels: the number of kernel which the input of model have
        out_channels: the number of kernel which the output of model have
        stride: number of steps for the first conv block
        downsample: the modification for the residual side make the shape of both side fixed to each other
    """
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResNetBlock, self).__init__()
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
        if self.downsample:
            residual = self.downsample(x)
        else:
            residual = x
            
        out = self.conv1(x)
        out = self.conv2(out)
        
        out += residual
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    """ ResNet
    Revised from the basic ResNet architecture
    
    Args:
        layers: a list of numbers indicate the number of blocks in each layer
        num_classes: number of output classes
    """
    def __init__(self, layers, num_classes = 3):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(64, layers[0], stride = 1)
        self.layer1 = self._make_layer(128, layers[1], stride = 1)
        self.layer2 = self._make_layer(256, layers[2], stride = 2)
        self.layer3 = self._make_layer(512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(ResNetBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(self.inplanes, planes))

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
    
if __name__ == "__main__":
    from circleDataset import FixCircleDataset
    from torch.utils.data import DataLoader
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    
    test_loader = DataLoader(FixCircleDataset(data_size=128), batch_size=16, shuffle=True)
    test_data = iter(test_loader)
    model = ResNet([3,4,6,3])
    print(model(next(test_data)[0]))
    print(next(test_data)[1])