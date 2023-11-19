import torch.nn as nn
from circle.circle_detection_data import show_circle

# resnet block
# class ResnetBlock(nn.Module):

#     def __init__(self, in_channels, out_channels, stride=1):
        
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != self.expansion * out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * out_channels)
#             )

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         out += self.shortcut(residual)
#         out = self.relu(out)

#         return out

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
                                nn.Linear(128, 2))
        
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

    
    test_loader = DataLoader(FixCircleDataset(data_size=64), batch_size=64, shuffle=True)
    test_data = iter(test_loader)
    model = Network()
    model(next(test_data)[0])