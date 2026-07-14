import torch
import torch.nn as nn
import torch.nn.functional as F

class CameraAI(nn.Module):
    def __init__(self):
        super(CameraAI, self).__init__()
        # Input image size: 3 channels (RGB) x 144 height x 256 width
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1) # -> 16 x 72 x 128
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # -> 32 x 36 x 64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # -> 64 x 18 x 32
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # -> 128 x 9 x 16
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # -> 128 x 1 x 1
        
        # Outputting 7 values: Location (X, Y, Z), Rotation/Position (X, Y, Z), and Focal Length
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 7) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
