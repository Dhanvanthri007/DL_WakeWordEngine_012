import torch
import torch.nn as nn
import torch.nn.functional as F

class WakeWordCNN(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        """
        A lightweight Convolutional Neural Network for Wake Word detection.
        Args:
            num_classes (int): Number of output classes (e.g. 2 for binary: Wake Word vs Other)
            in_channels (int): Input channels (1 for mono audio MFCC)
        """
        super(WakeWordCNN, self).__init__()
        
        # Input shape: (Batch, Channels, MFCC_features, Time)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.dropout = nn.Dropout(0.3)
        
        # Adaptive pooling to handle varying time lengths, reducing down to a fixed spatial dimension
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x is (batch, in_channels=1, n_mfcc, time)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(x)
        return x
