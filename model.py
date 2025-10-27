import torch.nn as nn

class VegetableNet(nn.Module):
  def __init__(self, num_classes=15, dropout_rate=0.4):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(256, 512, 3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    self.dropout = nn.Dropout(dropout_rate)
    self.fc = nn.Linear(512, num_classes)

  def forward(self, x):
    x = self.conv(x)
    x = self.dropout(x)
    return self.fc(x)
