import torch.nn as nn
import torchvision.models as models

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


class VegetableResNet(nn.Module):
  def __init__(self, num_classes=15, dropout_rate=0.4, pretrained=True):
    super().__init__()
    self.resnet = models.resnet50(pretrained=pretrained)

    self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(dropout_rate),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(512, num_classes)
    )

  def forward(self, x):
    x = self.resnet(x)
    x = self.classifier(x)
    return x
