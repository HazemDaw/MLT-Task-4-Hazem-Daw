"""Binary classifiers. Every forward pass returns one raw logit per image."""
import torch
from torch import nn
from torchvision import models


class CNNModel(nn.Module):
    """The exact three-convolution, 150 x 150 course baseline."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 18 * 18, 64),
                                  nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


class TransferModel(nn.Module):
    def __init__(self, architecture, pretrained=True):
        super().__init__()
        if architecture == 'vgg16':
            net = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features = nn.Sequential(net.features, nn.AdaptiveAvgPool2d((7, 7)), nn.Flatten())
            self.head = nn.Sequential(nn.Linear(512 * 7 * 7, 256), nn.ReLU(),
                                      nn.Dropout(0.5), nn.Linear(256, 1))
        elif architecture == 'resnet18':
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features = nn.Sequential(*list(net.children())[:-1], nn.Flatten())
            self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))
        else:
            raise ValueError(architecture)
        self.features.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        # Freezing parameters alone does not freeze ResNet's BatchNorm statistics.
        self.features.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            features = self.features(x)
        return self.head(features).squeeze(1)


class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, stride=1):
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False),
                                  nn.BatchNorm2d(outputs), nn.ReLU(),
                                  nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(outputs))
        self.skip = nn.Identity() if inputs == outputs and stride == 1 else nn.Sequential(
            nn.Conv2d(inputs, outputs, 1, stride, bias=False), nn.BatchNorm2d(outputs))

    def forward(self, x):
        return torch.relu(self.main(x) + self.skip(x))


class ScratchResNet(nn.Module):
    """Compact residual network: random initialization only; no external weights."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            ResidualBlock(32, 32), ResidualBlock(32, 64, 2),
            ResidualBlock(64, 128, 2), ResidualBlock(128, 128),
            ResidualBlock(128, 256, 2), ResidualBlock(256, 256),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, 1))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def build_model(name, pretrained=False):
    if name == 'cnn':
        return CNNModel()
    if name == 'scratch':
        return ScratchResNet()
    return TransferModel(name, pretrained=pretrained)
