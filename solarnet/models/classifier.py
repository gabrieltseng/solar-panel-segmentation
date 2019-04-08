from torch import nn

from .base import ResnetBase


class Classifier(ResnetBase):
    """A ResNet34 Model
    """

    def __init__(self, imagenet_base=True):
        super().__init__(imagenet_base=imagenet_base)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.pretrained(x)
        x = self.avgpool(x)
        return self.classifier(x.view(x.size(0), -1))
