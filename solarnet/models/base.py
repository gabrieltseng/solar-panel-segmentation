from torch import nn
from torchvision.models import resnet34


class ResnetBase(nn.Module):
    """ResNet pretrained on Imagenet. This serves as the
    base for the classifier, and subsequently the segmentation model
    """
    def __init__(self, imagenet_base=True):
        super().__init__()

        resnet = resnet34(pretrained=imagenet_base).float()
        self.pretrained = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        # Since this is just a base, forward() shouldn't directly
        # be called on it.
        raise NotImplementedError
