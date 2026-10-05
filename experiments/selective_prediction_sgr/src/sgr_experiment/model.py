"""The VGG-16 variant for CIFAR-10 of Liu and Deng (2015), as used by Geifman and El-Yaniv (2017)."""

from __future__ import annotations

from torch import nn

from probly.transformation import dropout

# 0 stands for max pooling, other entries are the number of output channels of a 3x3 convolution.
CFG = [64, 64, 0, 128, 128, 0, 256, 256, 256, 0, 512, 512, 512, 0, 512, 512, 512, 0]


def build_plain_vgg(num_classes: int = 10) -> nn.Sequential:
    """Build the VGG-16 variant without the dropout layers in front of the two Linear layers.

    Each convolution is followed by ReLU and BatchNorm. Following the reference Keras implementation, a dropout of
    0.3 follows the very first convolution and a dropout of 0.4 follows every other convolution that is not the
    last one of its block. The dropout layers before the two Linear layers (0.5 in the reference) are omitted on
    purpose: they are inserted by :func:`probly.transformation.dropout`.

    Args:
        num_classes: Number of output classes.

    Returns:
        The plain model, which outputs logits.
    """
    layers: list[nn.Module] = []
    in_channels = 3
    for i, entry in enumerate(CFG):
        if entry == 0:
            layers.append(nn.MaxPool2d(2, 2))
            continue
        layers += [nn.Conv2d(in_channels, entry, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(entry)]
        in_channels = entry
        is_last_in_block = CFG[i + 1] == 0
        if not is_last_in_block:
            layers.append(nn.Dropout(0.3 if i == 0 else 0.4))
    layers += [
        nn.Flatten(),
        nn.Linear(512, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Linear(512, num_classes),
    ]
    return nn.Sequential(*layers)


def to_mc_dropout(plain: nn.Module, p: float = 0.5) -> nn.Module:
    """Apply probly's MC dropout transformation, which inserts dropout in front of each Linear layer.

    Args:
        plain: The plain model from :func:`build_plain_vgg`.
        p: Dropout probability of the inserted layers.

    Returns:
        The transformed model.
    """
    return dropout(plain, p=p, predictor_type="logit_classifier")


def build_model(num_classes: int = 10, p: float = 0.5) -> nn.Module:
    """Build the plain VGG and apply the MC dropout transformation (architecture only, untrained)."""
    return to_mc_dropout(build_plain_vgg(num_classes), p=p)
