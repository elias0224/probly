"""The VGG-16 variant for CIFAR-10 of Liu and Deng (2015), as used by Geifman and El-Yaniv (2017)."""

from __future__ import annotations

from torch import nn

from probly.method.ddu import ddu
from probly.method.swag import swag
from probly.method.vbll import vbll
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


def to_swag(plain: nn.Module, max_rank: int = 20, scale: float = 0.5) -> nn.Module:
    """Wrap the plain model in probly's SWAG predictor (a copy of the model plus the posterior statistics)."""
    return swag(plain, max_rank=max_rank, scale=scale, predictor_type="logit_classifier")


def to_ddu(plain: nn.Module, sn_coeff: float = 3.0) -> nn.Module:
    """Apply probly's DDU transformation (spectral normalization, encoder, classification and density head)."""
    return ddu(plain, sn_coeff=sn_coeff, predictor_type="logit_classifier")


def to_vbll(plain: nn.Module, parameterization: str = "dense") -> nn.Module:
    """Replace the last Linear layer by a variational Bayesian last layer (a fresh, untrained one)."""
    return vbll(plain, parameterization=parameterization)


def disable_dropout(model: nn.Module) -> nn.Module:
    """Set ``p = 0`` on every ``nn.Dropout``, in place.

    probly's samplers force all ``nn.Dropout`` layers into train mode, which would add the dropout noise of the
    conv blocks to the samples of SWAG; with ``p = 0`` only the method's own randomness remains.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    return model


def build_model(num_classes: int = 10, p: float = 0.5) -> nn.Module:
    """Build the plain VGG and apply the MC dropout transformation (architecture only, untrained)."""
    return to_mc_dropout(build_plain_vgg(num_classes), p=p)
