import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple
from copy import deepcopy

__all__ = ["FullyConnectedClassifier", "ConvolutionClassifier"]


class FullyConnectedClassifier(nn.Module):
    """Simple fully connected, feed-forward network for classification

    Parameters
    ----------
    tag:
        `str` name for the model
    in_dim:
        `int` specifying the (non-batch) model input dimension
    out_dim:
        `int` specifying the (non-batch) model output dimension
    activation:
        `torch.nn.Module` activation function after each linear transform
    class_activation:
        `torch.nn.Module` activation after the final linear transform for class prediction
    hidden_layers:
        `List[int]` of hidden linear transform widths
    """

    def __init__(
        self,
        tag: str,
        in_dim: int,
        out_dim: int,
        activation: torch.nn.Module,
        class_activation: torch.nn.Module,
        hidden_layers: Optional[List[int]] = None,
    ):
        super(FullyConnectedClassifier, self).__init__()

        self.tag = tag
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        layers = []
        layers.append(nn.Linear(in_dim, hidden_layers[0]))
        layers.append(deepcopy(activation))
        if len(hidden_layers) > 1:
            for i in range(1, len(hidden_layers)):
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
                layers.append(deepcopy(activation))
        layers.append(nn.Linear(hidden_layers[-1], out_dim))
        layers.append(deepcopy(class_activation))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network. Input images
        are flattened along the pixel and channel dimensions to
        produce a batch of 1D tensors.

        Parameters
        ----------
        x:
            input `torch.Tensor` of shape (batch_size, n_channels, x_pixels, y_pixels)

        Returns
        -------
        x:
           `torch.Tensor` of shape (batch_size, n_classes)
        """
        n_channels = x.size()[1]
        pixel_size = x.size()[-1]
        x = x.view(-1, pixel_size * pixel_size * n_channels)
        for layer in self.net:
            x = layer(x)

        return x


class ConvolutionClassifier(nn.Module):
    """Simple Convolutional Neural Network with MaxPooling for classification

    Parameters
    ----------
    tag:
        `str` name for the model
    in_channels:
        `int` specifying the (non-batch) input image channels
    out_channels:
        `int` specifying the (non-batch) output channels after the final convolution
    in_dim:
        `int` specifying the (non-batch) fully-connected network input dimension
    out_dim:
        `int` specifying the (non-batch) fully-connected network output dimension
    activation:
        `torch.nn.Module` activation function after each linear transform
    class_activation:
        `torch.nn.Module` activation after the final linear transform for class prediction
    hidden_layers:
        `List[int]` of hidden linear transform widths in the fully connected network
    conv_channels:
        `List[int]` specifying the series of channels for each convolution layer
    conv_kernels:
        `List[Union[List[int], List[Tuple[int]]]]` specifying the series of kernel sizes for each
        convolutional layer
    pooling_kernels:
        `List[Union[List[int], List[Tuple[int]]]]` specifying the series of kernel sizes for each
        pooling layer
    """

    def __init__(
        self,
        tag: str,
        in_channels: int,
        in_dim: int,
        out_dim: int,
        activation: torch.nn.Module,
        class_activation: torch.nn.Module,
        hidden_layers: Optional[List[int]] = None,
        conv_channels: Optional[List[int]] = None,
        conv_kernels: Optional[Union[List[int], List[Tuple[int]]]] = None,
        pooling_kernels: Optional[Union[List[int], List[Tuple[int]]]] = None,
    ):
        super(ConvolutionClassifier, self).__init__()

        assert all([opt is None for opt in [conv_channels, conv_kernels]]) or all(
            [opt is not None for opt in [conv_channels, conv_kernels]]
        )

        self.tag = tag

        if conv_channels is None:
            conv_channels = [64, 64]
        if conv_kernels is None:
            conv_kernels = [2, 2]
        if pooling_kernels is None:
            pooling_kernels = [2, 2]
        if hidden_layers is None:
            hidden_layers = [64, 32]
        assert len(conv_channels) == len(conv_kernels) == len(pooling_kernels)

        conv_layers = []
        conv_layers.append(
            nn.Conv2d(
                in_channels,
                conv_channels[0],
                kernel_size=conv_kernels[0],
            )
        )
        conv_layers.append(nn.MaxPool2d(pooling_kernels[0]))
        conv_layers.append(deepcopy(activation))

        if len(conv_channels) > 1:
            for i in range(1, len(conv_channels)):
                conv_layers.append(
                    nn.Conv2d(
                        conv_channels[i - 1],
                        conv_channels[i],
                        kernel_size=conv_kernels[i],
                    )
                )
                conv_layers.append(nn.MaxPool2d(pooling_kernels[i]))
                conv_layers.append(deepcopy(activation))

        self.convolutions = nn.Sequential(*conv_layers)

        layers = []
        layers.append(nn.Linear(in_dim, hidden_layers[0]))
        layers.append(deepcopy(activation))
        if len(hidden_layers) > 1:
            for i in range(1, len(hidden_layers)):
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
                layers.append(deepcopy(activation))
        layers.append(nn.Linear(hidden_layers[-1], out_dim))
        layers.append(deepcopy(class_activation))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network.

        Parameters
        ----------
        x:
            input `torch.Tensor` of shape (batch_size, n_channels, x_pixels, y_pixels)

        Returns
        -------
        x:
           `torch.Tensor` of shape (batch_size, n_classes)
        """

        for layer in self.convolutions:
            x = layer(x)

        _, out_channels, pixel_x, pixel_y = x.size()
        x = x.view(-1, pixel_x * pixel_y * out_channel)
        for layer in self.net:
            x = layer(x)

        return x
