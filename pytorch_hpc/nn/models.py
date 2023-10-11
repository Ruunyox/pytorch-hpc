import torch
import torch.nn as nn
from typing import Optional, List


class FullyConnectedClassifier(nn.Module):
    """Simple fully connected, feed-forward network for classification"""

    def __init__(
        self,
        tag: str,
        in_dim: int,
        out_dim: int,
        hidden_layers: Optional[List[int]] = None,
        activation: torch.nn.modules.activation = torch.nn.modules.activation.ReLU,
        class_activation: torch.nn.modules.activation = torch.nn.modules.activation.Softmax,
    ):
        super(FullyConnectedClassifier, self).__init__()

        self.tag = tag
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        layers = []
        layers.append(nn.Linear(in_dim, hidden_layers[0]))
        if len(hidden_layers) > 1:
            layers.append(activation())
            for i in range(1, len(hidden_layers)):
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
                layers.append(activation())
            layers.append(nn.Linear(hidden_layers[-1], out_dim))
        layers.append(class_activation())

        self.net = nn.Sequential(*layers)

    def forward(x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network"""

        for layer in self.net:
            x = layer(x)

        return x


class ConvolutionClassifier(nn.Module):
    """Convolutional network classifier"""

    def __init__(
        self,
        tag: str,
        in_channels: int,
        out_channels: int,
        in_dim: int,
        out_dim: int,
        hidden_layers: Optional[List[int]] = None,
        conv_channels: Optional[List[int]] = None,
        conv_kernels: Optional[List[int]] = None,
        activation: torch.nn.modules.activation = torch.nn.modules.activation.ReLU,
        class_activation: torch.nn.modules.activation = torch.nn.modules.activation.Softmax,
    ):
        super(ConvolutionClassifier, self).__init__()

        assert all([opt is None for opt in [conv_channels, conv_kernels]]) or all(
            [opt is not None for opt in [conv_channels, conv_kernels]]
        )

        self.tag = tag
        if conv_channels is None:
            conv_channels = [64, 64, 64]
        if conv_kernels is None:
            conv_kernels = [3, 3, 3]
        if hidden_layers is None:
            hidden_layers = [64, 32]

        assert len(conv_channels) == len(conv_kernels)

        conv_layers = []
        conv_layers.append(
            nn.Conv2d(in_channels, conv_channels[0], kernel_size=conv_kernels[0])
        )
        conv_layers.append(activation())
        if len(conv_channels) > 1:
            for i in range(1, len(conv_layers)):
                conv_layers.append(
                    nn.Conv2d(
                        conv_channels[i - 1], conv_channels[i], kernel_size=conv_kernels
                    )
                )
                conv_layers.append(activation())
            conv_layers.append(
                nn.Conv2d(
                    conv_channels[i - 1], conv_channels[i], kernel_size=conv_kernels
                )
            )
            conv_layers.append(activation())

        self.convolutions = nn.Sequential(*conv_layers)

        layers = []
        layers.append(nn.Linear(in_dim, hidden_layers[0]))
        if len(hidden_layers) > 1:
            layers.append(activation())
            for i in range(1, len(hidden_layers)):
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
                layers.append(activation())
            layers.append(nn.Linear(hidden_layers[-1], out_dim))
        layers.append(class_activation())

        self.net = nn.Sequential(*layers)

    def forward(x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network"""

        for layer in self.convolutions:
            x = layer(x)
        for layer in self.net:
            x = layer(x)

        return x
