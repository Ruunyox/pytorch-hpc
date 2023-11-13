import torch
import torch_geometric
from typing import Optional, List, Union, Tuple, Callable
from copy import deepcopy


class GlobalAddPool(torch.nn.Module):
    """Class wrapper for `torch_geometric.nn.global_add_pool`.
    See `help(torch_geometric.nn.global_add_pool)` for more
    information.
    """

    def __init__(self):
        super(GlobalAddPool, self).__init__()
        self.readout = torch_geometric.nn.global_add_pool

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        size: Optional[int] = None,
    ) -> torch.Tensor:
        return self.readout(x, batch, size)


class GlobalMeanPool(torch.nn.Module):
    """Class wrapper for `torch_geometric.nn.global_mean_pool`.
    See `help(torch_geometric.nn.global_add_pool)` for more
    information.
    """

    def __init__(self):
        super(GlobalMeanPool, self).__init__()
        self.readout = torch_geometric.nn.global_mean_pool

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        size: Optional[int] = None,
    ) -> torch.Tensor:
        return self.readout(x, batch, size)


class FullyConnectedClassifier(torch.nn.Module):
    """Simple fully connected, feed-forward network for classification

    Parameters
    ----------
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
        in_dim: int,
        out_dim: int,
        activation: torch.nn.Module,
        class_activation: torch.nn.Module,
        hidden_layers: Optional[List[int]] = None,
    ):
        super(FullyConnectedClassifier, self).__init__()

        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        layers = []
        layers.append(torch.nn.Linear(in_dim, hidden_layers[0]))
        layers.append(deepcopy(activation))
        if len(hidden_layers) > 1:
            for i in range(1, len(hidden_layers)):
                layers.append(torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
                layers.append(deepcopy(activation))
        layers.append(torch.nn.Linear(hidden_layers[-1], out_dim))
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


class ConvolutionClassifier(torch.nn.Module):
    """Simple Convolutional Neural Network with MaxPooling for classification

    Parameters
    ----------
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
            torch.nn.Conv2d(
                in_channels,
                conv_channels[0],
                kernel_size=conv_kernels[0],
            )
        )
        conv_layers.append(torch.nn.MaxPool2d(pooling_kernels[0]))
        conv_layers.append(deepcopy(activation))

        if len(conv_channels) > 1:
            for i in range(1, len(conv_channels)):
                conv_layers.append(
                    torch.nn.Conv2d(
                        conv_channels[i - 1],
                        conv_channels[i],
                        kernel_size=conv_kernels[i],
                    )
                )
                conv_layers.append(torch.nn.MaxPool2d(pooling_kernels[i]))
                conv_layers.append(deepcopy(activation))

        self.convolutions = torch.nn.Sequential(*conv_layers)

        layers = []
        layers.append(torch.nn.Linear(in_dim, hidden_layers[0]))
        layers.append(deepcopy(activation))
        if len(hidden_layers) > 1:
            for i in range(1, len(hidden_layers)):
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
                layers.append(deepcopy(activation))
        layers.append(torch.nn.Linear(hidden_layers[-1], out_dim))
        layers.append(deepcopy(class_activation))

        self.net = torch.nn.Sequential(*layers)

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
        x = x.view(-1, pixel_x * pixel_y * out_channels)
        for layer in self.net:
            x = layer(x)

        return x


class GraphRegressor(torch.nn.Module):
    """General graph Regressor. Meant to be used readily with torch_geometric.nn.models

    Parameters
    ----------
    graph_model:
        `torch.nn.Module` model performing general node feature updates. E.g.,
        `torch_geometric.nn.models.GCN`
    out_module:
        `torch.nn.Module` for transforming output node features from the `graph_model`. E.g.,
        `torch_geometric.nn.models.MLP`
    readout:
        `torch.nn.Module` or `Callable` that aggregates per-graph node/edge output features
        after the `out_module`. E.g., `torch_geometric.nn.global_add_pool()`
    """

    def __init__(
        self,
        graph_model: torch.nn.Module,
        out_module: torch.nn.Module,
        readout: torch.nn.Module = GlobalMeanPool(),
    ):
        super(GraphRegressor, self).__init__()
        self.graph_model = graph_model
        self.out_module = out_module
        self.readout = readout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward network predition on expanded input graph data

        Parameters
        ----------
        x:
            input `torch.Tensor` of shape (num_batch_nodes, num_node_features),
            determining the features for each node in the batch
        edge_index:
            input `torch.Tensor` of shape (2, num_batch_edges),
            determining which nodes are connected pairwise
        edge_index:
            input `torch.Tensor` of shape (2, num_batch_edges)
        edge_weight:
            input `torch.Tensor` of shape (num_batch_edges,),
            determining the weight of each edge in the batch
        edge_attr:
            input `torch.Tensor` of shape (num_batch_edges, num_edge_features),
            determining the features for each edge in the batch
        batch:
            input `torch.tensor` of shape (num_batch_nodes),
            determining the graph index of each node in the batch

        Returns
        -------
        x:
           `torch.Tensor` of shape (batch_size, n_classes) of
           updated node features
        """

        out = self.graph_model(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            batch=batch,
        )
        out = self.out_module(out)
        out = self.readout(out, batch)
        return out
