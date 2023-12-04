import torch
import torch_geometric
from typing import Tuple, Dict


class StandardGraphExpander(torch.nn.Module):
    """Wrapper for standard graph input expansion

    Parameters
    ----------
    scalar:
        If `True`, single channel `Data.x` inputs will be reshaped as
        `(num_nodes, -1)`
    """

    def __init__(self, scalar: bool = False):
        super(StandardGraphExpander, self).__init__()
        self.scalar = scalar

    def forward(self, data: torch_geometric.data.Data) -> Dict[str, torch.Tensor]:
        """Returns subset of `torch_geometric.data.DataBatch/Data` features for
        flexible model input using a LightningModule

        Parameters
        ----------
        data:
            Full input `torch_geometric.data.DataBatch/Data` instance

        Returns
        -------
            `Dict[str, torch.Tensor]` of necessary model inputs
        """

        return {
            "x": data.x.view(data.x.shape[0], -1) if self.scalar else data.x,
            "edge_index": data.edge_index,
            "edge_weight": data.edge_weight,
            "edge_attr": data.edge_attr,
            "batch": data.batch,
        }
