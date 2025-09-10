import torch
from nequip.nn import GraphModuleMixin
from typing import Union, Sequence, Dict
from math import sqrt
from nequip.data import AtomicDataDict


class AvgNumNeighborsNorm(torch.nn.Module):
    def __init__(
        self,
        type_names: Sequence[str],
        avg_num_neighbors: Union[float, Dict[str, float]],
    ) -> None:
        """
        Module to normalize features during training using per type edge sum normalization.

        Args:

        """
        super().__init__()
        self.type_names = type_names
        self.avg_num_neighbors = avg_num_neighbors

        # Put avg_num_neighbors in a list (global or per type)
        if isinstance(avg_num_neighbors, float):
            avg_num_neighbors = [avg_num_neighbors] * len(type_names)
        elif isinstance(avg_num_neighbors, dict):
            assert set(type_names) == set(avg_num_neighbors.keys())
            avg_num_neighbors = [avg_num_neighbors[k] for k in type_names]
        else:
            raise RuntimeError(
                "Unrecognized format for `avg_num_neighbors`, only floats or dicts allowed."
            )
        assert isinstance(avg_num_neighbors, list) and len(avg_num_neighbors) == len(
            type_names
        )

        # If only one type, no need to do embedding lookup in forward
        self.norm_shortcut = len(type_names) == 1

        # Tensorize avg_num_neighbors and register as buffer
        scatter_norm_factor = torch.tensor([(1.0 / sqrt(N)) for N in avg_num_neighbors])
        scatter_norm_factor = scatter_norm_factor.reshape(-1, 1)
        self.register_buffer("scatter_norm_factor", scatter_norm_factor)

    def forward(self, data: AtomicDataDict.Type, num_local_nodes: int) -> torch.Tensor:
        if self.norm_shortcut:
            # No need to do embedding lookup in forward
            scatter_norm = self.scatter_norm_factor
        else:
            # Embed each avg_num_neighbors value per type and reshape to (num_local_nodes, 1)
            scatter_norm = torch.nn.functional.embedding(
                data[AtomicDataDict.ATOM_TYPE_KEY][:num_local_nodes],
                self.scatter_norm_factor,
            )
        return scatter_norm
