import torch
from nequip.data import AtomicDataDict
from nequip.nn._graph_mixin import GraphModuleMixin

from e3nn import o3
from e3nn.nn import NormActivation

    
class VectorReadout(GraphModuleMixin, torch.nn.Module):

    def __init__(
        self, 
        irreps_in: None,
        field: str = AtomicDataDict.NODE_FEATURES_KEY,
        out_field: str = AtomicDataDict.FORCE_KEY,
        hidden_dim: int = 16,
        bias: bool = False,
    ):
        super().__init__()
        self.field = field
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[self.field],
        )
        in_irreps = self.irreps_in[self.field]
        assert len(in_irreps) == 1
        assert in_irreps[0].ir == (1, -1), f"expected 1o, got {in_irreps[0].ir}"  # vectors

        hidden_irreps = o3.Irreps(f"{hidden_dim}x1o")  # hidden vector irreps
        out_irreps = o3.Irreps("1x1o")  # single vector

        self.lin1 = o3.Linear(in_irreps, hidden_irreps)

        self.act = NormActivation(
            irreps_in = hidden_irreps,
            scalar_nonlinearity = torch.nn.functional.silu,
            normalize = True,
            bias = bias,
        )
        self.lin2 = o3.Linear(hidden_irreps, out_irreps)


        self.irreps_out.update(self.irreps_in)
        self.irreps_out[self.out_field] = out_irreps

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        x = data[self.field] # (n_atoms, num_feaures*3)
        x = self.lin1(x) # (n_atoms, hidden_dim*3)
        x = self.act(x) # (n_atoms, hidden_dim*3)
        x = self.lin2(x) # (n_atoms, 3)
        data[self.out_field] = x
        return data