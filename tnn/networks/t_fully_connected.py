import torch.nn as nn
from tnn.layers import tLinearLayer

_UNSET = object()  # sentinel: distinguishes "not provided" from explicitly passing None


class tFullyConnected(nn.Module):

    def __init__(self, layer_widths, dim3, M, activation=None, bias=True,
                 last_activation=_UNSET, device=None, dtype=None):
        """
        last_activation: activation for the final layer only.
        If not provided, all layers use `activation`.
        Pass last_activation=None explicitly to give the last layer no activation
        while hidden layers use `activation` (e.g. Tanh).
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(tFullyConnected, self).__init__()

        self.M = M
        self.depth = len(layer_widths)
        n_layers = self.depth - 1

        layers = []
        for i in range(n_layers):
            is_last = (i == n_layers - 1)
            act = (last_activation if (is_last and last_activation is not _UNSET)
                   else activation)
            layers.append(tLinearLayer(layer_widths[i], layer_widths[i + 1], dim3,
                                       activation=act, bias=bias, **factory_kwargs))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, self.M)
        return x
