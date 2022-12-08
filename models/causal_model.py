import torch.nn as nn
import numpy as np
import torch
from collections import OrderedDict
from models.CITRIS_encoder_decoder import Encoder, SimpleEncoder

class TanhScaled(nn.Module):
    """ Tanh activation function with scaling factor """

    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        assert self.scale > 0, 'Only positive scales allowed.'

    def forward(self, x):
        return torch.tanh(x / self.scale) * self.scale


class CausalNet(nn.Module):
    """ Network trained supervisedly on predicted the ground truth causal factors from input data, e.g. images """
    def __init__(self, causal_var_info, img_width, c_in, c_hid, is_mlp):
        super().__init__()
        self.causal_var_info = causal_var_info
        if not is_mlp:
            self.encoder = Encoder(num_latents=c_hid,
                                   c_in=max(3, c_in),
                                   c_hid=c_hid,
                                   width=img_width,
                                   act_fn=lambda: nn.SiLU(),
                                   variational=False)
        else:
            self.encoder = nn.Sequential(
                        nn.Linear(c_in, c_hid),
                        nn.Tanh(),
                        nn.Linear(c_hid, c_hid),
                        nn.Tanh()
                    )

        # For each causal variable, we create a separate layer as 'head'.
        # Depending on the domain, we use different specifications for the head.
        self.pred_layers = nn.ModuleDict()
        for var_key in causal_var_info:
            var_info = causal_var_info[var_key]
            if var_info.startswith('continuous'):  # Regression
                self.pred_layers[var_key] = nn.Sequential(
                    nn.Linear(c_hid, 1),
                    TanhScaled(scale=float(var_info.split('_')[-1]))
                )
            elif var_info.startswith('angle'):  # Predicting 2D vector for the angle
                self.pred_layers[var_key] = nn.Linear(c_hid, 2)
            elif var_info.startswith('categ'):  # Classification
                self.pred_layers[var_key] = nn.Linear(c_hid, int(var_info.split('_')[-1]))
            else:
                assert False, f'Do not know how to handle key \"{var_key}\" in CausalEncoder initialization.'

    def forward(self, x):
        z = self.encoder(x)
        v = OrderedDict()
        for var_key in self.causal_var_info:
            v[var_key] = self.pred_layers[var_key](z)
        return v