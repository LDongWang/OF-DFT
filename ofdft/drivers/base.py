from contextlib import contextmanager

import numpy as np

import torch
import torch.nn as nn

import pyscf
import pyscf.df
import pyscf.dft

from ofdft.modules import OFDFT, BackwardableOFDFT
from ofdft.init import get_init_coeff
from ofdft.integrals import int1c1e_int_analytical

class BaseOFDFTDriver(nn.Module):
    def __init__(self, mol, grid=None, grid_type='basic', grid_slice_size=32768,
                 init_method='initguess_minao',
                 init_normalize=False, normalize_coeff=False,
                 auxbasis='def2-universal-jfit',
                 **kwargs):
        super().__init__()

        self.mol = mol

        self.grid = grid

        self.auxmol = pyscf.df.addons.make_auxmol(mol, auxbasis=auxbasis)

        self.should_normalize = normalize_coeff

        self.ofdft = BackwardableOFDFT(
            self.mol,
            self.auxmol,
            self.preprocess_fn,
            self.tsbase_fn,
            self.xc_fn,
            self.correction_fn,
            grid_coords=self.grid.coords if self.grid is not None else None,
            grid_weights=self.grid.weights if self.grid is not None else None,
            grid_type=grid_type,
            grid_slice_size=grid_slice_size,
        )

        make_param = lambda a: nn.Parameter(torch.tensor(a), requires_grad=False)

        self.norm_vec = make_param(int1c1e_int_analytical(self.auxmol))

        if isinstance(init_method, str):
            var_init = torch.tensor(get_init_coeff(init_method, self.mol, self.auxmol, use_dm=self.use_dm))
            if init_normalize:
                var_init = self.normalize_coeff(var_init)
        else:
            var_init = torch.tensor(init_method(self.mol, self.auxmol))

        self.var = nn.Parameter(var_init, requires_grad=True)

    def build_grid(self, mol=None, level=None):
        grid = pyscf.dft.gen_grid.Grids(mol or self.mol)
        if level is not None:
            grid.level = level
        grid.build()
        return grid

    def normalize_var(self, var):
        int_nelec = (self.norm_vec * var).sum()
        norm_factor = self.mol.nelectron / int_nelec
        var = var * norm_factor
        return var

    @property
    def coeff_var(self):
        return self.var

    @property
    def normalized_var(self):
        return self.normalize_var(self.coeff_var)

    @property
    def normalized_coeff(self):
        return self.normalized_var

    @property
    def coeff_for_input(self):
        return self.normalized_coeff if self.should_normalize else self.coeff_var

    def auxrho(self):
        return self.ofdft.all_auxao_values() @ self.coeff_for_input.detach().cpu()

    def preprocess_fn(self, coeff):
        pass

    def correction_fn(self, data):
        pass

    def forward(self):
        return self.ofdft(self.coeff_for_input)

    @contextmanager
    def context(self):
        with self.ofdft.context(self.coeff_for_input) as ctx:
            yield ctx

    def forward_and_backward(self, forward_parts=None, backward_parts=None):
        return self.ofdft.forward_and_backward(
            lambda: self.coeff_for_input,
            forward_parts=forward_parts,
            backward_parts=backward_parts
        )

    def evaluate_veff(self):
        assert self.var.grad is None or torch.all(self.var.grad.eq(0))
        coeffs = lambda: self.coeff_for_input
        self.ofdft.evaluate_energy(coeffs, self.ofdft.compute_j, backward=True)
        self.ofdft.evaluate_energy_grid(coeffs, self.ofdft.compute_xc, backward=True)
        veff = self.var.grad.detach()
        self.var.grad = None
        return veff

    def forward_and_backward_with_fixed_veff(self, veff):
        backward_parts = ['vext', 'corr']
        rets = self.ofdft.forward_and_backward(lambda: self.coeff_for_input, backward_parts=backward_parts)
        self.var.grad += veff
        return rets
