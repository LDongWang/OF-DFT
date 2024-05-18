# For unpolarized density only

import math
from types import SimpleNamespace
from typing import Callable, Dict, Tuple, Union, Optional
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyscf
import pyscf.df

from ofdft.functionals import *
from ofdft.integrals import *


class BasicGridValueProvider(nn.Module):
    def __init__(self, auxmol, grid_coords, grid_weights, slice_size=None):
        super().__init__()
        self.auxmol = auxmol
        self.sliced = slice_size is not None
        auxao_values = pyscf.dft.numint.eval_ao(self.auxmol, grid_coords, deriv=1)
        auxao_values = torch.tensor(auxao_values)
        grid_weights = torch.tensor(grid_weights)
        if self.sliced:
            self.nslice = math.ceil(grid_weights.shape[0] / slice_size)
            self.auxao_values = nn.ParameterList(
                nn.Parameter(t, requires_grad=False)
                for t in auxao_values.split(slice_size, dim=1)
            )
            self.grid_weights = nn.ParameterList(
                nn.Parameter(t, requires_grad=False)
                for t in grid_weights.split(slice_size, dim=0)
            )
            assert len(self.auxao_values) == len(self.grid_weights)
            assert self.nslice == len(self.auxao_values)
        else:
            self.auxao_values = nn.Parameter(auxao_values, requires_grad=False)
            self.grid_weights = nn.Parameter(grid_weights, requires_grad=False)

    def auxao(self, slice_idx=None):
        if self.sliced:
            return self.auxao_values[slice_idx]
        else:
            return self.auxao_values

    def weights(self, slice_idx=None):
        if self.sliced:
            return self.grid_weights[slice_idx]
        else:
            return self.grid_weights

    def all_auxao_values(self):
        if self.sliced:
            return torch.cat([t[0].detach().cpu() for t in self.auxao_values], dim=0)
        else:
            return self.auxao_values[0].detach().cpu()


class LazyGridValueProvider(nn.Module):
    def __init__(self, auxmol, grid_coords, grid_weights, slice_size):
        super().__init__()
        self.auxmol = auxmol
        self.grid_coords = grid_coords
        self.grid_weights = grid_weights
        self.slice_size = slice_size
        self.nslice = math.ceil(grid_weights.shape[0] / slice_size)

    def get_slice(self, slice_idx):
        start = slice_idx * self.slice_size
        end = start + self.slice_size
        return slice(start, end)

    def auxao(self, slice_idx, deriv=1):
        grid_coords = self.grid_coords[self.get_slice(slice_idx)]
        auxao_values = pyscf.dft.numint.eval_ao(self.auxmol, grid_coords, deriv=deriv)
        return torch.tensor(auxao_values)

    def weights(self, slice_idx):
        grid_weights = self.grid_weights[self.get_slice(slice_idx)]
        return torch.tensor(grid_weights)

    def all_auxao_values(self):
        return torch.cat([
            self.get_auxao_value(i, 0) for i in range(self.nslice)
        ], dim=0)


class BaseOFDFT(nn.Module):
    def __init__(
        self,
        mol: pyscf.gto.Mole,
        auxmol: pyscf.gto.Mole,
        preprocess_fn: Callable[[torch.Tensor], Tuple[Dict, torch.Tensor]],
        tsbase_fn: Callable[[DensityVars, torch.Tensor], torch.Tensor],
        xc_fn: Callable[[DensityVars, torch.Tensor], torch.Tensor],
        correction_fn: Callable[[Dict], torch.Tensor],
    ):
        super().__init__()
        self.correction_fn = correction_fn
        self.tsbase_fn = tsbase_fn
        self.xc_fn = xc_fn

        self.mol = mol
        self.auxmol = auxmol

        self.auxao_2c2e = nn.Parameter(torch.tensor(int2c2e_analytical(auxmol)), requires_grad=False)
        self.auxao_1c1e_nuc = nn.Parameter(torch.tensor(int1c1e_nuc_analytical(auxmol)), requires_grad=False)

        self.preprocess_fn = preprocess_fn


class OFDFT(BaseOFDFT):
    def __init__(self, *args, grid_coords, grid_weights, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.tsbase_fn.is_empty or not self.xc_fn.is_empty:
            self.grid = BasicGridValueProvider(self.auxmol, grid_coords, grid_weights)

    def forward(self, coeffs: torch.Tensor):
        with self.context(coeffs) as ctx:
            return ctx.loss, ctx.terms, ctx.tsxc_terms

    @contextmanager
    def context(self, coeffs: torch.Tensor):
        if not self.tsbase_fn.is_empty or not self.xc_fn.is_empty:
            auxao_value = self.grid.auxao().to(self.auxao_2c2e.device)
            grid_weights = self.grid.weights().to(self.auxao_2c2e.device)
            rho = auxao_value @ coeffs
            d = DensityVars(rho, coeffs)

            tsbase, _ = self.tsbase_fn(d, grid_weights)
            xc, xc_terms = self.xc_fn(d, grid_weights)
        else:
            tsbase = 0.0
            xc = 0.0
            xc_terms = {}

        j = compute_coulumb(coeffs, self.auxao_2c2e)
        vext = compute_vext(coeffs, self.auxao_1c1e_nuc)

        data = self.preprocess_fn(coeffs)
        correction = self.correction_fn(data)

        terms = {'vext': vext, 'j': j, 'tsbase': tsbase, 'xc': xc, 'corr': correction}
        e_tot = vext + j + tsbase + xc + correction

        loss = e_tot

        ctx = SimpleNamespace(**locals())
        try:
            yield ctx
        finally:
            pass

    def all_auxao_values(self):
        return self.grid.all_auxao_values()


class BackwardableOFDFT(BaseOFDFT):
    def __init__(
        self,
        *args,
        grid_coords, grid_weights,
        grid_slice_size=32768, grid_type='basic',
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # only save grid-related params when they are needed
        if not self.tsbase_fn.is_empty or not self.xc_fn.is_empty:
            assert grid_type in ['basic', 'lazy']
            if grid_type == 'basic':
                self.grid = BasicGridValueProvider(self.auxmol, grid_coords, grid_weights, grid_slice_size)
            elif grid_type == 'lazy':
                self.grid = LazyGridValueProvider(self.auxmol, grid_coords, grid_weights, grid_slice_size)

    def compute_j(self, coeffs: torch.Tensor):
        return compute_coulumb(coeffs, self.auxao_2c2e)

    def compute_vext(self, coeffs: torch.Tensor):
        return compute_vext(coeffs, self.auxao_1c1e_nuc)

    def compute_tsbase(self, auxao_value: torch.Tensor, grid_weights: torch.Tensor, coeffs: torch.Tensor):
        rho = auxao_value @ coeffs
        d = DensityVars(rho, coeffs)
        tsxc, _ = self.tsbase_fn(d, grid_weights)
        return tsxc

    def compute_xc(self, auxao_value: torch.Tensor, grid_weights: torch.Tensor, coeffs: torch.Tensor):
        rho = auxao_value @ coeffs
        d = DensityVars(rho, coeffs)
        tsxc, _ = self.xc_fn(d, grid_weights)
        return tsxc

    def compute_corr(self, coeffs: torch.Tensor):
        data = self.preprocess_fn(coeffs)
        correction = self.correction_fn(data)
        return correction

    def evaluate_energy(self, coeffs, energy_func, backward=False):
        if isinstance(coeffs, torch.Tensor):
            # assert not backward
            pass
        else:
            coeffs = coeffs()
        e = energy_func(coeffs)
        if backward:
            e.backward()
            e = e.detach()
        return e

    def evaluate_energy_grid(self, coeffs, energy_func, backward=False):
        es = []
        for islice in range(self.grid.nslice):
            auxao = self.grid.auxao(islice).to(self.auxao_2c2e.device)
            gw = self.grid.weights(islice).to(self.auxao_2c2e.device)
            e = self.evaluate_energy(coeffs, lambda c: energy_func(auxao, gw, c), backward)
            es.append(e)
            del auxao
            del gw
        total_e = sum(es)
        return total_e

    def evaluate(
        self,
        coeffs: Union[torch.Tensor, Callable[[], torch.Tensor]],
        backward=False,
        forward_parts=None,
        backward_parts=None
    ):
        should_forward = lambda n: forward_parts is None or n in forward_parts
        should_backward = lambda n: backward and (backward_parts is None or n in backward_parts)

        if should_forward('j'):
            j = self.evaluate_energy(coeffs, self.compute_j, backward=should_backward('j'))
        else:
            j = torch.tensor(0.0).to(self.auxao_2c2e)

        if should_forward('xc') and not self.xc_fn.is_empty:
            xc = self.evaluate_energy_grid(coeffs, self.compute_xc, backward=should_backward('xc'))
        else:
            xc = torch.tensor(0.0).to(j)

        if should_forward('tsbase') and not self.tsbase_fn.is_empty:
            tsbase = self.evaluate_energy_grid(coeffs, self.compute_tsbase, backward=should_backward('tsbase'))
        else:
            tsbase = torch.tensor(0.0).to(j)

        if should_forward('vext'):
            vext = self.evaluate_energy(coeffs, self.compute_vext, backward=should_backward('vext'))
        else:
            vext = torch.tensor(0.0).to(j)

        if should_forward('corr'):
            correction = self.evaluate_energy(coeffs, self.compute_corr, backward=should_backward('corr'))
        else:
            correction = torch.tensor(0.0).to(j)

        terms = {'vext': vext, 'j': j, 'tsbase': tsbase, 'xc': xc, 'corr': correction}
        e_tot = vext + j + tsbase + xc + correction
        loss = e_tot
        return loss, terms, None # does not support returning tsxc_terms

    def forward_and_backward(self, coeffs: Callable[[], torch.Tensor],
                             forward_parts=None, backward_parts=None):
        return self.evaluate(coeffs, backward=True, forward_parts=forward_parts, backward_parts=backward_parts)

    def forward(self, coeffs: torch.Tensor):
        return self.evaluate(coeffs, backward=False)

    def all_auxao_values(self):
        return self.grid.all_auxao_values()
