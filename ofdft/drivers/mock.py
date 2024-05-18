import torch

from ofdft.functionals import build_tsxc

from ofdft.drivers.base import BaseOFDFTDriver

class MockOFDFTDriver(BaseOFDFTDriver):
    def __init__(self, mol, *args, tsbase, xc, grid=None, **kwargs):
        self.xc_fn = xc
        self.tsbase_fn = tsbase
        if grid is None and not self.xc_fn.is_empty:
            grid = self.build_grid(mol)

        super().__init__(*args, mol=mol, grid=grid, **kwargs)

    def preprocess_fn(self, coeff):
        return coeff

    def correction_fn(self, data):
        coeff = data
        return (coeff - coeff).sum() # implements grad_fn
