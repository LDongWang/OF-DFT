from functools import lru_cache

import numpy as np

import pyscf
import pyscf.df

from ofdft.integrals import int_2c2e, int_3c2e


@lru_cache(16)
def df_coeff(mol, auxmol):
    nao, naux = mol.nao, auxmol.nao
    coeff = np.linalg.solve(int_2c2e(auxmol), int_3c2e(mol, auxmol).reshape(nao*nao, naux).T)
    return coeff


@lru_cache(16)
def df_coeff_jext(mol, auxmol):
    from ofdft.integrals import int1c1e_nuc_analytical
    nao, naux = mol.nao, auxmol.nao
    int_1c1e_nuc = int1c1e_nuc_analytical(auxmol)
    a = np.concatenate([int_2c2e(auxmol), int_1c1e_nuc[None]], axis=0)
    b = int_3c2e(mol, auxmol).reshape(nao*nao, naux).T
    b = np.concatenate([b, mol.intor('int1e_nuc').reshape(1, -1)], axis=0)
    coeff = np.linalg.lstsq(a, b, rcond=None)[0]
    return coeff


def df_coeff_inv(df_coeff):
    return np.linalg.pinv(df_coeff)
