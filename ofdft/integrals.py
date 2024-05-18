import warnings
from functools import lru_cache
import numpy as np
import pyscf

# See: CINTcommon_fac_sp in https://github.com/sunqm/libcint/blob/master/src/g1e.c
S_ORBITAL_COMMON_FACTOR = 0.282094791773878143

@lru_cache(16)
def build_1c1e_helper_mol(mol):
    # Build a helper mol with an invalid basis.
    # That is, the helper mol will have only 1 AO for each atom, which is an S orbital with exp=0.
    # (so that the ao will have values of 1 everywhere in space.)
    old_config = pyscf.gto.mole.NORMALIZE_GTO
    pyscf.gto.mole.NORMALIZE_GTO = False
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*divide by zero.*')
        helper_mol = pyscf.M(atom=mol.atom, basis={'default': [[0, (0.0, 1.0)]]})
    pyscf.gto.mole.NORMALIZE_GTO = old_config
    # helper_mol will have a coeff of 0.0, which need the following fix.
    for bas_id in range(helper_mol.nbas):
        nprim = helper_mol.bas_nprim(bas_id)
        nctr = helper_mol.bas_nctr(bas_id)
        ptr = helper_mol._bas[bas_id, pyscf.gto.mole.PTR_COEFF]
        helper_mol._env[ptr:ptr+nprim*nctr] = 1 / S_ORBITAL_COMMON_FACTOR
    return helper_mol


@lru_cache(16)
def int1c1e_nuc_analytical(auxmol: pyscf.gto.Mole):
    helper_mol = build_1c1e_helper_mol(auxmol)
    intor = pyscf.gto.mole.intor_cross('int1e_nuc', helper_mol, auxmol)
    idx = [
        auxmol.bas_atom(ibas)
        for ibas in range(auxmol.nbas) for _ in range(auxmol.bas_angular(ibas) * 2 + 1)
    ]
    intor = intor[idx, range(auxmol.nao)]
    intor = intor / 2  # NOTE: not sure why we need this
    return intor


@lru_cache(16)
def int1c1e_int_analytical(auxmol: pyscf.gto.Mole):
    helper_mol = build_1c1e_helper_mol(auxmol)
    intor = pyscf.gto.mole.intor_cross('int1e_ovlp', helper_mol, auxmol)
    intor = intor[0]
    return intor


@lru_cache(16)
def int2c2e_analytical(auxmol: pyscf.gto.Mole):
    auxao_2c2e = auxmol.intor('int2c2e')
    return auxao_2c2e


@lru_cache(16)
def int1c1e_nuc_numerical(
    auxmol: pyscf.gto.Mole, auxao_values: np.ndarray,
    grid_coords: np.ndarray, grid_weights: np.ndarray
):
    charges = auxmol.atom_charges()
    atom_coords = auxmol.atom_coords()
    dist = ((atom_coords[:, None] - grid_coords[None]) ** 2).sum(-1) ** 0.5
    weights = -charges @ (1. / dist) * grid_weights
    return weights @ auxao_values


def compute_coulumb(coeffs, auxao_2c2e):
    return coeffs @ auxao_2c2e @ coeffs / 2


def compute_vext(coeffs, auxao_1c1e_nuc):
    return coeffs @ auxao_1c1e_nuc


@lru_cache(16)
def int_3c2e(mol, auxmol):
    return pyscf.df.incore.aux_e2(mol, auxmol, intor='int3c2e')


@lru_cache(16)
def int_3c2e_sym(mol, auxmol):
    return pyscf.df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s2ij')


@lru_cache(16)
def int_2c2e(auxmol):
    return auxmol.intor('int2c2e')
