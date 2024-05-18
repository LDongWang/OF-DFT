"""
Functions for computing Hellmann-Feynman forces for density represented by coeffs on auxbasis.
"""

import functools

import numpy as np

import pyscf
import pyscf.grad

from ofdft.integrals import build_1c1e_helper_mol


def hellmann_feynman_force(auxmol, coeff):
    de = np.zeros((auxmol.natm, 3))
    ext_deriv = extgrad_generator(auxmol)
    for ia in range(auxmol.natm):
        de[ia] += ext_deriv[ia] @ coeff
    de += grad_nuc(auxmol)
    return de


def hellmann_feynman_force_dm(mol, dm):
    de = np.zeros((mol.natm, 3))
    ext_deriv = extgrad_generator_dm(mol)
    for ia in range(mol.natm):
        de[ia] += np.einsum('xij,ij->x', ext_deriv[ia], dm)
    de += grad_nuc(mol)
    return de


def grad_nuc(mol):
    return pyscf.grad.rhf.grad_nuc(mol)


@functools.lru_cache(1)
def extgrad_generator(auxmol):
    aoslices = auxmol.aoslice_by_atom()
    helper_mol = build_1c1e_helper_mol(auxmol)
    ext_derivs = []
    for atm_id in range(auxmol.natm):
        shl0, shl1, p0, p1 = aoslices[atm_id]
        vrinv = np.zeros((3, auxmol.nao))
        with auxmol.with_rinv_at_nucleus(atm_id):
            with helper_mol.with_rinv_at_nucleus(atm_id):
                vrinv = pyscf.gto.mole.intor_cross(
                    'int1e_iprinv',
                    auxmol,
                    helper_mol,
                    comp=3
                ) # <\nabla|1/r|>
                vrinv = vrinv[:, :, 0]
                vrinv = vrinv / 2
                vrinv *= -auxmol.atom_charge(atm_id)
        ext_derivs.append(vrinv * 2)
    return ext_derivs


def int1c1e_ipnuc_analytical(auxmol: pyscf.gto.Mole):
    helper_mol = build_1c1e_helper_mol(auxmol)
    intor = pyscf.gto.mole.intor_cross('int1e_ipnuc', auxmol, helper_mol, comp=3)
    intor = intor[:, :, 0]
    intor = intor / 2
    return intor


@functools.lru_cache(1)
def extgrad_generator_dm(mol):
    aoslices = mol.aoslice_by_atom()
    ext_derivs = []
    for atm_id in range(mol.natm):
        shl0, shl1, p0, p1 = aoslices[atm_id]
        vrinv = np.zeros((3, mol.nao, mol.nao))
        with mol.with_rinv_at_nucleus(atm_id):
            vrinv = mol.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
            vrinv *= -mol.atom_charge(atm_id)
        ext_derivs.append(vrinv + vrinv.transpose(0,2,1))
    return ext_derivs
