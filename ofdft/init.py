import numpy as np

import pyscf
import pyscf.df

from ofdft.integrals import int1c1e_nuc_analytical, int1c1e_int_analytical, int_3c2e_sym, int_2c2e


INIT_XC = 'LDA,VWN'

def compute_jaux(mol, auxmol, dm):
    ti = np.tril_indices(mol.nao)
    tw = (np.ones(mol.nao) * 2 - np.eye(mol.nao))[ti]

    dml = dm[ti] * tw

    pmol = pyscf.gto.mole.conc_mol(mol, auxmol)
    jaux_pieces = []
    for l in range(auxmol.nbas):
        shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas + l, mol.nbas + l + 1)
        int3c2e = pmol.intor('int3c2e', aosym='s2ij', shls_slice=shls_slice)
        jaux_pieces.append(dml @ int3c2e)
        del int3c2e
    jaux = np.concatenate(jaux_pieces, axis=-1)
    return jaux


def get_rho_coeff(mol, auxmol, dm):
    jaux = compute_jaux(mol, auxmol, dm)
    rho_coeff = np.linalg.solve(int_2c2e(auxmol), jaux)
    return rho_coeff


def get_rho_coeff_jext_fit(mol, auxmol, dm):
    jaux = compute_jaux(mol, auxmol, dm)
    vext = (dm * mol.intor('int1e_nuc')).sum()
    int_1c1e_nuc = int1c1e_nuc_analytical(auxmol)
    a = np.concatenate([int_2c2e(auxmol), int_1c1e_nuc[None]], axis=0)
    b = np.concatenate([jaux, vext[None]])
    rho_coeff = np.linalg.lstsq(a, b, rcond=None)[0]
    return rho_coeff


def get_rho_coeff_jextnelec_fit(mol, auxmol, dm):
    jaux = compute_jaux(mol, auxmol, dm)
    vext = (dm * mol.intor('int1e_nuc')).sum()
    nelec = np.array([mol.nelectron])
    int_1c1e_nuc = int1c1e_nuc_analytical(auxmol)
    int_1c1e_int = int1c1e_int_analytical(auxmol)
    a = np.concatenate([int_2c2e(auxmol), int_1c1e_nuc[None], int_1c1e_int[None]], axis=0)
    b = np.concatenate([jaux, vext[None], nelec])
    rho_coeff = np.linalg.lstsq(a, b, rcond=None)[0]
    return rho_coeff


def get_rho_coeff_density_fit(mol, auxmol, dm):
    grid = pyscf.dft.gen_grid.Grids(mol)
    grid.build()
    ao = pyscf.dft.numint.eval_ao(mol, grid.coords, deriv=0)
    density = (ao[:, None, :] * ao[:, :, None] * dm[None]).sum((1, 2))
    auxao = pyscf.dft.numint.eval_ao(auxmol, grid.coords, deriv=0)
    v = (density[:, None] * auxao * grid.weights[:, None]).sum(0)
    coeff = np.linalg.solve(auxmol.intor('int1e_ovlp'), v)
    return coeff


init_get_rho_coeff = get_rho_coeff


def initguess_minao(mol, auxmol, use_dm=False):
    dm = pyscf.scf.get_init_guess(mol, key='minao')
    return dm if use_dm else init_get_rho_coeff(mol, auxmol, dm)


def initguess_fastminao(mol, auxmol, use_dm=False):
    assert not use_dm, 'fastminao does not support use_dm=True'
    assert mol.basis == '6-31G(2df,p)' # and also etb beta=2.5

    separated_atom_minao_coeff = {
        'H': [0.00941537, 0.00861688, 0.04246949, 0.07798538, 0.09247137, 0.03512736] + [0.] * 14,
        'C': [2.50026903e-02, 8.51211887e-02, 1.82006255e-01, 6.26637654e-01, 1.38423615e+00, 2.36860522e+00, 1.60647577e+00, 7.47287867e-04, -6.03031616e-02, 5.57420045e-01, 1.96031741e-01] + [0.] * 98,
        'N': [0.06474738, 0.04587693, 0.36253395, 0.74939406, 2.0237005, 2.93289245, 1.93891821, -0.28034655, 0.22793199, 0.78507542, 0.2494403] + [0.] * 105,
        'O': [0.10790467, -0.00467717, 0.55329868, 0.84883261, 2.67425785, 3.5159073, 2.37671818, -0.53322403, 0.54521305, 0.94777563, 0.40014992] + [0.] * 105,
        'F': [0.14128195, -0.05177755, 0.69881375, 0.90095582, 3.19922358, 4.1422579, 3.06678716, -0.71806152, 0.82631697, 1.25401064, 0.61586898] + [0.] * 105,
    }
    fastminao = np.concatenate([separated_atom_minao_coeff[mol.elements[ia]] for ia in range(mol.natm)])
    return fastminao


def initguess_huckel(mol, auxmol, use_dm=False):
    dm = pyscf.scf.get_init_guess(mol, key='huckel')
    return dm if use_dm else init_get_rho_coeff(mol, auxmol, dm)


def initguess_atom(mol, auxmol, use_dm=False):
    dm = pyscf.scf.get_init_guess(mol, key='atom')
    return dm if use_dm else init_get_rho_coeff(mol, auxmol, dm)


def initguess_1e(mol, auxmol, use_dm=False):
    dm = pyscf.scf.get_init_guess(mol, key='1e')
    return dm if use_dm else init_get_rho_coeff(mol, auxmol, dm)


def init_random(mol, auxmol, use_dm=False):
    if use_dm:
        return np.random.rand(mol.nao, mol.nao)
    else:
        return np.random.rand(auxmol.nao)


def init_gt(mol, auxmol, use_dm=False):
    mf = mol.RKS(xc=INIT_XC)
    mf.kernel()
    dm = mf.make_rdm1()
    return dm if use_dm else init_get_rho_coeff(mol, auxmol, dm)


def init_halfgt(mol, auxmol, steps=1, guess='minao', use_dm=False):
    mf = mol.RKS(xc=INIT_XC)
    mf.max_cycle = steps
    init_dm = mf.get_init_guess(mol, key=guess)
    mf.kernel(dm0=init_dm)
    dm = mf.make_rdm1()
    return dm if use_dm else init_get_rho_coeff(mol, auxmol, dm)


def get_init_coeff(init_method, mol, auxmol, *args, use_dm=False, **kwargs):
    methods = {
        'initguess_minao': initguess_minao,
        'initguess_fastminao': initguess_fastminao,
        'initguess_atom': initguess_atom,
        'initguess_1e': initguess_1e,
        'initguess_huckel': initguess_huckel,
        'random': init_random,
        'gt': init_gt,
        'halfgt': init_halfgt,
    }
    method = methods[init_method]
    return method(mol, auxmol, *args, use_dm=use_dm, **kwargs)


def ref_etb(mol, beta):
    reference_mol = pyscf.M(atom='H 0 0 0; C 0 -0.5 0.5; N 0 0.5 1; O 0 0.5 -0.5; F 0.5 0 0', basis=mol.basis, spin=1)
    basis = pyscf.df.aug_etb(reference_mol, beta)
    return basis
