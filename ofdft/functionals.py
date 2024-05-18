# For unpolarized density only

import math
from typing import Dict, Optional, Callable, List

import numpy as np
import torch

import pyscf

class DensityVars:
    def __init__(self, rho: torch.Tensor, coeff: Optional[torch.Tensor] = None):
        MIN_DENSITY = 1e-14
        MIN_DENSITY_DERIV = 1e-14

        self.coeff = coeff
        self.rho = rho

        # NOTE: this mask as well as the double torch.where is very important to solving
        #   the NaN issue! See https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
        self.mask = (rho[0] < MIN_DENSITY) | ((rho[1:4] ** 2).sum(0) < MIN_DENSITY_DERIV)

        placeholder = torch.tensor(1.0).to(rho)
        self.n = torch.where(self.mask, placeholder, rho[0])
        self.deriv = torch.where(self.mask, placeholder, rho[1:4])

        self.a = self.b = self.n * 0.5
        self.gnn = (self.deriv ** 2).sum(0)
        self.gaa = self.gbb = ((self.deriv * 0.5) ** 2).sum(0)
        self.rs = (3.0 / (self.n * 4.0 * np.pi)) ** (1.0 / 3.0)
        self.xs = ((self.gaa / 4) ** 0.5) / (self.a ** (4 / 3)) * 2


@torch.jit.script
def lda_k_tf_custom(d: DensityVars):
    c_tf = 0.3 * ((3 * np.pi * np.pi) ** (2.0 / 3.0))
    return d.n ** (5 / 3) * c_tf# / d.n


@torch.jit.script
def gga_k_vw_custom(d: DensityVars):
    c_vw = 1.0 / 8.0
    return (c_vw * d.gaa / d.a + c_vw * d.gbb / d.b)


@torch.jit.script
def gga_k_apbe_custom(d: DensityVars):
    kappa = 0.8040
    mu = 0.23889
    x2s = 1/(2*(6*np.pi**2)**(1/3))
    s = x2s * d.xs
    pbe = 1 + kappa*(1 - kappa/(kappa + mu*s**2))

    k_factor_c = 3/10*(6*np.pi**2)**(2/3)
    rs_factor = (3/(4*np.pi))**(1/3)
    lda = k_factor_c*(2**(-5/3))*((rs_factor/d.rs)**2)

    xc = lda * pbe * 2
    return xc * d.n


@torch.jit.script
def lda_x_lda_custom(d: DensityVars):
    c_slater = (81 / (32 * np.pi)) ** (1.0 / 3.0)
    return -c_slater * ((d.n * 0.5) ** (4.0 / 3.0) * 2)


@torch.jit.script
def gga_x_b88_custom(d: DensityVars):
    beta = 0.0042
    gamma = 6.0
    x_factor_c = 3/8*(3/np.pi)**(1/3)*4**(2/3)
    lda_x_factor = -x_factor_c
    rs_factor = (3/(4*np.pi))**(1/3)
    lda = lda_x_factor*2**(-4/3)*(rs_factor/d.rs)
    b88 = 1 + beta / x_factor_c * d.xs ** 2 / (1 + gamma * beta * d.xs * torch.arcsinh(d.xs))

    xc = lda * b88 * 2
    return xc * d.n


@torch.jit.script
def gga_x_pbe_custom(d: DensityVars):
    kappa = 0.8040
    mu = 0.2195149727645171
    x2s = 1/(2*(6*np.pi**2)**(1/3))
    s = x2s * d.xs
    pbe = 1 + kappa*(1 - kappa/(kappa + mu*s**2))

    x_factor_c = 3/8*(3/np.pi)**(1/3)*4**(2/3)
    lda_x_factor = -x_factor_c
    rs_factor = (3/(4*np.pi))**(1/3)
    lda = lda_x_factor*2**(-4/3)*(rs_factor/d.rs)

    xc = lda * pbe * 2
    return xc * d.n


@torch.jit.script
def lda_c_vwn_custom(d: DensityVars):
    p = [-0.10498, 0.0621813817393097900698817274255, 3.72744, 12.9352]
    vwn_a = p[0] * p[2] / (p[0] * p[0] + p[0] * p[2] + p[3]) - 1
    vwn_b = 2 * (p[0] * p[2] / (p[0] * p[0] + p[0] * p[2] + p[3]) - 1) + 2
    vwn_c = (
        2 * p[2] * (1 / ((4 * p[3] - p[2] * p[2]) ** 0.5) -
        p[0] / ((p[0] * p[0] + p[0] * p[2] + p[3]) * ((4 * p[3] - p[2] * p[2]) ** 0.5) /
        (p[2] + 2 * p[0])))
    )

    s = torch.sqrt(d.rs)
    vwn_x = s * s + p[2] * s + p[3]
    vwn_y = s - p[0]
    vwn_z = ((4 * p[3] - p[2] * p[2]) ** 0.5) / (2 * s + p[2])
    vwn_f = (
        0.5 * p[1] * (2 * torch.log(s) + vwn_a * torch.log(vwn_x) - vwn_b *
        torch.log(vwn_y) + vwn_c * torch.arctan(vwn_z))
    )

    return vwn_f * d.n


@torch.jit.script
def gga_c_lyp_custom(d: DensityVars):
    A = 0.04918
    B = 0.132
    C = 0.2533
    Dd = 0.349
    CF = 0.3 * ((3 * np.pi * np.pi) ** (2.0 / 3.0))
    icbrtn = pow(d.n, -1.0 / 3.0)
    P = 1 / (1 + Dd * icbrtn)
    omega = torch.exp(-C * icbrtn) * P * pow(d.n, -11.0 / 3.0)
    delta = icbrtn * (C + Dd * P)
    n2 = d.n * d.n
    ret = -A * (4 * d.a * d.b * P / d.n +
                B * omega *
                    (d.a * d.b *
                        (pow(2, 11.0 / 3.0) * CF *
                            (pow(d.a, 8.0 / 3.0) + pow(d.b, 8.0 / 3.0)) +
                        (47.0 - 7.0 * delta) * d.gnn / 18.0 -
                        (2.5 - delta / 18.0) * (d.gaa + d.gbb) -
                        (delta - 11.0) / 9.0 * (d.a * d.gaa + d.b * d.gbb) / d.n) -
                    2.0 / 3.0 * n2 * d.gnn + (2.0 / 3.0 * n2 - d.a * d.a) * d.gbb +
                    (2.0 / 3.0 * n2 - d.b * d.b) * d.gaa))
    return ret


def gga_c_pbe_custom(d: DensityVars):
    def pw92eps(d):
        PW92C_PARAMS = torch.tensor([
            [0.03109070, 0.21370, 7.59570, 3.5876, 1.63820, 0.49294, 1],
            [0.01554535, 0.20548, 14.1189, 6.1977, 3.36620, 0.62517, 1],
            [0.01688690, 0.11125, 10.3570, 3.6231, 0.88026, 0.49671, 1]
        ]).to(d.n)
        def eopt(sqrtr, t):
            return -2 * t[0] * (1 + t[1] * sqrtr * sqrtr) * \
                torch.log(1 + 0.5 / (t[0] * \
                        (sqrtr * \
                        (t[2] + sqrtr * (t[3] + sqrtr * (t[4] + t[5] * sqrtr))))))
        sqrtr = torch.sqrt(d.rs)
        e0 = eopt(sqrtr, PW92C_PARAMS[0])
        return e0

    param_gamma = (1 - math.log(2.0)) / (np.pi * np.pi)
    param_beta_accurate = 0.06672455060314922
    param_beta_gamma = param_beta_accurate / param_gamma

    n_m13 = d.n ** (-1 / 3)
    a_23 = d.a ** (2 / 3)
    b_23 = d.b ** (2 / 3)

    eps = pw92eps(d)
    u = 2 ** (-1 / 3) * n_m13 * n_m13 * (a_23 + b_23)
    d2 = ((1.0 / 12 * (3.0 ** (5.0 / 6.0)) / (np.pi ** (-1.0 / 6))) ** 2.0) * \
          d.gnn / (u * u * torch.pow(d.n, 7.0 / 3.0))
    u3 = u ** 3
    A = param_beta_gamma / (torch.exp(-eps / (param_gamma * u3)) - 1)
    d2A = d2 * A
    H = param_gamma * u3 * torch.log(1 + param_beta_gamma * d2 * (1 + d2A) / (1 + d2A * (1 + d2A)))
    pbec = d.n * (eps + H)

    return pbec


@torch.jit.script
def mock_zero_functional(d: DensityVars):
    return d.n - d.n


FUNCTIONALS = {
    'ZERO': mock_zero_functional,
    'LDA_K_TF': lda_k_tf_custom, 'GGA_K_VW': gga_k_vw_custom, 'GGA_K_APBE': gga_k_apbe_custom,
    'LDA_X': lda_x_lda_custom, 'GGA_X_B88': gga_x_b88_custom, 'GGA_X_PBE': gga_x_pbe_custom,
    'LDA_C_VWN': lda_c_vwn_custom, 'GGA_C_LYP': gga_c_lyp_custom, 'GGA_C_PBE': gga_c_pbe_custom,
}

XCCODE_TO_NPZ_NAME = {
    'ZERO': 'ZERO',
    'LDA_K_TF': 'Ttf_aux_default', 'GGA_K_VW': 'Tvw_aux_default',
    'LDA_X': 'X_LDA_aux_default', 'GGA_X_B88': 'X_B88_aux_default',
    'LDA_C_VWN': 'C_VWN_aux_default', 'GGA_C_LYP': 'C_LYP_aux_default',
    'GGA_K_APBE': 'Tapbe_aux_default', 'GGA_X_PBE': 'X_PBE_aux_default', 'GGA_C_PBE': 'C_PBE_aux_default', 
}


def eval_xc_func(xcfunc: Callable[[DensityVars], torch.Tensor], d: DensityVars, grid_weights: torch.Tensor):
    return (torch.where(d.mask, d.n.new_tensor(0.0), xcfunc(d)) * grid_weights).sum()


def build_tsxc(c: Dict[str, float], bias: float = 0.0):
    def f(d: DensityVars, grid_weights: torch.Tensor):
        fxc = {n: eval_xc_func(FUNCTIONALS[n], d, grid_weights) for n in c.keys()}
        weighted = {n: fxc[n] * c[n] for n in c.keys()}
        exc = sum(weighted.values())
        return exc + bias, fxc
    def numpy_f(npz):
        val = lambda xccode: npz[XCCODE_TO_NPZ_NAME[xccode]] if xccode != 'ZERO' else 0.0
        return sum(c[xccode] * val(xccode) for xccode in c) + bias
    f.coeffs = c
    f.numpy = numpy_f
    f.is_empty = len(c) == 0 or (len(c) == 1 and 'ZERO' in c)
    return f


def compute_functionals_libxc(d: DensityVars, grid_weights: torch.Tensor):
    rho = d.rho.detach().cpu().numpy()
    weights = grid_weights.cpu().numpy() * rho[0]
    return {
        xcname: weights @ pyscf.dft.libxc.eval_xc(xcname, rho)[0]
        for xcname in FUNCTIONALS.keys()
    }


def eval_xc_libxc(xccode: str, d: DensityVars, grid_weights: torch.Tensor):
    rho = d.rho.detach().cpu().numpy()
    weights = grid_weights.cpu().numpy() * rho[0]
    return weights @ pyscf.dft.libxc.eval_xc(xccode, rho)[0]
