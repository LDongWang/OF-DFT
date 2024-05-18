# First, we define:
#   molecule_type in ['qm9', 'ethanol']
#   energy_option in ['Ts', 'Ts_res', 'TsExc']; optional [atomref] (e.g. TsExc[atomref])
#   coeff_option in ['orig', 'rep']
#   reparam_version in ['v1', 'v2', ...]
# Then, query like this:
#   energy_mean[molecule_type][energy_option]
#   coeff_mean[molecule_type][coeff_option]
#   reparam_factors[molecule_type][coeff_option][energy_option][version]
# Use a spec string: molecule_type:energy_option:coeff_option:reparam_version
#   example: qm9:Ts_res:orig:v1, ethanol:TsExc[atomref]:LF_alt_SVD:v1

import functools
from .util_general import energy_mean, energy_std, coeff_mean, grad_mean, reparam_factors, atomic_linear_coeffs, delta_coeff_mean, delta_coeff_std

@functools.lru_cache(1)
def parse_spec_str(spec_str):
    spec_parts = spec_str.split(':')
    if len(spec_parts) == 4:
        molecule_type, energy_option, coeff_option, reparam_version = spec_parts
        init_method = 'minao'
    elif len(spec_parts) == 5:
        molecule_type, energy_option, coeff_option, reparam_version, init_method = spec_parts
    else:
        raise NotImplementedError()
    from argparse import Namespace
    spec = Namespace()
    spec.molecule_type = molecule_type
    if energy_option.endswith('[atomref]'):
        spec.energy_option = energy_option.replace('[atomref]', '')
        spec.use_atomref = True
    else:
        spec.energy_option = energy_option
        spec.use_atomref = False
    spec.coeff_option = coeff_option
    spec.reparam_version = reparam_version
    spec.init_method = init_method

    return spec


def get_energy_mean(spec):
    if spec.use_atomref:
        return energy_mean[spec.molecule_type][spec.init_method][spec.energy_option+'_ref'][spec.coeff_option]
    else:
        return energy_mean[spec.molecule_type][spec.init_method][spec.energy_option]


def get_energy_std(spec):
    if spec.use_atomref:
        return energy_std[spec.molecule_type][spec.init_method][spec.energy_option+'_ref'][spec.coeff_option]
    else:
        return energy_std[spec.molecule_type][spec.init_method][spec.energy_option]


def get_coeff_mean(spec):
    return coeff_mean[spec.molecule_type][spec.init_method][spec.coeff_option][spec.energy_option]


def get_grad_mean(spec):
    return grad_mean[spec.molecule_type][spec.init_method][spec.coeff_option][spec.energy_option]


def get_reparam_factors(spec):
    return reparam_factors[spec.molecule_type][spec.init_method][spec.coeff_option][spec.energy_option][spec.reparam_version]


def get_atomic_linear_coeffs(spec):
    return atomic_linear_coeffs[spec.molecule_type][spec.init_method][spec.coeff_option][spec.energy_option]


def get_delta_coeff_mean(spec):
    return delta_coeff_mean[spec.molecule_type][spec.init_method][spec.coeff_option][spec.energy_option]


def get_delta_coeff_std(spec):
    return delta_coeff_std[spec.molecule_type][spec.init_method][spec.coeff_option][spec.energy_option]


def get_reparam(spec_str):
    spec = parse_spec_str(spec_str)
    return spec, get_energy_mean(spec), get_energy_std(spec), get_coeff_mean(spec), get_reparam_factors(spec)