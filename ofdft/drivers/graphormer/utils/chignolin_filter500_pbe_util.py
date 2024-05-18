from .chignolin_filter500_pbe_bins import chignolin_filter500_bin2_total_nograd, chignolin_filter500_bin3_total_nograd, chignolin_filter500_bin4_total_nograd, chignolin_filter500_bin5_total_nograd

chignolin_filter500_pbe_energy_mean_minao = {
    'TsExc_ref': {
        'v7_diis': 0,
    },
    'total_ref': {
        'v7_diis': 0,
    },
    'total_nograd_ref': {
        'v7_diis': 0,
    },
}

chignolin_filter500_pbe_energy_std_minao = {
    'TsExc_ref': {
        'v7_diis': 1,
    },
    'total_ref': {
        'v7_diis': 1,
    },
    'total_nograd_ref': {
        'v7_diis': 1,
    },
}

chignolin_filter500_pbe_energy_mean = {
    'chignolin_filter500_bin2_pbe': {
        'minao': chignolin_filter500_pbe_energy_mean_minao,
    },
    'chignolin_filter500_bin3_pbe': {
        'minao': chignolin_filter500_pbe_energy_mean_minao,
    },
    'chignolin_filter500_bin4_pbe': {
        'minao': chignolin_filter500_pbe_energy_mean_minao,
    },
    'chignolin_filter500_bin5_pbe': {
        'minao': chignolin_filter500_pbe_energy_mean_minao,
    },
}

chignolin_filter500_pbe_energy_std = {
    'chignolin_filter500_bin2_pbe': {
        'minao': chignolin_filter500_pbe_energy_std_minao,
    },
    'chignolin_filter500_bin3_pbe': {
        'minao': chignolin_filter500_pbe_energy_std_minao,
    },
    'chignolin_filter500_bin4_pbe': {
        'minao': chignolin_filter500_pbe_energy_std_minao,
    },
    'chignolin_filter500_bin5_pbe': {
        'minao': chignolin_filter500_pbe_energy_std_minao,
    },
}

chignolin_filter500_pbe_coeff_mean = {
    'chignolin_filter500_bin2_pbe':{
        'minao': {
            'v7_diis': {
                'total_nograd': chignolin_filter500_bin2_total_nograd.coeff_mean_v7_minao_diis
            },
        },
    },
    'chignolin_filter500_bin3_pbe':{
        'minao': {
            'v7_diis': {
                'total_nograd': chignolin_filter500_bin3_total_nograd.coeff_mean_v7_minao_diis
            },
        },
    },
    'chignolin_filter500_bin4_pbe':{
        'minao': {
            'v7_diis': {
                'total_nograd': chignolin_filter500_bin4_total_nograd.coeff_mean_v7_minao_diis
            },
        },
    },
    'chignolin_filter500_bin5_pbe':{
        'minao': {
            'v7_diis': {
                'total_nograd': chignolin_filter500_bin5_total_nograd.coeff_mean_v7_minao_diis,
            },
        },
    },
}

chignolin_filter500_pbe_grad_mean = {
    'chignolin_filter500_bin2_pbe':{
        'minao': {
            'v7_diis': {
                'total_nograd': chignolin_filter500_bin2_total_nograd.grad_mean_v7_minao_diis_tsexc,
            },
        },
    },
    'chignolin_filter500_bin3_pbe':{
        'minao': {
            'v7_diis': {
                'total_nograd': chignolin_filter500_bin3_total_nograd.grad_mean_v7_minao_diis_tsexc,
            },
        },
    },
    'chignolin_filter500_bin4_pbe':{
        'minao': {
            'v7_diis': {
                'total_nograd': chignolin_filter500_bin4_total_nograd.grad_mean_v7_minao_diis_tsexc,
            },
        },
    },
    'chignolin_filter500_bin5_pbe':{
        'minao': {
            'v7_diis': {
                'total_nograd': chignolin_filter500_bin5_total_nograd.grad_mean_v7_minao_diis_tsexc,
            },
        },
    },
}

chignolin_filter500_pbe_reparams = {
    'chignolin_filter500_bin2_pbe':{
        'minao': {
            'v7_diis': {
                'total_nograd': {
                    'ref_v1': chignolin_filter500_bin2_total_nograd.ref_reparam_v7_minao_diis_tsexc_v1
                },
            },
        },
    },
    'chignolin_filter500_bin3_pbe':{
        'minao': {
            'v7_diis': {
                'total_nograd': {
                    'ref_v1': chignolin_filter500_bin3_total_nograd.ref_reparam_v7_minao_diis_tsexc_v1
                },
            },
        },
    },
    'chignolin_filter500_bin4_pbe':{
        'minao': {
            'v7_diis': {
                'total_nograd': {
                    'ref_v1': chignolin_filter500_bin4_total_nograd.ref_reparam_v7_minao_diis_tsexc_v1
                },
            },
        },
    },
    'chignolin_filter500_bin5_pbe':{
        'minao': {
            'v7_diis': {
                'total_nograd': {
                    'ref_v1': chignolin_filter500_bin5_total_nograd.ref_reparam_v7_minao_diis_tsexc_v1
                },
            },
        },
    },
}

chignolin_filter500_pbe_delta_coeff_mean = {
    'chignolin_filter500_bin2_pbe':{
        'minao': {
            'v7_diis': {
                'total_nograd': chignolin_filter500_bin2_total_nograd.delta_coeff_mean_v7_minao_diis,
            }
        },
    },
    'chignolin_filter500_bin3_pbe':{
        'minao': {
            'v7_diis': {
                'total_nograd': chignolin_filter500_bin3_total_nograd.delta_coeff_mean_v7_minao_diis,
            }
        },
    },
    'chignolin_filter500_bin4_pbe':{
        'minao': {
            'v7_diis': {
                'total_nograd': chignolin_filter500_bin4_total_nograd.delta_coeff_mean_v7_minao_diis,
            }
        },
    },
    'chignolin_filter500_bin5_pbe':{
        'minao': {
            'v7_diis': {
                'total_nograd': chignolin_filter500_bin5_total_nograd.delta_coeff_mean_v7_minao_diis,
            }
        },
    },
}

chignolin_filter500_pbe_delta_coeff_std = {
    'chignolin_filter500_bin2_pbe':{
        'minao': {
            'v7_diis': {
                'total_nograd': chignolin_filter500_bin2_total_nograd.delta_coeff_std_v7_minao_diis,
            }
        },
    },
    'chignolin_filter500_bin3_pbe':{
        'minao': {
            'v7_diis': {
                'total_nograd': chignolin_filter500_bin3_total_nograd.delta_coeff_std_v7_minao_diis,
            }
        },
    },
    'chignolin_filter500_bin4_pbe':{
        'minao': {
            'v7_diis': {
                'total_nograd': chignolin_filter500_bin4_total_nograd.delta_coeff_std_v7_minao_diis,
            }
        },
    },
    'chignolin_filter500_bin5_pbe':{
        'minao': {
            'v7_diis': {
                'total_nograd': chignolin_filter500_bin5_total_nograd.delta_coeff_std_v7_minao_diis,
            }
        },
    },
}

chignolin_filter500_pbe_linear_coeffs = {
    'chignolin_filter500_bin2_pbe': {
        'minao': {
            'v7_diis': {
                'total_nograd': chignolin_filter500_bin2_total_nograd.linear_coeffs_minao_diis_tsexc
            },
        },
    },
    'chignolin_filter500_bin3_pbe': {
        'minao': {
            'v7_diis': {
                'total_nograd': chignolin_filter500_bin3_total_nograd.linear_coeffs_minao_diis_tsexc
            },
        },
    },
    'chignolin_filter500_bin4_pbe': {
        'minao': {
            'v7_diis': {
                'total_nograd': chignolin_filter500_bin4_total_nograd.linear_coeffs_minao_diis_tsexc
            },
        },
    },
    'chignolin_filter500_bin5_pbe': {
        'minao': {
            'v7_diis': {
                'total_nograd': chignolin_filter500_bin5_total_nograd.linear_coeffs_minao_diis_tsexc,
            },
        },
    },
}