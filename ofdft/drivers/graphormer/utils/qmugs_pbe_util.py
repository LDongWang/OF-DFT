from .qmugs_pbe_10bins import qmugs_bin0, qmugs_bin0_ts_res, qmugs_bin1, qmugs_bin2, qmugs_bin3, qmugs_bin4, qmugs_bin5, qmugs_bin6, qmugs_bin7, qmugs_bin8, qmugs_bin9

qmugs_pbe_energy_mean_minao = {
    'TsExc_ref': {
        'v7_diis': 0,
    },
    'Ts_res_ref': {
        'v7_diis': 0,
    },
}

qmugs_pbe_energy_std_minao = {
    'TsExc_ref': {
        'v7_diis': 1,
    },
    'Ts_res_ref': {
        'v7_diis': 1,
    },
}

qmugs_pbe_energy_mean = {
    'qm9_pbe': {
        'minao': qmugs_pbe_energy_mean_minao,
    },
    'qmugs_bin1_pbe': {
        'minao': qmugs_pbe_energy_mean_minao,
    },
    'qmugs_bin2_pbe': {
        'minao': qmugs_pbe_energy_mean_minao,
    },
    'qmugs_bin3_pbe': {
        'minao': qmugs_pbe_energy_mean_minao,
    },
    'qmugs_bin4_pbe': {
        'minao': qmugs_pbe_energy_mean_minao,
    },
    'qmugs_bin5_pbe': {
        'minao': qmugs_pbe_energy_mean_minao,
    },
    'qmugs_bin6_pbe': {
        'minao': qmugs_pbe_energy_mean_minao,
    },
    'qmugs_bin7_pbe': {
        'minao': qmugs_pbe_energy_mean_minao,
    },
    'qmugs_bin8_pbe': {
        'minao': qmugs_pbe_energy_mean_minao,
    },
    'qmugs_bin9_pbe': {
        'minao': qmugs_pbe_energy_mean_minao,
    },
}

qmugs_pbe_energy_std = {
    'qm9_pbe': {
        'minao': qmugs_pbe_energy_std_minao,
    },
    'qmugs_bin1_pbe': {
        'minao': qmugs_pbe_energy_std_minao,
    },
    'qmugs_bin2_pbe': {
        'minao': qmugs_pbe_energy_std_minao,
    },
    'qmugs_bin3_pbe': {
        'minao': qmugs_pbe_energy_std_minao,
    },
    'qmugs_bin4_pbe': {
        'minao': qmugs_pbe_energy_std_minao,
    },
    'qmugs_bin5_pbe': {
        'minao': qmugs_pbe_energy_std_minao,
    },
    'qmugs_bin6_pbe': {
        'minao': qmugs_pbe_energy_std_minao,
    },
    'qmugs_bin7_pbe': {
        'minao': qmugs_pbe_energy_std_minao,
    },
    'qmugs_bin8_pbe': {
        'minao': qmugs_pbe_energy_std_minao,
    },
    'qmugs_bin9_pbe': {
        'minao': qmugs_pbe_energy_std_minao,
    },
}

qmugs_pbe_coeff_mean = {
    'qm9_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin0.coeff_mean_v7_minao_diis,
                'Ts_res': qmugs_bin0.coeff_mean_v7_minao_diis,
            },
        },
    },
    'qmugs_bin1_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin1.coeff_mean_v7_minao_diis
            },
        },
    },
    'qmugs_bin2_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin2.coeff_mean_v7_minao_diis
            },
        },
    },
    'qmugs_bin3_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin3.coeff_mean_v7_minao_diis
            },
        },
    },
    'qmugs_bin4_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin4.coeff_mean_v7_minao_diis
            },
        },
    },
    'qmugs_bin5_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin5.coeff_mean_v7_minao_diis
            },
        },
    },
    'qmugs_bin6_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin6.coeff_mean_v7_minao_diis
            },
        },
    },
    'qmugs_bin7_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin7.coeff_mean_v7_minao_diis
            },
        },
    },
    'qmugs_bin8_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin8.coeff_mean_v7_minao_diis
            },
        },
    },
    'qmugs_bin9_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin9.coeff_mean_v7_minao_diis
            },
        },
    },
}

qmugs_pbe_grad_mean = {
    'qm9_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin0.grad_mean_v7_minao_diis_tsexc,
                'Ts_res': qmugs_bin0_ts_res.grad_mean_v7_minao_diis_ts_res,
            },
        },
    },
    'qmugs_bin1_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin1.grad_mean_v7_minao_diis_tsexc,
            },
        },
    },
    'qmugs_bin2_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin2.grad_mean_v7_minao_diis_tsexc,
            },
        },
    },
    'qmugs_bin3_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin3.grad_mean_v7_minao_diis_tsexc,
            },
        },
    },
    'qmugs_bin4_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin4.grad_mean_v7_minao_diis_tsexc,
            },
        },
    },
    'qmugs_bin5_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin5.grad_mean_v7_minao_diis_tsexc,
            },
        },
    },
    'qmugs_bin6_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin6.grad_mean_v7_minao_diis_tsexc,
            },
        },
    },
    'qmugs_bin7_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin7.grad_mean_v7_minao_diis_tsexc,
            },
        },
    },
    'qmugs_bin8_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin8.grad_mean_v7_minao_diis_tsexc,
            },
        },
    },
    'qmugs_bin9_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin9.grad_mean_v7_minao_diis_tsexc,
            },
        },
    },
}

qmugs_pbe_reparams = {
    'qm9_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': {
                    'ref_v1': qmugs_bin0.ref_reparam_v7_minao_diis_tsexc_v1
                }, 
                'Ts_res': {
                    'ref_v1': qmugs_bin0_ts_res.ref_reparam_v7_minao_diis_ts_res_v1
                }, 
            },
        },
    },
    'qmugs_bin1_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': {
                    'ref_v1': qmugs_bin1.ref_reparam_v7_minao_diis_tsexc_v1
                },
            },
        },
    },
    'qmugs_bin2_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': {
                    'ref_v1': qmugs_bin2.ref_reparam_v7_minao_diis_tsexc_v1
                },
            },
        },
    },
    'qmugs_bin3_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': {
                    'ref_v1': qmugs_bin3.ref_reparam_v7_minao_diis_tsexc_v1
                },
            },
        },
    },
    'qmugs_bin4_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': {
                    'ref_v1': qmugs_bin4.ref_reparam_v7_minao_diis_tsexc_v1
                },
            },
        },
    },
    'qmugs_bin5_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': {
                    'ref_v1': qmugs_bin5.ref_reparam_v7_minao_diis_tsexc_v1
                },
            },
        },
    },
    'qmugs_bin6_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': {
                    'ref_v1': qmugs_bin6.ref_reparam_v7_minao_diis_tsexc_v1
                },
            },
        },
    },
    'qmugs_bin7_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': {
                    'ref_v1': qmugs_bin7.ref_reparam_v7_minao_diis_tsexc_v1
                },
            },
        },
    },
    'qmugs_bin8_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': {
                    'ref_v1': qmugs_bin8.ref_reparam_v7_minao_diis_tsexc_v1
                },
            },
        },
    },
    'qmugs_bin9_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': {
                    'ref_v1': qmugs_bin9.ref_reparam_v7_minao_diis_tsexc_v1
                },
            },
        },
    },
}

qmugs_pbe_delta_coeff_mean = {
    'qm9_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin0.delta_coeff_mean_v7_minao_diis,
                'Ts_res': qmugs_bin0.delta_coeff_mean_v7_minao_diis,
            }
        },
    },
    'qmugs_bin1_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin1.delta_coeff_mean_v7_minao_diis,
            }
        },
    },
    'qmugs_bin2_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin2.delta_coeff_mean_v7_minao_diis,
            }
        },
    },
    'qmugs_bin3_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin3.delta_coeff_mean_v7_minao_diis,
            }
        },
    },
    'qmugs_bin4_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin4.delta_coeff_mean_v7_minao_diis,
            }
        },
    },
    'qmugs_bin5_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin5.delta_coeff_mean_v7_minao_diis,
            }
        },
    },
    'qmugs_bin6_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin6.delta_coeff_mean_v7_minao_diis,
            }
        },
    },
    'qmugs_bin7_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin7.delta_coeff_mean_v7_minao_diis,
            }
        },
    },
    'qmugs_bin8_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin8.delta_coeff_mean_v7_minao_diis,
            }
        },
    },
    'qmugs_bin9_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin9.delta_coeff_mean_v7_minao_diis,
            }
        },
    },
}

qmugs_pbe_delta_coeff_std = {
    'qm9_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin0.delta_coeff_std_v7_minao_diis,
                'Ts_res': qmugs_bin0.delta_coeff_std_v7_minao_diis,
            }
        },
    },
    'qmugs_bin1_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin1.delta_coeff_std_v7_minao_diis,
            }
        },
    },
    'qmugs_bin2_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin2.delta_coeff_std_v7_minao_diis,
            }
        },
    },
    'qmugs_bin3_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin3.delta_coeff_std_v7_minao_diis,
            }
        },
    },
    'qmugs_bin4_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin4.delta_coeff_std_v7_minao_diis,
            }
        },
    },
    'qmugs_bin5_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin5.delta_coeff_std_v7_minao_diis,
            }
        },
    },
    'qmugs_bin6_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin6.delta_coeff_std_v7_minao_diis,
            }
        },
    },
    'qmugs_bin7_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin7.delta_coeff_std_v7_minao_diis,
            }
        },
    },
    'qmugs_bin8_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin8.delta_coeff_std_v7_minao_diis,
            }
        },
    },
    'qmugs_bin9_pbe':{
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin9.delta_coeff_std_v7_minao_diis,
            }
        },
    },
}

qmugs_pbe_linear_coeffs = {
    'qm9_pbe': {
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin0.linear_coeffs_minao_diis_tsexc,
                'Ts_res': qmugs_bin0_ts_res.linear_coeffs_minao_diis_ts_res,
            },
        },
    },
    'qmugs_bin1_pbe': {
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin1.linear_coeffs_minao_diis_tsexc
            },
        },
    },
    'qmugs_bin2_pbe': {
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin2.linear_coeffs_minao_diis_tsexc
            },
        },
    },
    'qmugs_bin3_pbe': {
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin3.linear_coeffs_minao_diis_tsexc
            },
        },
    },
    'qmugs_bin4_pbe': {
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin4.linear_coeffs_minao_diis_tsexc
            },
        },
    },
    'qmugs_bin5_pbe': {
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin5.linear_coeffs_minao_diis_tsexc
            },
        },
    },
    'qmugs_bin6_pbe': {
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin6.linear_coeffs_minao_diis_tsexc
            },
        },
    },
    'qmugs_bin7_pbe': {
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin7.linear_coeffs_minao_diis_tsexc
            },
        },
    },
    'qmugs_bin8_pbe': {
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin8.linear_coeffs_minao_diis_tsexc
            },
        },
    },
    'qmugs_bin9_pbe': {
        'minao': {
            'v7_diis': {
                'TsExc': qmugs_bin9.linear_coeffs_minao_diis_tsexc
            },
        },
    },
}