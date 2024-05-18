import os
import sys
import shutil
import random
import argparse
from datetime import datetime
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import pyscf
import pyscf.df

import ofdft.init
import ofdft.grad

XC = 'PBE'
XC_C = {'GGA_X_PBE': 1.0, 'GGA_C_PBE': 1.0}
TIMEOUT = None

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--ngpu', type=int, help='Number of GPUs. CPU is used if this is set to 0.')
    parser.add_argument('--nworker', type=int, help='Number of worker processes.')
    parser.add_argument('--molecule', type=str, required=True, choices=['qm9.pbe.isomer', 'ethanol.pbe', 'qmugs.bin2_18.pbe', 'chignolin.pbe'])
    parser.add_argument('--model-type', type=str, default='graphormer', choices=['graphormer', 'ofdft'])
    parser.add_argument('--use-ema', action='store_true')
    parser.add_argument('--init', type=str, required=True,
                        choices=['minao', 'huckel'])
    parser.add_argument('--task-count', type=int, default=4,
                        help='Number of tasks to be sampled. A number < 0 indicates evaluation on the whole set.'
                             'Will be ignored if --task-list is specified.')
    parser.add_argument('--task-list', type=str,
                        help='Path to a textfile containing ids of the tasks to be run, one per line.'
                             'Overwrites --task-count if specified.')
    parser.add_argument('--task-id', type=int,
                        help='Selected molecule index.'
                             'Overwrites --task-count and --task-list if specified.')
    parser.add_argument('--bin-id', type=int,
                        help='Selected bin index.'
                             'Only valid when molecule type is qmugs.bin2-18')
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument('--reparam-spec', type=str, required=True)
    parser.add_argument('--prediction-type', type=str, required=True,
                        choices=['Ts', 'Ts_res', 'TsExc', 'total'])
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--evaluate-rho', action='store_true')
    parser.add_argument('--evaluate-force', action='store_true')
    parser.add_argument('--use-svd', action='store_true')
    parser.add_argument('--use-local-frame', action='store_true')
    parser.add_argument('--add-delta-at-init', action='store_true')
    parser.add_argument('--add-delta-interval', type=int, default=0)
    parser.add_argument('--grid-level', type=int)
    parser.add_argument('--grid-type', type=str, default='basic', choices=['basic', 'lazy', 'disk_lazy'])
    parser.add_argument('--grid-slice-size', type=int, default=32768)
    parser.add_argument('--save-coeff-interval', type=int, default=0)
    parser.add_argument('--compute-init-energy', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--init-delta-ratio', type=float, default=1)
    parser.add_argument('--coeff-dim', type=int, default=477)
    parser.add_argument('--ts-func', type=str, default='APBE', choices=['APBE', 'TF', 'TFVW', 'TFVW1.1'])
    parser.add_argument('--use-grad-momentum', action='store_true')
    parser.add_argument('--grad-momentum-lambda', type=float, default=0.5)

    args = parser.parse_args()

    args.ngpu = args.ngpu or int(os.getenv('NGPU', -1))
    args.nworker = args.nworker or int(os.getenv('NWORKER', -1)) or args.ngpu

    print(f'NGPU: {args.ngpu}, NWORKER: {args.nworker}')

    if args.ngpu == -1 or args.nworker == -1:
        print('Please set ngpu and nworker through command line or env variable!')
        sys.exit()

    return args


def get_data_path(args):
    if args.molecule == 'qm9.pbe.isomer':
        return 'data/QM9/input'
    elif args.molecule == 'ethanol.pbe':
        return 'data/Ethanol/input'
    elif args.molecule == 'qmugs.bin2_18.pbe':
        return 'data/QMugs/input'
    elif args.molecule == 'chignolin.pbe':
        return 'data/Chignolin/input'
    else:
        raise NotImplementedError()


def get_tasks(args):
    if args.molecule == 'qmugs.bin2_18.pbe':
        if args.bin_id is not None and args.bin_id > -1:
            return list(range(50 * args.bin_id, 50 * args.bin_id + 50))
        else:
            return list(range(850))
    if args.task_id is not None and args.task_id > -1:
        print('NOTE: --task-id specified, ignoring --task-list and --task-count...')
        return [args.task_id]
    elif args.task_list is not None:
        print('NOTE: --task-list specified, ignoring --task-count...')
        with open(args.task_list, 'r') as f:
            return list(map(int, filter(None, [line.strip() for line in f.readlines()])))

    random.seed(0)
    if args.molecule == 'qm9.pbe.isomer':
        partition = 'data/QM9/test_id.txt'
    elif args.molecule == 'ethanol.pbe':
        partition = 'data/Ethanol/test_id.txt'
    elif args.molecule == 'qmugs.bin2_18.pbe':
        partition = 'data/QMugs/test_id.txt'
    elif args.molecule == 'chignolin.pbe':
        partition = 'data/Chignolin/test_id.txt'
    else:
        raise NotImplementedError()

    with open(partition, 'r') as f:
        lines = f.readlines()
        full_set = [int(line.strip()) for line in lines]

    if args.task_count < 0 or len(full_set) <= args.task_count:
        tasks = full_set
    else:
        tasks = random.sample(full_set, args.task_count)
    return tasks


def get_data(idx):
    data_path = get_data_path(args)
    example = np.load(os.path.join(data_path, f'{idx}_test_data.npz'), allow_pickle=True)['data'].item()
    return example


def load_model(args, device):
    if args.model_type == 'graphormer':
        from ofdft.drivers.graphormer import load_model
        model = load_model(args.ckpt_path, use_ema=args.use_ema) # model is shared across mols
    elif args.model_type == 'ofdft':
        return None
    else:
        raise NotImplementedError()
    model = model.to(device)
    return model


def get_tsbase_and_xc_and_forward_spec(args):
    import ofdft.functionals
    if args.prediction_type == 'Ts':
        tsbase = ofdft.functionals.build_tsxc({'ZERO': 1.0})
        xc = ofdft.functionals.build_tsxc(XC_C)
        forward_spec = ['xc', 'corr', 'j', 'vext']
    elif args.prediction_type == 'Ts_res':
        if args.ts_func == 'APBE':
            tsbase = {'GGA_K_APBE': 1.0}
        elif args.ts_func == 'TFVW':
            tsbase = {'LDA_K_TF': 1.0, 'GGA_K_VW': 1/9}
        elif args.ts_func == 'TFVW1.1':
            tsbase = {'LDA_K_TF': 1.0, 'GGA_K_VW': 1.0}
        elif args.ts_func == 'TF':
            tsbase = {'LDA_K_TF': 1.0}
        tsbase = ofdft.functionals.build_tsxc(tsbase)
        xc = ofdft.functionals.build_tsxc(XC_C)
        forward_spec = ['tsbase', 'xc', 'corr', 'j', 'vext']
    elif args.prediction_type == 'TsExc':
        tsbase = ofdft.functionals.build_tsxc({'ZERO': 1.0})
        xc = ofdft.functionals.build_tsxc({'ZERO': 1.0})
        forward_spec = ['corr', 'j', 'vext']
    elif args.prediction_type == 'total':
        tsbase = ofdft.functionals.build_tsxc({'ZERO': 1.0})
        xc = ofdft.functionals.build_tsxc({'ZERO': 1.0})
        forward_spec = ['corr']
    else:
        raise NotImplementedError()
    return tsbase, xc, forward_spec


def get_init_coeff_and_energy(args, mol, auxmol):
    if args.init in ['minao', 'huckel']:
        init_dm = ofdft.init.get_init_coeff(f'initguess_{args.init}', mol, auxmol, use_dm=True)
    elif args.init == 'fastminao':
        init_dm = ofdft.init.get_init_coeff(f'initguess_{args.init}', mol, auxmol, use_dm=False)
    elif args.init.startswith('halfgt'):
        from ofdft.init import init_halfgt
        parts = args.init.split(':')
        guess, step = parts[1], int(parts[2])
        init_dm = init_halfgt(mol, auxmol, step, guess=guess, use_dm=True)
    else:
        raise NotImplementedError()
    mf = mol.RKS(xc=XC)
    init_coeff = ofdft.init.get_rho_coeff(mol, auxmol, init_dm)
    if args.compute_init_energy:
        init_energy = mf.energy_elec(dm=init_dm)[0]
    else:
        init_energy = float('nan')
    return init_coeff, init_energy


def worker(args, device, model, taskid):
    example = get_data(taskid)
    mol = pyscf.gto.Mole.loads(example['mol'])
    mol = pyscf.M(atom=mol.atom, basis=mol.basis)

    stats_path = os.path.join(args.output_dir, 'csv', f'stats_{taskid}.csv')
    if os.path.exists(stats_path):
        return None
    print(f'processing {taskid}...')

    original_stdout = sys.stdout
    new_stdout_file = open(os.path.join(args.output_dir, 'log', f'log_{taskid}.log'), 'w', buffering=1)
    sys.stdout = new_stdout_file

    start_time = datetime.utcnow()

    auxbasis = ofdft.init.ref_etb(mol, 2.5)
    auxmol = pyscf.df.addons.make_auxmol(mol, auxbasis)
    print('Name:', example['name'])
    print(f'| N(atom) = {mol.natm}, N(elec) = {mol.nelectron}, N(AO) = {mol.nao}, N(AUX) = {auxmol.nao}')

    init_coeff, init_etot = get_init_coeff_and_energy(args, mol, auxmol)

    grid = None
    if (not args.prediction_type in ['TsExc', 'total']) and (args.grid_level is not None):
        grid = pyscf.dft.gen_grid.Grids(mol)
        grid.level = args.grid_level
        grid.build()

    tsbase, xc, forward_spec = get_tsbase_and_xc_and_forward_spec(args)

    if args.model_type == 'graphormer':
        from ofdft.drivers.graphormer import GraphormerOFDFTDriver
        driver = GraphormerOFDFTDriver(
            mol,
            grid=grid,
            grid_type=args.grid_type,
            grid_slice_size=args.grid_slice_size,
            model=model,
            reparam=args.reparam_spec,
            tsbase=tsbase,
            xc=xc,
            init_method=lambda _, __: init_coeff,
            auxbasis=ofdft.init.ref_etb(mol, 2.5),
            normalize_coeff=False,
            init_normalize=False,
            use_local_frame=args.use_local_frame,
            use_svd_space=args.use_svd,
            init_add_delta=False,
            coeff_dim=args.coeff_dim,
        ).to(device)
    elif args.model_type == 'ofdft':
        from ofdft.drivers.mock import MockOFDFTDriver
        driver = MockOFDFTDriver(
            mol,
            grid=grid,
            grid_type=args.grid_type,
            grid_slice_size=args.grid_slice_size,
            tsbase=tsbase,
            xc=xc,
            init_method=lambda _, __: init_coeff,
            auxbasis=ofdft.init.ref_etb(mol, 2.5),
            normalize_coeff=False,
            init_normalize=False,
        ).to(device)
    if args.add_delta_at_init:
        driver.var.data += driver.predict_delta_coeff(project=True) * args.init_delta_ratio

    grid = driver.grid
    # gt_tsxc = driver.xc_fn.numpy(example) + driver.tsbase_fn.numpy(example)
    gt_tsxc = 0
    gt_etot = sum(example['scf_summary'].values()) - example['scf_summary']['nuc']
    gt_coeff = torch.tensor(example['rho_coeff_default']).to(device)
    gt_terms = {
        'j': example['scf_summary']['coul'], 'vext': example['scf_summary']['e1'] - example['Ts'],
        'tsxc': gt_tsxc, 'corr': example['Ts'] + example['scf_summary']['exc'] - gt_tsxc
    }
    if args.prediction_type == 'total':
        init_etot += example['scf_summary']['nuc']
        gt_etot += example['scf_summary']['nuc']
    print(f'| INIT etot:', init_etot)
    print(f'| GT etot:', gt_etot)
    print(f'| GT terms:', gt_terms)

    if args.evaluate_rho:
        grid_weights = torch.tensor(grid.weights)
        gt_rho = driver.ofdft.all_auxao_values() @ gt_coeff.cpu()
    
    if args.molecule in ['qm9.pbe.isomer', 'ethanol.pbe']:
        auxao_ovlp = torch.tensor(driver.auxmol.intor('int1e_ovlp')).to(device)

    if args.evaluate_force:
        force_nuc = torch.tensor(ofdft.grad.grad_nuc(auxmol)).to(device)
        force_ext_derivs = torch.stack([
            torch.tensor(t)
            for t in ofdft.grad.extgrad_generator(auxmol)
        ]).to(device)
        compute_hf_force = lambda c: force_ext_derivs @ c + force_nuc
        gt_hf_force = compute_hf_force(gt_coeff)

    def grad_proj(grad):
        N = driver.norm_vec
        grad_proj = (grad - (grad @ N) / (N @ N) * N)
        return grad_proj

    if args.use_grad_momentum:
        optimizer = torch.optim.SGD([driver.coeff_var], lr=args.lr, momentum=args.grad_momentum_lambda, dampening=1-args.grad_momentum_lambda)
    else:
        optimizer = torch.optim.SGD([driver.coeff_var], lr=args.lr)


    stat_records = []
    all_coeffs = [init_coeff]
    decay_delta_coeff = 0.8
    if args.add_delta_interval > 0:
        print('will recycle add_delta...')

    for i in range(args.steps):
        if args.save_coeff_interval > 0:
            all_coeffs.append(driver.coeff_for_input.detach().cpu().numpy())

        optimizer.zero_grad(set_to_none=True)
        loss, terms, _ = driver.forward_and_backward(forward_parts=forward_spec)
        loss_py = loss.item()

        if args.add_delta_interval > 0 and i > 0 and i % args.add_delta_interval == 0:
            # modify grad
            print('will recycle add_delta...')
            driver.var.grad = driver.predict_delta_coeff(project=False) / args.lr * decay_delta_coeff
            decay_delta_coeff *= 0.8

        total_grad_norm = driver.coeff_var.grad.norm().detach().item()
        projected_grad = grad_proj(driver.coeff_var.grad)
        total_grad_proj_norm = projected_grad.detach().norm()

        if args.evaluate_rho:
            auxrho = driver.auxrho()
            minrho = auxrho.min().item()
            maxrho = auxrho.max().item()
            rho_le0 = (auxrho < 0).sum().item()
            rho_ae = (gt_rho - auxrho).abs()
            rho_mae = rho_ae.mean().item()
            rho_ae_weighted_sum = (rho_ae @ grid_weights).sum().item()
        else:
            minrho = float('nan')
            maxrho = float('nan')
            rho_le0 = float('nan')
            rho_mae = float('nan')
            rho_ae_weighted_sum = float('nan')

        coeff_diff = driver.coeff_for_input - gt_coeff
        coeff_mae = coeff_diff.abs().mean().item()
        # temporalily, we only calculate density loss for small molecules to reduce memory consumption
        if args.molecule in ['qm9.pbe.isomer', 'ethanol.pbe']: 
            density_loss = (coeff_diff @ auxao_ovlp @ coeff_diff).item()
        else:
            density_loss = 0

        if args.evaluate_force:
            hf_force = compute_hf_force(driver.coeff_for_input.detach())
            hf_force_mae = (gt_hf_force - hf_force).abs().mean().item()
        else:
            hf_force_mae = float('nan')

        stat_records.append({
            'step': i,
            'loss': loss_py,
            'min(rho)': minrho,
            'max(rho)': maxrho,
            'count(rho < 0)': rho_le0,
            'projected gradnorm': total_grad_norm,
            'mae(rho)': rho_mae,
            'weighted ae(rho)': rho_ae_weighted_sum,
            'mae(coeff)': coeff_mae,
            'density loss': density_loss,
            'mae(HF force)': hf_force_mae,
            'term[corr]': terms['corr'].detach().item(),
            'term[vext]': terms['vext'].detach().item(),
            'term[j]': terms['j'].detach().item(),
            'term[xc]': terms['xc'].detach().item(),
            'term[tsbase]': terms['tsbase'].detach().item(),
        })

        if not args.quiet:
            print(f'[Step {i+1}] loss = {loss_py} min rho = {minrho} max rho = {maxrho} count(rho < 0) = {rho_le0} corr = {terms["corr"].detach().item()} projgradnorm = {total_grad_proj_norm} gradnorm = {total_grad_norm} rho_mae = {rho_mae} rho_ae_weighted_sum = {rho_ae_weighted_sum} coeff_mae = {coeff_mae} density_loss = {density_loss} hf_force_mae = {hf_force_mae}')# lr = {scheduler.get_last_lr()}')
            print(f'    terms = {dict(map(lambda t: (t[0], t[1].item()), terms.items()))}')

        
        driver.coeff_var.grad = projected_grad

        optimizer.step()

    end_time = datetime.utcnow()
    walltime = (end_time - start_time).total_seconds()

    stats_path = os.path.join(args.output_dir, 'csv', f'stats_{taskid}.csv')
    stats_df = pd.DataFrame.from_records(stat_records)
    stats_df.to_csv(stats_path, index=False)

    etot_last = stats_df.iloc[-1]['loss']


    from scipy.signal import argrelextrema
    gradnorm_local_minima = argrelextrema(stats_df['projected gradnorm'].to_numpy(), np.less)[0]
    if len(gradnorm_local_minima) > 0:
        etot_1st_projgrad_min = stats_df.iloc[gradnorm_local_minima[0]]['loss']
        hf_force_mae_1st_projgrad_min = stats_df.iloc[gradnorm_local_minima[0]]['mae(HF force)']
    else:
        etot_1st_projgrad_min = None
        hf_force_mae_1st_projgrad_min = None

    losses = stats_df['loss'].to_numpy()
    delta_e = abs(losses[1:] - losses[:-1])
    delta_e_min_step = np.argmin(delta_e) + 1
    delta_e_local_minima = argrelextrema(delta_e, np.less)[0] + 1
    if len(delta_e_local_minima) > 0:
        delta_e_local_minima = delta_e_local_minima[:1]
        etot_1st_deltae_min = stats_df.iloc[delta_e_local_minima[0]]['loss']
        hf_force_mae_1st_deltae_min = stats_df.iloc[delta_e_local_minima[0]]['mae(HF force)']
    else:
        etot_1st_deltae_min = None
        hf_force_mae_1st_deltae_min = None
    etot_argmin_deltae = stats_df.iloc[delta_e_min_step]['loss']
    hf_force_mae_argmin_deltae = stats_df.iloc[delta_e_min_step]['mae(HF force)']

    total_ema_delta_e = []
    ema_delta_e = delta_e[0]
    decay_e = 0.5
    for temp_e in delta_e:
        ema_delta_e = (1 - decay_e) * ema_delta_e + decay_e * temp_e
        total_ema_delta_e.append(ema_delta_e)
    ema_delta_e_min_step = np.argmin(total_ema_delta_e) + 1
    etot_argmin_emadeltae = stats_df.iloc[ema_delta_e_min_step]['loss']
    hf_force_mae_argmin_emadeltae = stats_df.iloc[ema_delta_e_min_step]['mae(HF force)']

    sys.stdout = original_stdout
    new_stdout_file.close()

    if args.save_coeff_interval > 0:
        coeffs_to_save = {
            'stop(ARGMIN DELTA-E)': all_coeffs[delta_e_min_step],
            'stop(FIRST DELTA-E LOCAL MINIMA)': all_coeffs[delta_e_local_minima[0]] if len(delta_e_local_minima) > 0 else None,
            'stop(FIRST PROJGRAD LOCAL MINIMA)': all_coeffs[gradnorm_local_minima[0]] if len(gradnorm_local_minima) > 0 else None,
        }
        coeffs_to_save.update({
            f'step({step})': all_coeffs[step]
            for step in range(0, len(all_coeffs), args.save_coeff_interval)
        })
        np.savez_compressed(os.path.join(args.output_dir, 'coeff', f'coeffs_{taskid}.npz'), coeffs_to_save)

    best_e_step = np.argmin(np.abs(stats_df['loss'] - gt_etot))
    best_e_loss = np.min(np.abs(stats_df['loss'] - gt_etot))
    best_f_step = np.argmin(stats_df['mae(HF force)'])
    best_f_loss = np.min(stats_df['mae(HF force)'])
    return {
        'taskid': taskid,
        'time': walltime,
        'nstep': args.steps,
        'natom': mol.natm,
        'nheavy': sum([mol.elements[i].upper() != 'H' for i in range(mol.natm)]),
        'nao': mol.nao,
        'naux': auxmol.nao,
        'nelec': mol.nelectron,

        'init_method': args.init,
        'gt_etot': gt_etot,
        'nuc': example['scf_summary']['nuc'],
        'init_etot': init_etot,

        'etot(FIRST)': stats_df.iloc[0]['loss'], 
        'etot(LAST)': etot_last,
        'etot(FIRST PROJGRAD LOCAL MINIMA)': etot_1st_projgrad_min,
        'etot(FIRST DELTA-E LOCAL MINIMA)': etot_1st_deltae_min,
        'etot(ARGMIN DELTA-E)': etot_argmin_deltae,
        'loss(BEST FORCE)': gt_etot - stats_df.iloc[best_f_step]['loss'],

        'loss(FIRST)': gt_etot - stats_df.iloc[0]['loss'],
        'loss(LAST)': gt_etot - etot_last,
        'loss(FIRST PROJGRAD LOCAL MINIMA)': gt_etot - etot_1st_projgrad_min if etot_1st_projgrad_min is not None else None,
        'loss(FIRST DELTA-E LOCAL MINIMA)': gt_etot - etot_1st_deltae_min if etot_1st_deltae_min is not None else None,
        'loss(ARGMIN DELTA-E)': gt_etot - etot_argmin_deltae if etot_argmin_deltae is not None else None,
        'loss(ARGMIN EMA-DELTA-E)': gt_etot - etot_argmin_emadeltae,

        'HF-force-mae(FIRST)': stats_df.iloc[0]['mae(HF force)'],
        'HF-force-mae(LAST)': stats_df.iloc[-1]['mae(HF force)'],
        'HF-force-mae(FIRST PROJGRAD LOCAL MINIMA)': hf_force_mae_1st_projgrad_min,
        'HF-force-mae(FIRST DELTA-E LOCAL MINIMA)': hf_force_mae_1st_deltae_min,
        'HF-force-mae(ARGMIN DELTA-E)': hf_force_mae_argmin_deltae,
        'HF-force-mae(ARGMIN EMA-DELTA-E)': hf_force_mae_argmin_emadeltae,
        'HF-force-mae(BEST LOSS)': stats_df.iloc[best_e_step]['mae(HF force)'],

        'gradnorm_local_minima': gradnorm_local_minima,
        'deltae_local_minima': delta_e_local_minima,
        'deltae_min_step': delta_e_min_step,
        'ema_delta_e_min_step': ema_delta_e_min_step,

        'best_eloss_step': best_e_step,
        'best_floss_step': best_f_step,
        'best_eloss': best_e_loss,
        'best_floss': best_f_loss,
    }

def worker_wrapper(*args, **kwargs):
    try:
        return worker(*args, **kwargs)
    except Exception as e:
        import traceback
        print(e)
        print(traceback.format_exc())
        return None


def worker_proc(args, rank, task_queue, result_queue):
    if args.ngpu != 0:
        igpu = rank % args.ngpu
        device = f'cuda:{igpu}'
    else:
        device = 'cpu'
    model = load_model(args, device=device)
    while True:
        taskid = task_queue.get()
        result = worker(args, device, model, taskid)
        result_queue.put(result)


if __name__ == '__main__':
    args = parse_args()

    print('OUTPUT_DIR:', args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copyfile(__file__, os.path.join(args.output_dir, 'script.py'))

    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        f.write(str(args.__dict__) + '\n')

    os.makedirs(os.path.join(args.output_dir, 'log'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'csv'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'coeff'), exist_ok=True)

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    processes = []
    for rank in range(args.nworker):
        proc = mp.Process(target=worker_proc, args=(args, rank, task_queue, result_queue), daemon=True)
        proc.start()
        processes.append(proc)
    print('Workers started.')

    total_stats = []
    def save_stats():
        total_stats_df = pd.DataFrame.from_records(total_stats)
        save_path = os.path.join(args.output_dir, 'total.csv')
        if os.path.exists(save_path):
            previous_df = pd.read_csv(save_path)
            total_stats_df = pd.concat([previous_df, total_stats_df], axis=0)
            total_stats_df.drop_duplicates(subset='taskid', inplace=True)
        total_stats_df.to_csv(os.path.join(args.output_dir, 'total.csv'), index=False)

    tasks = get_tasks(args)
    print(f'There are {len(tasks)} tasks ...')
    for task in tasks:
        task_queue.put(task)

    for i in range(len(tasks)):
        res = result_queue.get(timeout=TIMEOUT)
        if res is not None:
            total_stats.append(res)
            save_stats()

    for proc in processes:
        proc.terminate()

