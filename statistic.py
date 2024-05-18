import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='OOD')
parser.add_argument('--path', type=str, default=None)
parser.add_argument('--molecule', type=str, default=None)
parser.add_argument('--eval-mode', type=str, choices=['relative', 'absolute'], default='absolute')
args = parser.parse_args()


har2kcal = 627.5094740630558
domain2cri = {
    'IID': ['ARGMIN DELTA-E'],
    'OOD': ['FIRST PROJGRAD LOCAL MINIMA', 'FIRST DELTA-E LOCAL MINIMA', 'ARGMIN DELTA-E'],
}
def print_metrics(df, domain):
    criterions = domain2cri[domain]
    eng_loss, natm = [], []
    for index, row in df.iterrows():
        for criterion in criterions:
            if not np.isnan(row[f'loss({criterion})']):
                eng_loss.append(row[f'loss({criterion})'])
                natm.append(row['natom'])
                break
    assert len(eng_loss) == df.shape[0]
    resdf = df[['taskid', 'natom', 'nheavy']]
    resdf['eng_loss'] = eng_loss
    resdf['pred_eng'] = df['gt_etot'] - resdf['eng_loss'] + df['nuc']
    resdf['gt_etot'] = df['gt_etot'] + df['nuc']
    resdf['eng_mae'] = np.abs(resdf['eng_loss']) * har2kcal
    resdf['eng_peratom_mae'] = resdf['eng_mae'] / df['natom']
    resdf['pred_eng'] = resdf['pred_eng'] * har2kcal
    resdf['gt_etot'] = resdf['gt_etot'] * har2kcal
    
    return resdf

eval2molecule = {
    'ethanol.pbe': 'Ethanol',
    'qm9.pbe.isomer': 'QM9',
    'qmugs.bin2_18.pbe': 'QMugs',
    'chignolin.pbe': 'Chignolin'
}

if __name__ == "__main__":
    path = args.path
    molecule = eval2molecule[args.molecule]
    eval_mode = args.eval_mode
    print('--'*30)
    print(f'|{molecule}|path: {path}')
    df = pd.read_csv(path)
    df.drop_duplicates(subset='taskid', inplace=True)
    print(f'|calculating metrics for {df.shape[0]} molecules')
    results = print_metrics(df, args.mode)
    assert results.shape[0] > 0, 'invalid results'
    if results.shape[0] == 1:
        eng_mae, eng_peratom_mae = results.iloc[0]['eng_mae'], results.iloc[0]['eng_peratom_mae']
        print(f'|{molecule}|absolute energy MAE: {eng_mae:.2f} kcal/mol; eng MAE/atom: {eng_peratom_mae:.2f}')
    else:
        if molecule == 'Ethanol':
            if eval_mode == 'absolute':
                eng_mae = np.mean(results['eng_mae'])
                print(f'|Ethanol|absolute energy MAE: {eng_mae:.2f} kcal/mol')
            else:
                results = results.sort_values(by='taskid').reset_index()
                assert results.iloc[0]['taskid'] == 0
                engs = results['pred_eng'].values 
                rel_eng_pred = engs[1:] - engs[0]
                engs = results['gt_etot'].values 
                rel_eng_gt = engs[1:] - engs[0]
                rel_eng_mae = np.mean(np.abs(rel_eng_gt - rel_eng_pred))
                print(f'|Ethanol|relative energy MAE: {rel_eng_mae:.2f} kcal/mol')

        elif molecule == 'QM9':
            if eval_mode == 'absolute':
                eng_mae = np.mean(results['eng_mae'])
                print(f'|QM9|absolute energy MAE: {eng_mae:.2f} kcal/mol')
            else:
                engs = results['pred_eng'].values 
                len_ = len(engs)
                rel_eng_pred = engs.reshape(len_, 1) - engs.reshape(1, len_)
                engs = results['gt_etot'].values 
                rel_eng_gt = engs.reshape(len_, 1) - engs.reshape(1, len_)
                off_diag = np.where(~np.eye(len_, dtype=bool))
                rel_eng_mae = np.mean(np.abs(rel_eng_gt - rel_eng_pred)[off_diag])
                print(f'|QM9|relative energy MAE: {rel_eng_mae:.2f} kcal/mol')

        elif molecule == 'QMugs':
            bins = list(range(16, 101, 5))
            for i in range(len(bins)-1):
                bin_results = results[(results['nheavy'] < bins[i+1]) & (results['nheavy'] >= bins[i])]
                if bin_results.shape[0] < 1:
                    continue
                eng_peratom_mae = np.mean(bin_results['eng_peratom_mae'])
                print(f'|QMugs|bin:{i+2}| per-atom absolute eng MAE: {eng_peratom_mae:.2f} kcal/mol') 

        elif molecule == 'Chignolin':
            if eval_mode == 'absolute':
                eng_mae = np.mean(results['eng_mae'])
                print(f'|Chignolin|absolute energy MAE: {eng_mae:.2f} kcal/mol')
            else:
                engs = results['pred_eng'].values 
                len_ = len(engs)
                rel_eng_pred = engs.reshape(len_, 1) - engs.reshape(1, len_)
                engs = results['gt_etot'].values 
                rel_eng_gt = engs.reshape(len_, 1) - engs.reshape(1, len_)
                off_diag = np.where(~np.eye(len_, dtype=bool))
                rel_eng_mae = np.mean(np.abs(rel_eng_gt - rel_eng_pred)[off_diag]) / 168
                print(f'|Chignolin|per-atom relative energy MAE: {rel_eng_mae:.2f} kcal/mol')
        else:
            raise NotImplementedError()
        
    print('--'*30)
    
