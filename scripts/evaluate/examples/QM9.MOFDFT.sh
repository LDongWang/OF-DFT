#!/bin/sh
set -a 

save_root=outputs/QM9.MOFDFT
MOLECULE=qm9.pbe.isomer
CKPT_PATH=ckpts/QM9.MOFDFT.pt
REPARAM_SPEC='qm9_pbe:Ts_res[atomref]:v7_diis:ref_v1'
PREDICTION_TYPE=Ts_res

OUTPUT_ROOT=${save_root}
TAG=1e-3
STEPS=1000
LR=1e-3
INIT=minao
EXTRACMD="--use-svd --use-local-frame --grid-level 2 --task-id -1 --task-count -1 --evaluate-force --add-delta-at-init"

## run M-OFDFT
bash scripts/evaluate/eval_flexible.sh
## calculate satistics
path=$OUTPUT_ROOT/total.csv
python statistic.py --mode IID --path $path --molecule $MOLECULE --eval-mode relative
