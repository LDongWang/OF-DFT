#!/bin/sh
set -a 

save_root=outputs/Ethanol.MOFDFT
MOLECULE=ethanol.pbe
CKPT_PATH=ckpts/Ethanol.MOFDFT.pt
REPARAM_SPEC='ethanol_pbe:Ts_res[atomref]:v7_diis:ref_v1'
PREDICTION_TYPE=Ts_res
OUTPUT_ROOT=${save_root}
TAG=5e-4
STEPS=1000
LR=5e-4
INIT=minao
EXTRACMD="--use-svd --use-local-frame --grid-level 2 --task-id -1 --task-count -1 --evaluate-force --add-delta-at-init --coeff-dim 361"

## run M-OFDFT
bash scripts/evaluate/eval_flexible.sh
## calculate satistics
path=$OUTPUT_ROOT/total.csv
python statistic.py --mode IID --path $path --molecule $MOLECULE --eval-mode relative
