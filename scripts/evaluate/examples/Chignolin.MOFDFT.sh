#!/bin/sh
set -a 

save_root=outputs/Chignolin.MOFDFT
MOLECULE=chignolin.pbe
CKPT_PATH=ckpts/Chignolin.pep5.MOFDFT.pt
REPARAM_SPEC='chignolin_filter500_bin5_pbe:total_nograd[atomref]:v7_diis:ref_v1'
PREDICTION_TYPE=total
INIT=minao

OUTPUT_ROOT=${save_root}
TAG=1e-3
STEPS=1000
LR=1e-3
EXTRACMD="--use-svd --use-local-frame --task-count -1 --evaluate-force --add-delta-at-init --coeff-dim 361" 

## run M-OFDFT
bash scripts/evaluate/eval_flexible.sh
## calculate satistics
path=$OUTPUT_ROOT/total.csv
python statistic.py --mode OOD --path $path --molecule $MOLECULE --eval-mode relative
