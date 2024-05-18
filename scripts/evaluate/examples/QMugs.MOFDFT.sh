#!/bin/sh
set -a 

save_root=outputs/QMugs.MOFDFT
MOLECULE=qmugs.bin2_18.pbe
CKPT_PATH=ckpts/QMugs.bin1.MOFDFT.pt
REPARAM_SPEC='qmugs_bin1_pbe:TsExc[atomref]:v7_diis:ref_v1'
PREDICTION_TYPE=TsExc

OUTPUT_ROOT=${save_root}
TAG=1e-3
STEPS=1000
LR=1e-3
INIT=minao
EXTRACMD="--use-svd --use-local-frame --bin-id -1 --evaluate-force --add-delta-at-init"

## run M-OFDFT
bash scripts/evaluate/eval_flexible.sh
## calculate satistics
path=$OUTPUT_ROOT/total.csv
python statistic.py --mode OOD --path $path --molecule $MOLECULE --eval-mode absolute
