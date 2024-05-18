#!/bin/sh
KEDF=${1:-TF}
set -a 

save_root=outputs/Ethanol.OFDFT_classical
MOLECULE=ethanol.pbe
PREDICTION_TYPE=Ts_res
INIT=minao

TS_FUNC=$KEDF #TF, TFVW, TFVW1.1, APBE
echo "Set KEDF to $TS_FUNC"
OUTPUT_ROOT=${save_root}/${TS_FUNC}
TAG=5e-4
STEPS=1000
LR=5e-4
EXTRACMD="--task-id -1 --task-count -1 --model-type ofdft --evaluate-force --grid-level 2"

## run OFDFT
bash scripts/evaluate/eval_flexible_ofdft.sh
## calculate satistics
path=$OUTPUT_ROOT/total.csv
python statistic.py --mode IID --path $path --molecule $MOLECULE --eval-mode relative
