#!/bin/sh
KEDF=${1:-TF}
set -a 

save_root=outputs/QM9.OFDFT_classical
MOLECULE=qm9.pbe.isomer
PREDICTION_TYPE=Ts_res

TS_FUNC=$KEDF #TF, TFVW, TFVW1.1, APBE
echo "Set KEDF to $TS_FUNC"
INIT="minao"
OUTPUT_ROOT=${save_root}/${TS_FUNC}
TAG=1e-3
STEPS=1000
LR=1e-3
EXTRACMD="--task-id -1 --task-count -1 --model-type ofdft --evaluate-force --grid-level 2"

## run OFDFT
bash scripts/evaluate/eval_flexible_ofdft.sh
## calculate satistics
path=$OUTPUT_ROOT/total.csv
python statistic.py --mode IID --path $path --molecule $MOLECULE --eval-mode relative
