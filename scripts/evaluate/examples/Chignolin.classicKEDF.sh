#!/bin/sh
KEDF=${1:-TF}
set -a 

save_root=outputs/Chignolin.OFDFT_classical
MOLECULE=chignolin.pbe
PREDICTION_TYPE=Ts_res
INIT_TYPE=minao
TS_FUNC=$KEDF #TF, TFVW, TFVW1.1, APBE
echo "Set KEDF to $TS_FUNC"

INIT="minao"
OUTPUT_ROOT=${save_root}/${TS_FUNC}
TAG=1e-3
STEPS=1000
LR=1e-3
EXTRACMD="--task-count -1 --evaluate-force --grid-level 1 --model-type ofdft --grid-slice-size=8096" 

## run OFDFT
bash scripts/evaluate/eval_flexible_ofdft.sh
## calculate satistics
path=$OUTPUT_ROOT/total.csv
python statistic.py --mode OOD --path $path --molecule $MOLECULE --eval-mode relative

