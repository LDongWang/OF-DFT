echo "Molecule: $MOLECULE"
echo "Prediction type: $PREDICTION_TYPE"
echo "Output root: $OUTPUT_ROOT"
echo "Steps: $STEPS"
echo "LR: $LR"
echo "Inits: $INITS"
echo "TS_FUNC: $TS_FUNC"
echo "Extra commandline: $EXTRACMD"
#model
CKPT_PATH=None
REPARAM_SPEC=None

python evaluate.py \
    --molecule $MOLECULE --prediction-type $PREDICTION_TYPE \
    --ckpt-path $CKPT_PATH --reparam-spec $REPARAM_SPEC \
    --init $INIT --steps $STEPS --lr $LR --ts-func $TS_FUNC \
    --output-dir $OUTPUT_ROOT $EXTRACMD
