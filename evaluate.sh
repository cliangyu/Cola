export CUDA_VISIBLE_DEVICES=2

export PYTHONPATH=.
export DATA_DIR=./datasets/
# export DATASET=esnlive
# export DATASET=aokvqa
# export DATASET=vsr
# export DATASET=okvqa
# export DATASET=vqav2
export DATASET=clevr
# export DATASET=gqa
export SPLIT=val
# export SPLIT=test
export LOG_DIR=./logs/
export PREDS_DIR=./predictions
export PT_MODEL_DIR=./pretrained_models/

# export PREFIX=aok_insblipt5xl_insblipt5xxl_ft_xl_ag2/epoch_${epoch}/val
# export PREFIX=clevr_insblipt5xxl_vqa_val
export PREFIX=clevr_insblipt5xxl_insblipt5xl_xxl_cola2
# export PREFIX=clevr_insblipt5xl_insblipt5xxl_ft_xl2/epoch_2/val

# python evaluation/post_process.py \
# --pred ${PREDS_DIR}/${PREFIX}-da.json \
# --out ${PREDS_DIR}/${PREFIX}_post-da.json \
# --phrase "The answer is"
# # --phrase ":"
# export PREFIX=${PREFIX}_post

# # for MCQ datasets
# python evaluation/remap_predictions.py \
# --data-dir ${DATA_DIR} --dataset ${DATASET} \
# --split ${SPLIT} \
# --pred ${PREDS_DIR}/${PREFIX}-da.json \
# --out ${PREDS_DIR}/${PREFIX}-mc.json

# # for MCQ datasets
# python evaluation/prepare_predictions.py \
# --data-dir ${DATA_DIR} --dataset ${DATASET} \
# --split ${SPLIT} \
# --da ${PREDS_DIR}/${PREFIX}-da.json \
# --mc ${PREDS_DIR}/${PREFIX}-mc.json \
# --out ${PREDS_DIR}/${PREFIX}.json

# for Direct Answer datasets 
python evaluation/prepare_predictions.py \
--data-dir ${DATA_DIR} --dataset ${DATASET} \
--split ${SPLIT} \
--da ${PREDS_DIR}/${PREFIX}-da.json \
--out ${PREDS_DIR}/${PREFIX}.json

python evaluation/eval_predictions.py \
--data-dir ${DATA_DIR} --dataset ${DATASET} \
--split ${SPLIT} \
--preds ${PREDS_DIR}/${PREFIX}.json
