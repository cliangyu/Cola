export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_RUN_ID=clevr_insblipt5xl_insblipt5xxl_ft_xl2
python query/finetune_flan.py --data-dir ./datasets/ --dataset-name clevr --split train --val-split val --bs 64 --flan pretrained_models/aok_insblipt5xl_insblipt5xxl_ft_xl/google/flan-t5-xl_language_profile_bs32_epoch1 --vlm1 insblipt5xl --vlm2 insblipt5xxl --prediction-out placeholder --include-profile --include-caption 
# --include-choices 