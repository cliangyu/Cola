export CUDA_VISIBLE_DEVICES=2,3
python query/query_flan.py --data-dir ./datasets/ --dataset-name clevr --split val --vlm-task vqa --bs 128 --prediction-out ./predictions/clevr_insblipt5xxl_insblipt5xl_xxl_cola0-da.json --max-length 250 --flan google/flan-t5-xxl --vlm1 insblipt5xl --vlm2 insblipt5xxl --include-profile --include-caption 
# --incontext --num-examples 2