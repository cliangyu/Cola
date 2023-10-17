<!-- # 🥤 Cola: Language Models are Visual Reasoning Coordinators -->

<div align="center">

<h2>🥤 Cola [NeurIPS 2023] </br> Large Language Models are Visual Reasoning Coordinators</h2>

<div align="center">
    <a href='https://cliangyu.com/' target='_blank'>Liangyu Chen<sup>*,†,♥</sup></a>&emsp;
    <a href='https://brianboli.com/' target='_blank'>Bo Li<sup>*,♥</sup></a>&emsp;
    <a href='https://sincerass.github.io/' target='_blank'>Sheng Shen<sup>♣</sup></a>&emsp;
    <a href='https://jingkang50.github.io/' target='_blank'>Jingkang Yang<sup>♥</sup></a>&emsp;
    </br>
    <a href='https://chunyuan.li/' target='_blank'>Chunyuan Li<sup>♠</sup></a>&emsp;
    <a href='https://people.eecs.berkeley.edu/~keutzer/' target='_blank'>Kurt Keutzer<sup>♣</sup></a>&emsp;
    <a href='http://people.eecs.berkeley.edu/~trevor/' target='_blank'>Trevor Darrell<sup>♣</sup></a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu<sup>&#x2709,♥</sup></a>
</div>

<div align="center">
    <sup>♥</sup>S-Lab, Nanyang Technological University
    </br>
    <sup>♣</sup>University of California, Berkeley&emsp;
    <sup>♠</sup>Microsoft Research, Redmond
    </br>
    <sup>*</sup>Equal Contribution&emsp;
    <sup>†</sup>Project Lead&emsp;
    <sup>&#x2709</sup>Corresponding Author
    
</div>

---

<img src="https://i.postimg.cc/ZqXSn8rN/sm-teaser.png">

<h3>TL;DR</h3>
    
We use a language model (LM) to aggregate the outputs of 2+ vision-language models (VLMs). Our model assemble approach is named **Cola** (COordinative LAnguage model or visual reasoning). Cola is most effective with the LM finetuned (termed as Cola-FT). Cola is also effective with zero-shot or few-shot in-context learning (termed as Cola-Zero). Beside the performance gain, Cola is also more robust to the VLMs' errors. We show that Cola can be applied to various VLMs (including large multimodal models like InstructBLIP) and 7 datasets (VQA v2, OK-VQA, A-OKVQA, e-SNLI-VE, VSR, CLEVR, GQA), and it consistently improves the performance. 

---

<p align="center">
  <a href="https://openreview.net/forum?id=kdHpWogtX6Y" target='_blank'>[arXiv]</a> •
  <a href="https://www.youtube.com/watch?v=Tp5SN8AqYZE&ab_channel=AIJournalClub" target='_blank'>[talk (中文)]</a> • English talk coming soon
</p>

</div>

## 🍱 Environment Setup
I highly recommend you to update NVIDIA drivers and CUDA to the latest version in case of weird bugs. See `requirements.txt` for the environment where the code is tested with.

```shell
conda env create -f cola.yml
```
We use bf16 for inference and finetuning, which supports newer GPUs.

## 🥙 Prepare Datasets and Models
```shell
mkdir datasets
mkdir predictions
mkdir pretrained_models
```

Below are the datasets we tested, you don't have to download all. I suggest starting with A-OKVQA.

* A-OKVQA: download from [official page](https://allenai.org/project/a-okvqa/home)
* OK-VQA: download from [official page](https://okvqa.allenai.org/download.html)
* VQAv2: download from [official page](https://visualqa.org/download.html)
* CLEVR: download from [Kaggle](https://www.kaggle.com/datasets/timoboz/clevr-dataset)
* COCO & GQA: download by [LAVIS script](https://github.com/salesforce/LAVIS/tree/main/lavis/datasets/download_scripts)
* VSR: download from [official page](https://github.com/cambridgeltl/visual-spatial-reasoning)
* e-SNLI-VE & Flickr30k: download from [official page](https://github.com/maximek3/e-ViL)

We convert all the datasets to the format of A-OKVQA dataset. See `./data_utils` for conversion scripts.

```shell
datasets
├── aokvqa
├── okvqa
├── coco
│   ├── train2017
│   ├── val2017
│   └── test2017
├── esnlive
│   └── flicr30k_images
├── vsr
│   └── trainval2017
├── vqav2
├── gqa
│   └── images
└── clevr
```

Download OFA model from Huggingface. Other models can be downloaded automatically.
```shell
cd ..
git lfs clone https://huggingface.co/OFA-Sys/ofa-large
```

## 🚀 Inference

```shell
# 1. Get the plausible answers for the validation set

python query/query_blip.py --data-dir ./datasets/ --dataset-name aokvqa --split val --vlm-task vqa --bs 128 --prediction-out ./predictions/aokvqa_blip_vqa_val-da.json

python query/query_ofa.py --vlm-model-path ../OFA-large --data-dir ./datasets/ --dataset-name aokvqa --split val --vlm-task vqa --bs 128 --prediction-out ./predictions/aokvqa_ofa_vqa_val-da.json

# 2. Get the captions for the validation set

python query/query_blip.py --data-dir ./datasets/ --dataset-name aokvqa --split val --vlm-task caption --bs 128 --prediction-out ./predictions/aokvqa_blip_caption_val-da.json

python query/query_ofa.py --vlm-model-path ../OFA-large --data-dir ./datasets/ --dataset-name aokvqa --split val --vlm-task caption --bs 128 --prediction-out ./predictions/aokvqa_ofa_caption_val-da.json


# 3. Query the language model, Cola-Zero. Delete "--incontext --num-examples 2" for 0-shot inference.

python query/query_flan.py --data-dir ./datasets/ --dataset-name aokvqa --split val --vlm-task vqa --bs 128 --prediction-out ./predictions/aokvqa_cola2-da.json --max-new-tokens 250 --llm google/flan-t5-small --vlm1 ofa --vlm2 blip --include-profile --include-caption --include-choices --incontext --num-examples 2

## Anoterh example: query Mistral-7B, with InstructBLIP-XL and XXL
python query/query_llm.py --data-dir ./datasets/ --dataset-name aokvqa --split val --vlm-task vqa --bs 128 --prediction-out ./predictions/aokvqa_mistral_cola2-da.json --max-new-tokens 250 --llm mistralai/Mistral-7B-v0.1 --vlm1 insblipt5xl --vlm2 insblipt5xxl --include-profile --include-caption --include-choices --incontext --num-examples 2


# 4. Evaluate the predictions (multiple choice), see "evaluate.sh" for direct answer evalation.
export PYTHONPATH=.
export DATA_DIR=./datasets/
export DATASET=aokvqa
export SPLIT=val
export LOG_DIR=./logs/
export PREDS_DIR=./predictions
export PT_MODEL_DIR=./pretrained_models/
export PREFIX=aokvqa_cola0

python evaluation/prepare_predictions.py \
--data-dir ${DATA_DIR} --dataset ${DATASET} \
--split ${SPLIT} \
--da ${PREDS_DIR}/${PREFIX}-da.json \
--mc ${PREDS_DIR}/${PREFIX}-mc.json \
--out ${PREDS_DIR}/${PREFIX}.json

python evaluation/eval_predictions.py \
--data-dir ${DATA_DIR} --dataset ${DATASET} \
--split ${SPLIT} \
--preds ${PREDS_DIR}/${PREFIX}.json
```

## 🎛️ Finetuning

```shell
# Get the plausible answers and captions for both training and validation sets (see Step 1 and 2 of Inference)

# 1. Finetune the language model, Cola-FT. Delete "--include-choices" for direct answer datasets. Need to "wandb login" before finetuning.
export MODEL_NAME=aok_blip_ofa_ft
WANDB_RUN_ID=${MODEL_NAME} python query/finetune_flan.py \
--data-dir ./datasets/ --dataset-name aokvqa --split train --val-split val \
--bs 16 --llm google/flan-t5-xxl --vlm1 blip --vlm2 ofa \
--prediction-out placeholder --include-profile --include-caption --include-choices
# Another example: finetune Mistral-7B or other decoder-only models
export MODEL_NAME=aok_insblipt5xl_insblipt5xxl_mistral_ft
WANDB_RUN_ID=${MODEL_NAME} python query/finetune_llm.py --data-dir ./datasets/ --dataset-name aokvqa --split train --val-split val --bs 16 --llm mistralai/Mistral-7B-v0.1 --vlm1 insblipt5xl --vlm2 insblipt5xxl --prediction-out placeholder --include-profile --include-caption --include-choices

# 2. Query the finetuned model. We don't suggest using few-shot in-context learning for finetuned models.

python query/query_flan.py --data-dir ./datasets/ --dataset-name aokvqa --split val --vlm-task vqa --bs 128 --max-new-tokens 250 --prediction-out ./predictions/aokvqa_colaft-da.json --max-new-tokens 250 --llm pretrained_models/${MODEL_NAME}/{the_epoch_you_test} --vlm1 blip --vlm2 ofa --include-choices --include-profile --include-caption

# 3. Evaluate. The evaluation script is the same as Step 4 of Inference.
```

## 📚 Citation

If you use this code in your research, please kindly cite this work.

```bibtex
@inproceedings{
chen2023language,
title={Language Models are Visual Reasoning Coordinators},
author={Liangyu Chen and Bo Li and Sheng Shen and Jingkang Yang and Chunyuan Li and Kurt Keutzer and Trevor Darrell and Ziwei Liu},
booktitle={ICLR 2023 Workshop on Mathematical and Empirical Understanding of Foundation Models},
year={2023},
url={https://openreview.net/forum?id=kdHpWogtX6Y}
}
```

## 🙏 Acknowledgements

Evaluation code is borrowed from [aokvqa](https://github.com/allenai/aokvqa). Part of this README is borrowed from [visual_prompt_retrieval](https://github.com/ZhangYuanhan-AI/visual_prompt_retrieval).