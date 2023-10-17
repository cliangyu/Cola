import re
import torch
import os
import random
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import time
import wandb

# import multiprocessing
import torch.multiprocessing as multiprocessing
from queue import Queue
from utils import parse_args

from data import VisualReasoningDataset, custom_collate, load_dataset
from torch.utils.data import DataLoader
from loguru import logger
import sys
import gc

from utils import (
    parse_args,
    prompt_element,
    preprocess_language,
    extract_group,
    create_prompt,
    extend_prompts,
    create_question,
)

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD:HH:mm:ss}</green> | <cyan>{name}</cyan><cyan>:line {line}</cyan>: <level>{message}</level>",
    level="INFO",
)

from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering,
    BlipForConditionalGeneration,
)
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from accelerate import infer_auto_device_map, dispatch_model

def run_blip(args):
    # wandb.init(project="aokvqa", config=args)
    dataset = VisualReasoningDataset(
        dataset_dir=args.data_dir, dataset_name=args.dataset_name, split=args.split
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate,
    )
    train_context = {}
    context = {}
    if args.context_file is not None:
        train_context = json.load(args.train_context_file)
        context = json.load(args.context_file)
        # BLIP uses different models for captioning and vqa
    # # for instructblip
    # vlm_processor = InstructBlipProcessor.from_pretrained(
    #     "Salesforce/instructblip-flan-t5-xxl"
    # )
    # vlm_model = InstructBlipForConditionalGeneration.from_pretrained(
    #     "Salesforce/instructblip-flan-t5-xxl",
    #     # device_map="auto",
    #     torch_dtype=torch.bfloat16,
    # ).to("cuda")

    if args.vlm_task == "caption":
        vlm_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        vlm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")
    elif args.vlm_task == "vqa":
        vlm_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        vlm_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")
    else:
        raise ValueError("Invalid vlm_task")

    vlm_model.eval()
    total_predictions = {}
    for group in tqdm(dataloader, desc="Generating predictions"):
        if args.vlm_task == "caption":
            questions = create_prompt(group, "a photography of")
        elif args.vlm_task == "vqa":
            questions = extract_group(group, "question")
        else:
            raise ValueError("Invalid vlm_task")
        images = extract_group(group, "image")
        inputs = vlm_processor(
            images, questions, padding=True, truncation=True, return_tensors="pt"
        ).to(
            "cuda",
            torch.bfloat16,
        )
        # inputs = vlm_processor(images, questions,  padding=True, truncation=True, return_tensors="pt")
        out = vlm_model.generate(
            **inputs,
            no_repeat_ngram_size=3,
        )
        vlm_responses = vlm_processor.batch_decode(out, skip_special_tokens=True)
        total_predictions.update(
            {group[i]["question_id"]: vlm_responses[i] for i in range(len(group))}
        )
        gc.collect()

    print(len(total_predictions))
    print(len(dataset))
    assert len(total_predictions) == len(dataset)
    with open(args.prediction_output_file, "w") as output_file:
        json.dump(total_predictions, output_file)
    logger.info("Predictions written to {0}.".format(args.prediction_output_file))


if __name__ == "__main__":
    random.seed(0)
    args = parse_args()
    run_blip(args)
