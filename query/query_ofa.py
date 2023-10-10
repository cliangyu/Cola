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
import torch.multiprocessing as multiprocessing
from queue import Queue
from utils import parse_args

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))  # to load OFA
from ofa import OFATokenizer, OFAModel

from data import VisualReasoningDataset, custom_collate, load_dataset
from torch.utils.data import DataLoader
from loguru import logger
import gc

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD:HH:mm:ss}</green> | <cyan>{name}</cyan><cyan>:line {line}</cyan>: <level>{message}</level>",
    level="INFO",
)


def preprocess_language(flan_tokenizer, text, device="cuda:0"):
    inputs_dict = flan_tokenizer.batch_encode_plus(
        text, padding=True, truncation=True, return_tensors="pt"
    )
    input_ids = inputs_dict.input_ids.to(
        device
    )  # must be the first index of device map
    attention_mask = inputs_dict.attention_mask.to(device)
    return input_ids, attention_mask


def extract_group(group, key):
    return [item[key] for item in group]


def create_prompt(group, prompt_prefix):
    return [prompt_prefix for i in range(len(group))]


def extend_prompts(prompts, texts):
    if type(texts) is str:
        texts = [texts for i in range(len(prompts))]
    return [prompts[i] + texts[i] for i in range(len(prompts))]


def preprocess_vlm_text(texts):
    """Preprocess text for vlm model.
    Lower case, add space before marks.
    """
    return [re.sub("([.,!?()])", r" \1 ", text.lower()) for text in texts]


def run_ofa(args):
    # wandb.init(project="aokvqa", config=args)
    dataset = VisualReasoningDataset(
        dataset_dir=args.data_dir,
        dataset_name=args.dataset_name,
        split=args.split,
        transform="ofa",
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

    ofa_dir = args.ofa_model_path if args.ofa_model_path else "./OFA-large"
    ofa_device = "cuda"
    ofa_tokenizer = OFATokenizer.from_pretrained(ofa_dir)
    ofa_model = OFAModel.from_pretrained(
        ofa_dir,
        use_cache=True,
    )
    logger.info("OFA model loaded")
    ofa_model.to(ofa_device)
    ofa_model.eval()

    total_predictions = {}
    counter = 0
    for group in tqdm(dataloader, desc="Generating predictions"):
        if args.vlm_task == "caption":
            questions = create_prompt(group, "What does the image describe?")
        elif args.vlm_task == "vqa":
            questions = extract_group(group, "question")
        input_ids, attention_mask = preprocess_language(
            ofa_tokenizer,
            questions,
            device=ofa_device,
        )
        patch_img = [torch.unsqueeze(x, dim=0) for x in extract_group(group, "image")]
        patch_img = torch.vstack(patch_img).to(ofa_device)
        data = {}
        data["net_input"] = {
            "input_ids": input_ids,
            "patch_images": patch_img,
            "patch_masks": torch.tensor([True]),
        }
        vlm_start_time = time.time()
        gen = ofa_model.generate(
            input_ids,
            attention_mask=attention_mask,
            patch_images=patch_img,
            num_beams=5,
            no_repeat_ngram_size=3,
        )
        vlm_responses = ofa_tokenizer.batch_decode(gen, skip_special_tokens=True)
        vlm_end_time = time.time()
        counter = counter + 1
        total_predictions.update(
            {group[i]["question_id"]: vlm_responses[i] for i in range(len(group))}
        )
        gc.collect()

    assert len(total_predictions) == len(dataset)
    print(len(total_predictions))
    print(len(dataset))
    with open(args.prediction_output_file, "w") as output_file:
        json.dump(total_predictions, output_file)
    logger.info("Predictions written to {0} .".format(args.prediction_output_file))


if __name__ == "__main__":
    random.seed(0)
    args = parse_args()
    run_ofa(args)
