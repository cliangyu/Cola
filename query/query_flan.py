import torch
import random
import json
import time
from utils import (
    parse_args,
    prompt_element,
    preprocess_language,
    create_prompt,
    extend_prompts,
    create_question,
)

from data import VisualReasoningDataset, custom_collate, load_dataset
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from loguru import logger
import gc
import sys

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD:HH:mm:ss}</green> | <cyan>{name}</cyan><cyan>:line {line}</cyan>: <level>{message}</level>",
    level="INFO",
)


def run_flan(args):
    # wandb.init(project="aokvqa", config=args)
    train_set = VisualReasoningDataset(
        dataset_dir=args.data_dir,
        dataset_name=args.dataset_name,
        split="train",
        include_image=False,
    )
    dataset = VisualReasoningDataset(
        dataset_dir=args.data_dir,
        dataset_name=args.dataset_name,
        split=args.split,
        include_image=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate,
    )
    train_context = {}
    context = {}
    if args.context_file is not None:
        with open(args.context_file, "r") as f:
            context = json.load(args.context_file)
        with open(args.train_context_file, "r") as f:
            train_context = json.load(args.train_context_file)

    # Initiate FLAN model and tokenizer
    flan_device = "cuda:{}".format(str(0))
    flan_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    flan_model = T5ForConditionalGeneration.from_pretrained(args.llm)
    if args.adapter_name is not None:
        flan_model.load_adapter(args.adapter_name, set_active=True)
    flan_model.parallelize()  # Use this line instead of device_map="auto" for T5 models, very weird bug...

    with open(
        f"{args.prediction_output_dir}/{args.dataset_name}_{args.vlm1}_vqa_train-da.json",
        "r",
    ) as profile:  # hard code
        vlm1_answers_train = json.load(profile)
    with open(
        f"{args.prediction_output_dir}/{args.dataset_name}_{args.vlm2}_vqa_train-da.json",
        "r",
    ) as profile:  # hard code
        vlm2_answers_train = json.load(profile)
    with open(
        f"{args.prediction_output_dir}/{args.dataset_name}_{args.vlm1}_caption_train-da.json",
        "r",
    ) as profile:  # hard code
        vlm1_captions_train = json.load(profile)
    with open(
        f"{args.prediction_output_dir}/{args.dataset_name}_{args.vlm2}_caption_train-da.json",
        "r",
    ) as profile:  # hard code
        vlm2_captions_train = json.load(profile)
    with open(
        f"{args.prediction_output_dir}/{args.dataset_name}_{args.vlm1}_vqa_{args.split}-da.json",
        "r",
    ) as profile:  # hard code
        vlm1_answers_val = json.load(profile)
    with open(
        f"{args.prediction_output_dir}/{args.dataset_name}_{args.vlm2}_vqa_{args.split}-da.json",
        "r",
    ) as profile:  # hard code
        vlm2_answers_val = json.load(profile)
    with open(
        f"{args.prediction_output_dir}/{args.dataset_name}_{args.vlm1}_caption_{args.split}-da.json",
        "r",
    ) as profile:  # hard code
        vlm1_captions_val = json.load(profile)
    with open(
        f"{args.prediction_output_dir}/{args.dataset_name}_{args.vlm2}_caption_{args.split}-da.json",
        "r",
    ) as profile:  # hard code
        vlm2_captions_val = json.load(profile)
    # with open('predictions/fx_rationalization_train-da.json', 'r') as profile: # hard code
    #     rationalization_train = json.load(profile)
    total_predictions = {}
    counter = 0

    for group in dataloader:
        prompts = create_prompt(group, args.prompt_prefix)
        # prompts = extend_prompts(prompts, "Answer the following multiple choice question by reasoning step by step. vlm2 and vlm1 are two different vision-language models to help you answer this visual question. First, state your rationale based on vlm2 and vlm1's description and their answers to the visual question. Then give the final answer.\n")

        if args.incontext:
            for idx, e in enumerate(
                torch.utils.data.Subset(
                    train_set, torch.randperm(len(train_set))[: args.num_examples]
                )
            ):
                prompts = extend_prompts(
                    prompts,
                    prompt_element(
                        e,
                        include_choices=args.include_choices,
                        include_answer=True,
                        include_question=(idx == 0),
                        include_profile=args.include_profile,
                        include_rationale=args.include_rationale,
                        include_caption=args.include_caption,
                        vlm1_answers=vlm1_answers_train,
                        vlm2_answers=vlm2_answers_train,
                        vlm2_captions=vlm2_captions_train,
                        vlm1_captions=vlm1_captions_train,
                        cot=False,
                        # rationalization=rationalization_train,
                    ),
                )
                prompts = extend_prompts(prompts, "\n\n")
            prompts = extend_prompts(prompts, "\n\n")

        prompts = extend_prompts(
            prompts,
            create_question(
                group,
                include_choices=args.include_choices,
                include_answer=False,
                include_question=(not args.incontext),
                include_profile=args.include_profile,
                vlm1_answers=vlm1_answers_val,
                vlm2_answers=vlm2_answers_val,
                include_caption=args.include_caption,
                include_rationale=False,
                vlm2_captions=vlm2_captions_val,
                vlm1_captions=vlm1_captions_val,
                cot=args.cot,
                # rationalization=rationalization_train, # placeholder
            ),
        )
        if args.cot:
            prompts = extend_prompts(prompts, "\n\n")
            cot = "Answer in the following format: \n[YOUR RATIONALE]. The answer is [YOUR CHOICE]."
            prompts = extend_prompts(prompts, cot)
        input_ids, attention_mask = preprocess_language(
            flan_tokenizer, prompts, device=flan_device
        )
        flan_start_time = time.time()
        if args.num_return_sequences == 1:  # beam search (greedy decoding)
            outputs = flan_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                early_stopping=False,
                no_repeat_ngram_size=3,
            )
        if args.num_return_sequences > 1:  # diverse sampling
            outputs = flan_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                early_stopping=False,
                no_repeat_ngram_size=3,
                num_return_sequences=args.num_return_sequences,
                # num_beams=args.num_return_sequences, # should be the same as num_return_sequences or larger
                do_sample=True,
                top_k=50,
                top_p=0.95,
            )
        flan_end_time = time.time()
        flan_1st_responses_ungrouped = flan_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        logger.info(f"Batch: {counter}\tFLAN_1 time: {flan_end_time - flan_start_time}")

        # group every args.num_return_sequences together
        if args.num_return_sequences == 1:
            flan_1st_responses = flan_1st_responses_ungrouped
        else:  # return multiple responses as a list
            flan_1st_responses = []
            for i in range(
                len(flan_1st_responses_ungrouped) // args.num_return_sequences
            ):
                flan_1st_responses.append(
                    flan_1st_responses_ungrouped[
                        i
                        * args.num_return_sequences : (i + 1)
                        * args.num_return_sequences
                    ]
                )

        # Add data to queue
        for i in range(len(group)):
            group[i]["flan_1st_response"] = flan_1st_responses[i]
            group[i]["prompt"] = prompts[i]
            total_predictions[group[i]["question_id"]] = flan_1st_responses[i]
        counter += 1

    assert len(total_predictions) == len(dataloader.dataset)
    with open(args.prediction_output_file, "w") as output_file:
        json.dump(total_predictions, output_file)
    logger.info("Predictions written to {0} .".format(args.prediction_output_file))


if __name__ == "__main__":
    random.seed(0)
    args = parse_args()
    run_flan(args)
