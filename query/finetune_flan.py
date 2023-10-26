"""
This code is modified from
https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_summarization_wandb.ipynb
"""
import re
import torch
import os
import random
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import time
import wandb
import torch.multiprocessing as multiprocessing
from queue import Queue
from utils import (
    parse_args,
    prompt_element,
    preprocess_language,
    extract_group,
    create_prompt,
    extend_prompts,
    create_question,
)
from transformers.optimization import Adafactor, AdafactorSchedule

from data import VisualReasoningDataset, custom_collate, load_dataset
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from model import T5OnSpecificDevices

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

device = "cuda"

# Creating the training function. This will be called in the main function. It is run depending on the epoch value.
# The model is put into train mode and then we wnumerate over the training loader and passed to the defined network


def train(
    args, epoch, tokenizer, model, device, loader, optimizer, train_set, **kwargs
):
    model.train()
    for idx, data in enumerate(loader, 0):
        prompts = create_prompt(data, "")
        for e in torch.utils.data.Subset(
            train_set, torch.randperm(len(train_set))[: args.num_examples]
        ):
            prompts = extend_prompts(
                prompts,
                prompt_element(
                    e,
                    include_choices=args.include_choices,
                    include_profile=args.include_profile,
                    include_caption=args.include_caption,
                    include_rationale=args.include_rationale,
                    include_answer=True,
                    vlm2_captions=kwargs["vlm2_captions_train"],
                    vlm2_answers=kwargs["vlm2_answers_train"],
                    vlm1_captions=kwargs["vlm1_captions_train"],
                    vlm1_answers=kwargs["vlm1_answers_train"],
                ),
            )
            prompts = extend_prompts(prompts, "\n\n")

        current_questions = create_question(
            data,
            include_choices=args.include_choices,
            include_profile=args.include_profile,
            include_caption=args.include_caption,
            include_rationale=False,
            include_answer=False,
            vlm2_captions=kwargs["vlm2_captions_train"],
            vlm2_answers=kwargs["vlm2_answers_train"],
            vlm1_captions=kwargs["vlm1_captions_train"],
            vlm1_answers=kwargs["vlm1_answers_train"],
        )
        prompts = extend_prompts(prompts, current_questions)
        input_ids, attention_mask = preprocess_language(
            tokenizer,
            prompts,
            device=device,
        )
        if args.include_rationale:
            targets = create_prompt(data, "Rationale:")

            # # ground truth rationale
            # target_rationales = [random.choice(k) for k in extract_group(data, "rationales")]
            # the longest rationale
            # target_rationales = [max(k, key=len) for k in extract_group(data, "rationales")]
            target_rationales = [" ".join(k) for k in extract_group(data, "rationales")]
            targets = extend_prompts(targets, target_rationales)

            # # self-rationalization
            # qids = extract_group(data, "question_id")
            # group_rationalization = [rationalization[qid] for qid in qids]
            # # add dot if not present
            # group_rationalization = [r if r[-1] == '.' else r + '.' for r in group_rationalization]
            # targets = extend_prompts(targets, group_rationalization)

            separator = "<extra_id_82>"
            if args.answer_factor is not None:
                targets = extend_prompts(
                    targets, separator
                )  # the token for the rationale separator
            answers = extract_group(data, "mc_answer")
            targets = extend_prompts(targets, create_prompt(answers, " The answer is "))
            targets = extend_prompts(targets, answers)
            targets = extend_prompts(targets, ".")
        else:
            if args.dataset_name in ["okvqa", "vqav2", "gqa", "clevr"]:
                targets = [item["direct_answers"][0] for item in data]
            else:
                targets = extract_group(data, "mc_answer")
        gt_labels, gt_attention_mask = preprocess_language(
            tokenizer,
            targets,
            device="cuda",
        )
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=gt_labels
        )
        loss = outputs.loss

        if args.answer_factor is not None:
            separator_index = torch.where(
                gt_labels == tokenizer.convert_tokens_to_ids(separator)
            )[1]
            # rationale_logits = lm_logits.clone()
            # rationale_gt_labels = gt_labels.clone()
            # # mask out rationale_gt_labels after the separator_index
            # for sample in range(args.bs):
            #     rationale_gt_labels[sample, separator_index[sample]:] = -100
            # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            # rationale_loss = loss_fct(rationale_logits.view(-1, rationale_logits.size(-1)), rationale_gt_labels.view(-1))
            # mask out logits before the separator token
            answer_logits = outputs.logits.clone()
            answer_gt_labels = gt_labels.clone()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            for sample in range(args.bs):
                answer_gt_labels[sample, : separator_index[sample]] = -100
            answer_loss = loss_fct(
                answer_logits.view(-1, answer_logits.size(-1)),
                answer_gt_labels.view(-1),
            )
            # loss = rationale_loss + answer_loss * args.answer_factor
            loss = loss + answer_loss * args.answer_factor

        if idx % 10 == 0:
            wandb.log({"Training Loss": loss.item()})

        if idx % 500 == 0:
            logger.info(f"Epoch: {epoch}, Loss:  {loss.item()}")

        # scale the loss to the mean of the accumulated batch size
        loss = loss / args.ACCUM_ITER

        # backward pass
        loss.backward()

        # weights update
        if ((idx + 1) % args.ACCUM_ITER == 0) or (idx + 1 == len(loader)):
            optimizer.step()
            optimizer.zero_grad()


def validate(
    args,
    epoch,
    tokenizer,
    model,
    device,
    val_loader,
    optimizer,
    train_set,
    val_set,
    **kwargs,
):
    model.eval()
    total_predictions = {}
    with torch.no_grad():
        counter = 0
        for idx, group in enumerate(val_loader, 0):
            prompts = create_prompt(group, args.prompt_prefix)

            if args.incontext:
                for e in torch.utils.data.Subset(
                    train_set, torch.randperm(len(train_set))[: args.num_examples]
                ):
                    prompts = extend_prompts(
                        prompts,
                        prompt_element(
                            e,
                            include_choices=args.include_choices,
                            include_answer=True,
                            include_profile=args.include_profile,
                            include_rationale=args.include_rationale,
                            include_caption=args.include_caption,
                            vlm1_answers=kwargs["vlm1_answers_train"],
                            vlm2_answers=kwargs["vlm2_answers_train"],
                            vlm2_captions=kwargs["vlm2_captions_train"],
                            vlm1_captions=kwargs["vlm1_captions_train"],
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
                    include_profile=args.include_profile,
                    vlm1_answers=kwargs["vlm1_answers_val"],
                    vlm2_answers=kwargs["vlm2_answers_val"],
                    include_caption=args.include_caption,
                    include_rationale=False,
                    vlm2_captions=kwargs["vlm2_captions_val"],
                    vlm1_captions=kwargs["vlm1_captions_val"],
                    cot=args.cot,
                    # rationalization=rationalization_train, # placeholder
                ),
            )
            if args.cot:
                prompts = extend_prompts(prompts, "\n\n")
                cot = "Answer in the following format: \n[YOUR RATIONALE]. The answer is [YOUR CHOICE]."
                prompts = extend_prompts(prompts, cot)
            input_ids, attention_mask = preprocess_language(
                tokenizer, prompts, device=device
            )
            flan_start_time = time.time()
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                early_stopping=False,
                no_repeat_ngram_size=3,
            )
            flan_end_time = time.time()
            flan_1st_responses = tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            logger.info(
                f"Batch: {counter}\tFLAN_1 time: {flan_end_time - flan_start_time}"
            )
            # Add data to queue
            for i in range(len(group)):
                group[i]["flan_1st_response"] = flan_1st_responses[i]
                group[i]["prompt"] = prompts[i]
                total_predictions[group[i]["question_id"]] = flan_1st_responses[i]
            counter += 1
        assert len(total_predictions) == len(val_set)
        prediction_file_path = os.path.join(
            args.prediction_output_dir, os.environ["WANDB_RUN_ID"], f"epoch_{epoch}"
        )
        # make dir for prediction_file
        Path(prediction_file_path).mkdir(parents=True, exist_ok=True)
        prediction_file = os.path.join(
            prediction_file_path, f"{args.val_split}-da.json"
        )
        with open(prediction_file, "w") as output_file:
            json.dump(total_predictions, output_file)
        logger.info("Predictions written to {0} .".format(prediction_file))


def main():
    args = parse_args()

    # args.bs = 16    # input batch size for training (default: 64)
    # args.VALID_BATCH_SIZE = args.bs * 4  # input batch size for testing (default: 96)
    args.VALID_BATCH_SIZE = args.bs * 8  # input batch size for testing (default: 96)
    args.TRAIN_EPOCHS = 30  # number of epochs to train
    args.VAL_EPOCHS = 5 if args.dataset_name == "vsr" else 1
    args.LEARNING_RATE = 1e-4  # learning rate (default: 0.01)
    args.SEED = 42  # random seed (default: 42)
    args.MAX_LEN = 512
    args.SUMMARY_LEN = 150
    args.num_examples = 0
    args.peft = False
    args.answer_factor = None
    # args.answer_factor = 5
    args.ACCUM_ITER = 4

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(args.SEED)  # pytorch random seed
    np.random.seed(args.SEED)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    wandb.init(config=args, project="fine_tune_flan", entity="liangyu")

    # tokenizer for encoding the text
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

    # Creating the Training and Validation dataset
    train_set = VisualReasoningDataset(
        dataset_dir=args.data_dir,
        dataset_name=args.dataset_name,
        split=args.split,
        include_image=False,
    )
    if args.k_shot_finetuning > 0:
        train_set = torch.utils.data.Subset(
            train_set, torch.randperm(len(train_set))[: args.k_shot_finetuning]
        )
    val_set = VisualReasoningDataset(
        dataset_dir=args.data_dir,
        dataset_name=args.dataset_name,
        split=args.val_split,
        include_image=False,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": args.bs,
        "shuffle": True,
        "num_workers": 16,
        "pin_memory": True,
        "drop_last": False,
        "collate_fn": custom_collate,
    }

    val_params = {
        "batch_size": args.VALID_BATCH_SIZE,
        "shuffle": False,
        "num_workers": 16,
        "pin_memory": True,
        "drop_last": False,
        "collate_fn": custom_collate,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(train_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)
    logger.info("Training and Validation Dataloaders are created.")

    model = T5ForConditionalGeneration.from_pretrained(
        args.llm,
        #    torch_dtype=torch.bfloat16
    )

    if args.peft:
        from transformers.adapters import IA3Config

        adapter_name = "ia3_adapter"
        adapter_config = IA3Config()
        model.add_adapter(adapter_name, config=adapter_config)
        model.parallelize()
        model.train_adapter(adapter_name)
        model.set_active_adapters(adapter_name)
    else:
        model.parallelize()

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = Adafactor(
        model.parameters(),
        relative_step=False,
        warmup_init=False,
        lr=args.LEARNING_RATE,
    )

    logger.info("Model is created.")

    # Log metrics with wandb
    wandb.watch(model, log="all")
    # Training loop
    logger.info("Initiating Fine-Tuning for the model on our dataset")

    with open(
        f"{args.prediction_output_dir}/{args.dataset_name}_{args.vlm1}_vqa_{args.split}-da.json",
        "r",
    ) as profile:  # hard code
        vlm1_answers_train = json.load(profile)
    with open(
        f"{args.prediction_output_dir}/{args.dataset_name}_{args.vlm2}_vqa_{args.split}-da.json",
        "r",
    ) as profile:  # hard code
        vlm2_answers_train = json.load(profile)
    with open(
        f"{args.prediction_output_dir}/{args.dataset_name}_{args.vlm1}_caption_{args.split}-da.json",
        "r",
    ) as profile:  # hard code
        vlm1_captions_train = json.load(profile)
    with open(
        f"{args.prediction_output_dir}/{args.dataset_name}_{args.vlm2}_caption_{args.split}-da.json",
        "r",
    ) as profile:  # hard code
        vlm2_captions_train = json.load(profile)
    with open(
        f"{args.prediction_output_dir}/{args.dataset_name}_{args.vlm1}_vqa_{args.val_split}-da.json",
        "r",
    ) as profile:  # hard code
        vlm1_answers_val = json.load(profile)
    with open(
        f"{args.prediction_output_dir}/{args.dataset_name}_{args.vlm2}_vqa_{args.val_split}-da.json",
        "r",
    ) as profile:  # hard code
        vlm2_answers_val = json.load(profile)
    with open(
        f"{args.prediction_output_dir}/{args.dataset_name}_{args.vlm1}_caption_{args.val_split}-da.json",
        "r",
    ) as profile:  # hard code
        vlm1_captions_val = json.load(profile)
    with open(
        f"{args.prediction_output_dir}/{args.dataset_name}_{args.vlm2}_caption_{args.val_split}-da.json",
        "r",
    ) as profile:  # hard code
        vlm2_captions_val = json.load(profile)

    for epoch in range(args.TRAIN_EPOCHS):
        train(
            args,
            epoch,
            tokenizer,
            model,
            device,
            training_loader,
            optimizer,
            train_set,
            vlm2_captions_train=vlm2_captions_train,
            vlm1_captions_train=vlm1_captions_train,
            vlm2_answers_train=vlm2_answers_train,
            vlm1_answers_train=vlm1_answers_train,
        )
        if args.peft:
            model.save_adapter(
                os.path.join(
                    args.model_output_dir,
                    os.environ["WANDB_RUN_ID"],
                    "{}_{}_bs{}_epoch{}_{}".format(
                        args.llm,
                        "language_profile" if args.include_profile else "wo_profile",
                        args.bs,
                        epoch,
                        adapter_name,
                    ),
                ),
                adapter_name,
            )
        else:
            model.save_pretrained(
                os.path.join(
                    args.model_output_dir,
                    os.environ["WANDB_RUN_ID"],
                    "{}_{}_bs{}_epoch{}".format(
                        args.llm,
                        "language_profile" if args.include_profile else "wo_profile",
                        args.bs,
                        epoch,
                    ),
                )
            )

        if epoch % args.VAL_EPOCHS == 0:
            validate(
                args,
                epoch,
                tokenizer,
                model,
                device,
                val_loader,
                optimizer,
                train_set,
                val_set,
                vlm2_captions_train=vlm2_captions_train,
                vlm1_captions_train=vlm1_captions_train,
                vlm2_answers_train=vlm2_answers_train,
                vlm1_answers_train=vlm1_answers_train,
                vlm2_captions_val=vlm2_captions_val,
                vlm1_captions_val=vlm1_captions_val,
                vlm2_answers_val=vlm2_answers_val,
                vlm1_answers_val=vlm1_answers_val,
            )


if __name__ == "__main__":
    main()
