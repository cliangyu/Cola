import torch
import os
import json
from pathlib import Path
import numpy as np
import time
import wandb
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
import sys

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD:HH:mm:ss}</green> | <cyan>{name}</cyan><cyan>:line {line}</cyan>: <level>{message}</level>",
    level="INFO",
)

device = "cuda"

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
        
        if args.include_rationale:
            targets = create_prompt(data, "Rationale:")
            target_rationales = [" ".join(k) for k in extract_group(data, "rationales")]
            targets = extend_prompts(targets, target_rationales)

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
                
        question_and_answer = extend_prompts(prompts, targets)   
        input_ids, attention_mask = preprocess_language(
            tokenizer,
            question_and_answer,
            device=device,
        )
        
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=masking(input_ids)
        )
        loss = outputs.loss

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
        
def masking(input_ids, masking_number: int = -100, tokens: list = [29909, 29901]):
    """
    This function is designed to mask input_ids after a specific pattern, denoted by the tokens list.
    The pattern could be of any length.
    Currently hardcoded for models like mistral to find specific patterns that break the question/answer sequence.
    """
    # Initialize labels with masking_number
    labels = masking_number * torch.ones(input_ids.shape, dtype=torch.int64).to('cuda', non_blocking=True)
    
    token_length = len(tokens)  # Get the length of the tokens to match
    
    for i in range(input_ids.shape[0]):
        # Find all occurrences of the pattern in tokens
        occurrences = 0
        for j in range(input_ids.shape[1] - token_length + 1):
            if all(input_ids[i][j + k] == tokens[k] for k in range(token_length)):
                occurrences += 1

        # # Assert that the pattern exists only once
        # assert occurrences == 1, f"Pattern found {occurrences} times in sample {i}"

        # Find the breakpoint where we have the consecutive tokens
        for j in range(input_ids.shape[1] - token_length + 1):
            if all(input_ids[i][j + k] == tokens[k] for k in range(token_length)):
                breakpoint = j + token_length
                labels[i][breakpoint:] = input_ids[i][breakpoint:]
                break
                
    return labels



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
            llm_start_time = time.time()
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                early_stopping=False,
                no_repeat_ngram_size=3,
            )
            llm_end_time = time.time()
            llm_1st_responses = tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            logger.info(
                f"Batch: {counter}\tllm_1 time: {llm_end_time - llm_start_time}"
            )
            # Add data to queue
            for i in range(len(group)):
                group[i]["llm_1st_response"] = llm_1st_responses[i]
                group[i]["prompt"] = prompts[i]
                total_predictions[group[i]["question_id"]] = llm_1st_responses[i]
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
    wandb.init(config=args, project="fine_tune_llm", entity="liangyu")

    # tokenizer for encoding the text
    tokenizer = AutoTokenizer.from_pretrained(args.llm, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    model = AutoModelForCausalLM.from_pretrained(
        args.llm,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )

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
