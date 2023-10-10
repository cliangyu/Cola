import re
import os
import pathlib
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=["aokvqa", "vcr", "esnlive", "vsr", "okvqa", "vqav2", "gqa", "clevr"],
        required=True,
    )
    parser.add_argument(
        "--split", type=str, choices=["train", "val", "test"], required=True
    )
    parser.add_argument("--val-split", type=str, choices=["train", "val", "test"])
    parser.add_argument("--vlm-task", type=str, choices=["caption", "vqa"])
    parser.add_argument("--ofa-model-path", type=str, default=None)
    parser.add_argument("--blip-model-path", type=str, default=None)
    parser.add_argument("--num-examples", type=int, default=0)
    parser.add_argument(
        "--k-shot-finetuning",
        type=int,
        default=-1,
        help="k-shot finetuning, -1 for no k-shot finetuning",
    )
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument(
        "--incontext", action="store_true", help="incontext learning, give examples"
    )
    parser.add_argument("--train-context", type=str, dest="train_context_file")
    parser.add_argument("--context", type=str, dest="context_file")
    parser.add_argument("--prefix", type=str, default="", dest="prompt_prefix")
    parser.add_argument("--flan", type=str, default="google/flan-t5-xxl")
    parser.add_argument("--vlm1", type=str, default="blip")
    parser.add_argument("--vlm2", type=str, default="ofa")
    parser.add_argument("--adapter-name", type=str, default=None)
    parser.add_argument("--include-choices", action="store_true")
    parser.add_argument("--include-profile", action="store_true")
    parser.add_argument("--include-caption", action="store_true")
    parser.add_argument("--include-rationale", action="store_true")
    parser.add_argument("--include-answer", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument(
        "--prediction-out", type=str, required=True, dest="prediction_output_file"
    )
    parser.add_argument("--prediction-output-dir", type=str, default="./predictions")
    parser.add_argument("--model-output-dir", type=str, default="./pretrained_models")
    parser.add_argument("--all-out", type=str, dest="all_output_file")
    parser.add_argument(
        "--max-length",
        default=50,
        type=int,
        help="Max length of the generated answers, set small for VQA to avoid rubbish answers",
    )
    parser.add_argument(
        "--num-return-sequences",
        default=1,
        type=int,
        help="Sample multiple answers",
    )
    args = parser.parse_args()
    return args


def preprocess_vlm_text(texts):
    """Preprocess text for vlm model.
    Lower case, add space before marks.
    """
    return [re.sub("([.,!?()])", r" \1 ", text.lower()) for text in texts]


def dl_to_ld(dl):
    return [dict(zip(dl, t)) for t in zip(*dl.values())]


def ld_to_dl(ld):
    return {k: [dic[k] for dic in ld] for k in ld[0]}


def prompt_element(
    d,
    include_question=True,
    include_choices=False,
    include_profile=False,
    include_caption=False,
    include_rationale=False,
    include_answer=False,
    **kwargs,
):
    # # perturb OFA and BLIP captions, for ablation studies
    # caption_list = [kwargs['vlm2_captions'][d['question_id']], kwargs['vlm1_captions'][d['question_id']]]
    # random.shuffle(caption_list)

    # # perturb OFA and BLIP answers, for ablation studies
    # answer_list = [kwargs['vlm2_answers'][d['question_id']], kwargs['vlm1_answers'][d['question_id']]]
    # random.shuffle(answer_list)

    prompt = ""
    if include_question:
        prompt += (
            "Answer the following multiple choice question by OFA and BLIP's description and their answers to the visual question. "
            + "OFA and BLIP are two different vision-language models to provide clues. \n"
            + ("You should state your rationales. " if include_rationale else "")
            + "\n"
        )

    prompt += (
        (
            f"OFA's description: {kwargs['vlm2_captions'][d['question_id']]}\n"
            if include_caption
            else ""
        )
        + (
            f"BLIP's description: {kwargs['vlm1_captions'][d['question_id']]}\n"
            if include_caption
            else ""
        )
        + f"Q: {d['question']}"
        + "\n"
        + (
            f"OFA's answer: {kwargs['vlm2_answers'][d['question_id']]}.\n"
            if include_profile
            else ""
        )
        + (
            f"BLIP's answer: {kwargs['vlm1_answers'][d['question_id']]}.\n"
            if include_profile
            else ""
        )
        + "\n"
        + (
            f"Choices: {', '.join(d['choices'])}.\n"
            if include_choices
            else "Answer directly."
        )
        # + (f"Rationales: {' '.join(d['rationales'])}\n" if include_rationale else "")
        # + (f"Rationales: {kwargs['cot_rationale']}\n" if include_rationale else "")
        # + (f"The answer is:" if not include_rationale and not kwargs['cot'] else "A:")
        + (f"A:")
    )

    if "choices" in d:
        # + (f"The answer is:")
        prompt += f"{d['choices'][d['correct_choice_idx']]}" if include_answer else ""
        # + (f"The answer is {d['choices'][d['correct_choice_idx']]} \n Rationale: ")
        # + (f"{kwargs['rationalization'][d['question_id']]} The answer is {d['choices'][d['correct_choice_idx']]}." if include_answer else "")
    else:
        prompt += f"{d['direct_answers'][0]}" if include_answer else ""
    return prompt


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


def create_question(
    group,
    include_choices=False,
    answer=False,
    include_profile=False,
    context=None,
    include_caption=False,
    include_rationale=False,
    **kwargs,
):
    prompts = []
    for idx, d in enumerate(group):
        prompts.append(
            prompt_element(
                d,
                include_choices=include_choices,
                answer=answer,
                include_profile=include_profile,
                include_caption=include_caption,
                include_rationale=include_rationale,
                **kwargs,
            )
        )
    return prompts
