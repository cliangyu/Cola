import argparse
import pathlib
import json
import glob

from query.data import load_dataset


def eval_aokvqa(args, dataset, preds, multiple_choice=False, strict=True):
    if isinstance(dataset, list):
        dataset = {dataset[i]["question_id"]: dataset[i] for i in range(len(dataset))}

    if multiple_choice is False:
        if "difficult_direct_answer" in dataset[list(dataset.keys())[0]].keys():
            dataset = {
                k: v
                for k, v in dataset.items()
                if v["difficult_direct_answer"] is False
            }

    if strict:
        dataset_qids = set(dataset.keys())
        preds_qids = set(preds.keys())
        assert dataset_qids.issubset(preds_qids)

    # dataset = q_id (str) : dataset element (dict)
    # preds = q_id (str) : prediction (str)

    acc = []

    for q in dataset.keys():
        if q not in preds.keys():
            acc.append(0.0)
            continue

        pred = preds[q]
        if "direct_answers" in dataset[q].keys():
            direct_answers = dataset[q]["direct_answers"]
        # else:
        #     choices = dataset[q]["choices"]
        #     direct_answers = [dataset[q]["choices"][dataset[q]["correct_choice_idx"]]]
        if "choices" in dataset[q].keys():
            choices = dataset[q]["choices"]

        ## Multiple Choice setting
        if multiple_choice:
            if strict:
                assert pred in choices, "Prediction must be a valid choice"
            correct_choice_idx = dataset[q]["correct_choice_idx"]
            acc.append(float(pred == choices[correct_choice_idx]))
        ## Direct Answer setting
        else:
            direct_answers = dataset[q]["direct_answers"]
            num_match = sum([pred == da for da in direct_answers])
            vqa_acc = (
                min(1.0, num_match / 3.0)
                if args.dataset == "aokvqa"
                else min(1.0, num_match)
            )
            acc.append(vqa_acc)

    acc = sum(acc) / len(acc) * 100

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["aokvqa", "vcr", "esnlive", "vsr", "okvqa", "vqav2", "gqa", "clevr"],
        required=True,
    )
    parser.add_argument(
        "--split", type=str, choices=["train", "val", "test"], required=True
    )
    parser.add_argument("--preds", type=str, required=True, dest="prediction_files")
    args = parser.parse_args()

    dataset = load_dataset(args.data_dir, dataset_name=args.dataset, split=args.split)

    for prediction_file in glob.glob(args.prediction_files):
        predictions = json.load(open(prediction_file, "r"))

        # Multiple choice

        mc_predictions = {}

        for q in predictions.keys():
            if "multiple_choice" in predictions[q].keys():
                mc_predictions[q] = predictions[q]["multiple_choice"]

        if mc_predictions != {}:
            mc_acc = eval_aokvqa(
                args, dataset, mc_predictions, multiple_choice=True, strict=False
            )
            print(prediction_file, "MC", mc_acc)

        # Direct Answer

        da_predictions = {}

        for q in predictions.keys():
            if "direct_answer" in predictions[q].keys():
                da_predictions[q] = predictions[q]["direct_answer"]

        if da_predictions != {}:
            print("evaluate direct answers.")
            da_acc = eval_aokvqa(
                args, dataset, da_predictions, multiple_choice=False, strict=False
            )
            print(prediction_file, "DA", da_acc)
