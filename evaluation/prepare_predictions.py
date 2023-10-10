import argparse
import pathlib
import json

from query.data import load_dataset


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
    parser.add_argument("--mc", type=argparse.FileType("r"), dest="mc_pred_file")
    parser.add_argument("--da", type=argparse.FileType("r"), dest="da_pred_file")
    parser.add_argument("--out", type=argparse.FileType("w"), dest="output_file")
    args = parser.parse_args()
    assert args.mc_pred_file or args.da_pred_file

    dataset = load_dataset(args.data_dir, dataset_name=args.dataset, split=args.split)
    mc_preds = json.load(args.mc_pred_file) if args.mc_pred_file else None
    da_preds = json.load(args.da_pred_file) if args.da_pred_file else None
    predictions = {}

    for d in dataset:
        q = d["question_id"]
        predictions[q] = {}
        if mc_preds and q in mc_preds.keys():
            predictions[q]["multiple_choice"] = mc_preds[q]
        if da_preds and q in da_preds.keys():
            predictions[q]["direct_answer"] = da_preds[
                q
            ].strip()  # strip spaces for OFA

    json.dump(predictions, args.output_file)
