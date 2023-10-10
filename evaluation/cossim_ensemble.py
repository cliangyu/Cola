import argparse
import pathlib
import json
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from query.data import load_dataset


def ensemble(args, dataset, predictions_1, predictions_2, device="cpu"):
    if isinstance(dataset, list):
        dataset = {dataset[i]["question_id"]: dataset[i] for i in range(len(dataset))}

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    model.to(device)
    predictions = {}
    for q in tqdm(predictions_1.keys()):
        choices = dataset[q]["choices"]
        choice_embeddings = model.encode(
            [predictions_1[q], predictions_2[q]] + choices, convert_to_tensor=True
        )
        a_score = cos_sim(choice_embeddings[0], choice_embeddings[2:])
        b_score = cos_sim(choice_embeddings[1], choice_embeddings[2:])
        if args.strategy == "average":
            ensemble_score = (a_score + b_score) / 2
        else:
            raise ValueError(f"Unknown ensemble strategy: {args.strategy}")
        answer_idx = ensemble_score.argmax().item()
        predictions[q] = choices[answer_idx]

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["aokvqa", "vcr", "esnlive", "vsr", "gqa", "vqav2", "clevr"],
        required=True,
    )
    parser.add_argument(
        "--split", type=str, choices=["train", "val", "test"], required=True
    )
    parser.add_argument(
        "--pred-1", type=argparse.FileType("r"), required=True, dest="prediction_file_1"
    )
    parser.add_argument(
        "--pred-2", type=argparse.FileType("r"), required=True, dest="prediction_file_2"
    )
    parser.add_argument("--strategy", type=str, default="average", dest="strategy")
    parser.add_argument(
        "--out", type=argparse.FileType("w"), required=True, dest="output_file"
    )
    args = parser.parse_args()

    dataset = load_dataset(args.data_dir, dataset_name=args.dataset, split=args.split)
    predictions_1 = json.load(args.prediction_file_1)
    predictions_2 = json.load(args.prediction_file_2)
    predictions = ensemble(args, dataset, predictions_1, predictions_2)

    json.dump(predictions, args.output_file)
