import argparse
import pathlib
import json
from tqdm import tqdm


def postprocess(predictions, phrase, device="cpu"):
    # if isinstance(dataset, list):
    #     dataset = { dataset[i]['question_id'] : dataset[i] for i in range(len(dataset)) }

    # if all([p in dataset[q]['choices'] for q, p in predictions.items()]):
    #     return predictions

    for q in tqdm(predictions.keys()):
        # predictions[q] = predictions[q].split('The answer is:')[-1] if 'The answer is:' in predictions[q] else predictions[q]
        predictions[q] = (
            # predictions[q].split(phrase)[-1].strip('"').strip(" .")
            # predictions[q].split("A:")[1].split("\n")[0] # for mistral
            predictions[q].split("\n")[0] # for vicuna
            # if phrase in predictions[q]
            # else predictions[q]
        )
        
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred", type=argparse.FileType("r"), required=True, dest="prediction_file"
    )
    parser.add_argument(
        "--out", type=argparse.FileType("w"), required=True, dest="output_file"
    )
    parser.add_argument("--phrase", type=str, default="answer", dest="phrase")
    args = parser.parse_args()

    predictions = json.load(args.prediction_file)
    predictions = postprocess(predictions, args.phrase)

    json.dump(predictions, args.output_file)
