import json
import argparse


def convert(questions, split):
    for question in questions:
        question["direct_answers"] = question.pop("answer")
        question["image_id"] = int(
            question.pop("image").split("_")[-1].split(".jpg")[0]
        )
        del question["dataset"]
        question["split"] = split
        question["question_id"] = str(question["question_id"])
    return questions


def convert_ids_to_str(questions):
    for question in questions:
        question["question_id"] = str(question["question_id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in", type=argparse.FileType("r"), required=True, dest="prediction_file"
    )
    parser.add_argument("--split", type=str, required=True, dest="split")
    parser.add_argument(
        "--out", type=argparse.FileType("w"), required=True, dest="output_file"
    )
    args = parser.parse_args()

    questions = json.load(args.prediction_file)
    questions = convert(questions, args.split)
    # questions = convert(questions)

    json.dump(questions, args.output_file)
