import json
import argparse


def convert(questions, split):
    return_list = []
    qid_dict = {}
    for i, question in enumerate(questions):
        return_dict = {}
        if split in ["train", "val"]:
            return_dict["direct_answers"] = [question["answer"]]
            return_dict["rationales"] = [question["fullAnswer"]]
        return_dict["question"] = question["question"]
        return_dict["image_id"] = question["image"]
        return_dict["split"] = split
        if split in ["train"]:
            print(str(i))
            return_dict["question_id"] = str(i)
        else:
            return_dict["question_id"] = str(
                question["question_id"]
            )  # not usable for GQA training set, duplicate ids
        qid_dict[return_dict["question_id"]] = 1
        return_list.append(return_dict)
    print(len(qid_dict))
    print(len(return_list))
    return return_list


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

    json.dump(questions, args.output_file)
