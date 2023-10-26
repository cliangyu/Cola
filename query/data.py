import os
from PIL import Image
import numpy as np
import torchvision.transforms as T
from concurrent.futures import ThreadPoolExecutor
import json
import pathlib
import sys
import random

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))
# print(sys.path)
from torch.utils.data import Dataset
from torchvision import transforms
import time
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD:HH:mm:ss}</green> | <cyan>{name}</cyan><cyan>:line {line}</cyan>: <level>{message}</level>",
    level="INFO",
)


def load_dataset(dataset_dir, split, dataset_name="aokvqa", version="v1p0"):
    assert split in ["train", "val", "test", "test_w_ans"]
    if dataset_name == "aokvqa":
        assert version in ["v1p0", "v1p1"]
        dataset_path = os.path.join(
            dataset_dir, dataset_name, f"{dataset_name}_{version}_{split}.json"
        )
    else:
        dataset_path = os.path.join(
            dataset_dir, dataset_name, f"converted_{dataset_name}_{split}.json"
        )
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    return dataset


def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")


def get_coco_14_path(split, image_id, coco_dir):
    return os.path.join(
        coco_dir, f"{split}2014", f"COCO_{split}2014_{image_id:012}.jpg"
    )


def get_clevr_path(split, image_id, clevr_dir):
    return os.path.join(clevr_dir, split, f"CLEVR_{split}_{image_id:06}.png")


from copy import deepcopy


class VisualReasoningDataset(Dataset):
    def __init__(
        self,
        dataset_dir: pathlib.Path,
        dataset_name: str,
        split: str,
        transform: str = "none",
        include_image: bool = True,
    ):
        dataset = load_dataset(
            dataset_dir=dataset_dir, dataset_name=dataset_name, split=split
        )

        # cut down the dataset for fast training
        if dataset_name in ["gqa", "vqav2", "clevr"] and split == "train":
            dataset = random.sample(dataset, 200000)

        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 480
        self.ofa_transform = transforms.Compose(
            [
                transforms.Resize(
                    (resolution, resolution), interpolation=T.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        assert transform in ["none", "ofa"]

        self.dataset_name = dataset_name
        self.item = []
        self.question_ids = []
        self.image_id = []
        self.question = []
        self.choices = []
        self.correct_choice_idx = []
        self.mc_answer = []
        self.difficult_direct_answer = []
        self.rationales = []
        self.image_path = []
        self.image = []
        self.split = split
        self.include_image = include_image
        self.transform = transform

        # assume aokvqa is in the same dir as coco
        if dataset_name in ["aokvqa", "okvqa", "vqav2"]:
            self.image_path_finder = os.path.join(dataset_dir, "coco")
        elif dataset_name == "esnlive":
            self.image_path_finder = os.path.join(
                dataset_dir, "esnlive/flickr30k_images"
            )
        elif dataset_name == "vcr":
            self.image_path_finder = os.path.join(dataset_dir, "vcr/vcr1images")
        elif dataset_name == "vsr":
            self.image_path_finder = os.path.join(dataset_dir, "vsr/trainval2017")
        elif dataset_name == "gqa":
            self.image_path_finder = os.path.join(dataset_dir, "gqa/images")
        elif dataset_name == "clevr":
            self.image_path_finder = os.path.join(dataset_dir, "clevr/images")

        logger.info(f"Loading {dataset_name} {split} dataset")
        start_time = time.time()
        with ThreadPoolExecutor() as executor:
            self.item = executor.map(self.preprocess_data, dataset)

        self.item = list(self.item)
        end_time = time.time()
        logger.info("Loading dataset took {:.2f} seconds".format(end_time - start_time))

    def preprocess_data(self, d):
        return_dict = {}
        return_dict["question_id"] = d["question_id"]
        return_dict["image_id"] = d["image_id"]
        return_dict["question"] = d["question"]
        if self.dataset_name not in ["vsr", "esnlive"] and self.split != "test":
            return_dict["direct_answers"] = d["direct_answers"]
        if self.dataset_name in ["aokvqa", "vsr", "esnlive"]:
            return_dict["choices"] = d["choices"]
        if self.dataset_name == "aokvqa":
            return_dict["difficult_direct_answer"] = d["difficult_direct_answer"]
        if self.dataset_name == "vsr":
            return_dict["correct_choice_idx"] = d["correct_choice_idx"]
        if self.dataset_name in ["aokvqa", "vsr", "esnlive"]:
            if self.split != "test":
                return_dict["correct_choice_idx"] = d["correct_choice_idx"]
                return_dict["mc_answer"] = d["choices"][d["correct_choice_idx"]]
                if self.dataset_name != "vsr":
                    return_dict["rationales"] = d["rationales"]

        if self.dataset_name in ["aokvqa"]:
            d["image_path"] = get_coco_path(
                self.split, d["image_id"], self.image_path_finder
            )
        elif self.dataset_name in ["okvqa", "vqav2"]:
            d["image_path"] = get_coco_14_path(
                self.split, d["image_id"], self.image_path_finder
            )
        elif self.dataset_name in ["clevr"]:
            d["image_path"] = get_clevr_path(
                self.split, d["image_id"], self.image_path_finder
            )
        else:
            d["image_path"] = os.path.join(self.image_path_finder, d["image_id"])

        return_dict["image_path"] = d["image_path"]
        return return_dict

    def __getitem__(self, i: int):
        # cut reference to avoid memory leak
        # d should be single individual instance
        d = deepcopy(self.item[i])
        if self.include_image:
            with Image.open(d["image_path"]) as img:
                image = img.convert("RGB")
                if self.transform == "ofa":
                    image = self.ofa_transform(image)
                elif self.transform == "none":
                    image = np.array(image)
                else:
                    raise NotImplementedError
            d["image"] = image
        return d

    def __len__(self) -> int:
        return len(self.item)


def custom_collate(data):
    return data
