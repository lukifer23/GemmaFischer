import sys
from pathlib import Path

import torch
from datasets import Dataset

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.training.train_lora_poc import set_seed


def split_with_seed(seed: int):
    set_seed(seed)
    dataset = Dataset.from_dict({"x": list(range(10))})
    split = dataset.train_test_split(test_size=0.3, seed=seed)
    return split["train"]["x"], split["test"]["x"]


def weights_with_seed(seed: int):
    set_seed(seed)
    model = torch.nn.Linear(5, 1)
    return model.weight.detach().clone()


def test_dataset_split_deterministic():
    train1, test1 = split_with_seed(1234)
    train2, test2 = split_with_seed(1234)
    assert train1 == train2
    assert test1 == test2


def test_weight_init_deterministic():
    w1 = weights_with_seed(4321)
    w2 = weights_with_seed(4321)
    assert torch.equal(w1, w2)
