import pickle

import torch
from torch.utils.data import IterableDataset


class BERTPretrainingIterableDataset(IterableDataset):
    def __init__(self, dataset_path: str):
        super().__init__()
        self.dataset_path = dataset_path

    def __iter__(self):
        with open(self.dataset_path, "rb") as f:
            while True:
                try:
                    features = pickle.load(f)
                    yield {key: torch.tensor(value) for key, value in features.items()}
                except EOFError:
                    break
