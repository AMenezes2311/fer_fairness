# datasets/fer2013_dataset.py

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class FER2013Dataset(Dataset):
    """
    FER2013 dataset from a Kaggle-style CSV file.

    Args:
        csv_path (str): Path to fer2013.csv
        split (str): "Training", "PublicTest", or "PrivateTest"
        transform (callable): torchvision transforms
    """
    def __init__(self, csv_path, split="Training", transform=None):
        super().__init__()
        df = pd.read_csv(csv_path)

        # keep original CSV row index as a column
        df["dataset_index"] = df.index

        # filter by split but do not reset index
        self.data = df[df["Usage"] == split].copy()
        self.transform = transform

        self.num_classes = self.data["emotion"].nunique()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = int(row["emotion"])
        pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
        img = pixels.reshape(48, 48)  # grayscale 48x48

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        dataset_index = int(row["dataset_index"])  # original CSV index
        return img, label, dataset_index
