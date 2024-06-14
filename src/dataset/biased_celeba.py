# source: https://github.com/akrishna77/bias-discovery/blob/main/datasets/CelebADataset.py

import os

from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

from PIL import Image


class BiasedCelebADataset(Dataset):
    """CelebA dataset."""

    def __init__(self, csv_file, root_dir, transform=None, E=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.attributes = None
        self.csv_file = csv_file

        self.attributes = pd.read_csv(csv_file, delimiter="\s+", engine="python", header=0)
        self.labels = (self.attributes.iloc[:, self.attributes.columns.get_loc("Smiling")] + 1) // 2
        self.labels = self.labels.to_numpy()
        self.blackhairs = (self.attributes.iloc[:, self.attributes.columns.get_loc("Black_Hair")] + 1) // 2
        self.root_dir = root_dir
        self.transform = transform


        self.E = E 
        self.correct = None 

    def __len__(self):
        return len(self.attributes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = str(self.attributes.iloc[idx, 0])
        img_path = os.path.join(str(self.root_dir), img_name)

        # attr = self.attributes.iloc[idx, self.attributes.columns.get_loc("Smiling")]
        # if type(attr)==np.int64 and attr == -1:
        #     attr = 0
        # elif type(attr)==np.ndarray and -1 in attr:
        #     attr = (attr+1)//2

        # blackhair = self.attributes.iloc[idx, self.attributes.columns.get_loc("Black_Hair")]
        # if type(blackhair)==np.int64 and blackhair == -1:
        #     blackhair = 0
        # elif type(blackhair)==np.ndarray and -1 in blackhair:
        #     blackhair = (blackhair+1)//2
        attr = self.labels[idx]
        blackhair = self.blackhairs[idx]

        expl = -1
        corr = -1
        if self.E is not None:
            expl = self.E[idx]
        if self.correct is not None:
            corr = self.correct[idx]

        sample = {"idx": idx, "img_name": img_name, "img_path": img_path, "img": Image.open(img_path), "label": attr, "expl": expl, "blackhair": blackhair, "correct": corr}

        if self.transform:
            sample["img"] = self.transform(sample["img"])

        return sample

def saveBiasedCelebADFSourceFile(data_dir, df_source_file, train=True):
    df = pd.read_csv(f"{data_dir}/list_attr_celeba.txt", delimiter="\s+", header=0)
    df.replace(to_replace=-1, value=0, inplace=True)
    dataset_df = df.copy()

    # black, smiling: 18643 -> 10000
    # black, not smiling: 20263 -> 1000 -> 100 (strong)
    # blond, smiling: 14243 -> 1000 -> 100 (strong)
    # blond, not smiling: 10024 -> 10000
    # not black, not smiling: 64427 -> 30000 -> Not use
    # not black, smiling: 59437 -> 100 -> Not use
    # (59437+64427)/(18643+20263)

    if train:
        dataset_df = dataset_df[:162770]
        dataset_df.drop(dataset_df[(dataset_df["Black_Hair"] == 0) & (dataset_df["Blond_Hair"] == 0)].index, inplace=True)
        dataset_df.drop(dataset_df[(dataset_df["Black_Hair"] == 1) & (dataset_df["Blond_Hair"] == 1)].index, inplace=True)
        dataset_df.drop(dataset_df[(dataset_df["Black_Hair"] == 1) & (dataset_df["Smiling"] == 1)].sample(n=8641).index,inplace=True)
        dataset_df.drop(dataset_df[(dataset_df["Blond_Hair"] == 1) & (dataset_df["Smiling"] == 1)].sample(n=13241).index,inplace=True)
        dataset_df.drop(dataset_df[(dataset_df["Black_Hair"] == 1) & (dataset_df["Smiling"] == 0)].sample(n=19263).index,inplace=True)
        dataset_df.drop(dataset_df[(dataset_df["Blond_Hair"] == 1) & (dataset_df["Smiling"] == 0)].sample(n=24).index,inplace=True)
        # ADDED FOR STRONGER BIAS ## DEBUG
        dataset_df.drop(dataset_df[(dataset_df["Black_Hair"] == 1) & (dataset_df["Smiling"] == 0)].sample(n=900).index,inplace=True)
        dataset_df.drop(dataset_df[(dataset_df["Blond_Hair"] == 1) & (dataset_df["Smiling"] == 1)].sample(n=900).index,inplace=True)
    else:
        dataset_df = dataset_df[162770:]
        dataset_df.drop(dataset_df[(dataset_df["Black_Hair"] == 0) & (dataset_df["Blond_Hair"] == 0)].index, inplace=True)
        dataset_df.drop(dataset_df[(dataset_df["Black_Hair"] == 1) & (dataset_df["Blond_Hair"] == 1)].index, inplace=True)
        dataset_df.drop(dataset_df[(dataset_df["Black_Hair"] == 1) & (dataset_df["Smiling"] == 1)].sample(n=2614).index,inplace=True)
        dataset_df.drop(dataset_df[(dataset_df["Blond_Hair"] == 1) & (dataset_df["Smiling"] == 1)].sample(n=1453).index,inplace=True)
        dataset_df.drop(dataset_df[(dataset_df["Black_Hair"] == 1) & (dataset_df["Smiling"] == 0)].sample(n=2951).index,inplace=True)
        dataset_df.drop(dataset_df[(dataset_df["Blond_Hair"] == 1) & (dataset_df["Smiling"] == 0)].sample(n=262).index,inplace=True)

    if train: print("Done --- Training Set Data Filtering")
    else: print("Done --- Test Set Data Filtering")
    
    print(f"Black, Smiling: ", len(dataset_df[(dataset_df["Black_Hair"] == 1) & (dataset_df["Smiling"] == 1)]))
    print(f"Blond, Smiling: ", len(dataset_df[(dataset_df["Blond_Hair"] == 1) & (dataset_df["Smiling"] == 1)]))
    print(f"Black, Not Smiling: ", len(dataset_df[(dataset_df["Black_Hair"] == 1) & (dataset_df["Smiling"] == 0)]))
    print(f"Blond, Not Smiling: ", len(dataset_df[(dataset_df["Blond_Hair"] == 1) & (dataset_df["Smiling"] == 0)]))

    dataset_df.to_csv(df_source_file, sep=" ", index=False)

def saveUnbiasedCelebADFSourceFile(data_dir, unbiased_celeba_df_source_file, train_dataset):
    print(f"Training data size before biasing: {len(train_dataset)}")
    df = pd.read_csv(f"{data_dir}/list_attr_celeba.txt", delimiter="\s+", header=0)
    df.replace(to_replace=-1, value=0, inplace=True)

    train_set_df = df.copy()
    train_set_df = train_set_df[:162770]

    train_set_df.drop(
        train_set_df[(train_set_df["Black_Hair"] == 1) & (train_set_df["Smiling"] == 1)].sample(n=643).index,
        inplace=True,
    )
    train_set_df.drop(
        train_set_df[(train_set_df["Black_Hair"] == 0) & (train_set_df["Smiling"] == 1)].sample(n=3437).index,
        inplace=True,
    )
    train_set_df.drop(
        train_set_df[(train_set_df["Black_Hair"] == 1) & (train_set_df["Smiling"] == 0)].sample(n=2263).index,
        inplace=True,
    )
    train_set_df.drop(
        train_set_df[(train_set_df["Black_Hair"] == 0) & (train_set_df["Smiling"] == 0)].sample(n=8427).index,
        inplace=True,
    )

    print(f"Black Hair, Smiling: ", len(train_set_df[(train_set_df["Black_Hair"] == 1) & (train_set_df["Smiling"] == 1)]))
    print(f"Not black Hair, Smiling: ",len(train_set_df[(train_set_df["Black_Hair"] == 0) & (train_set_df["Smiling"] == 1)]))
    print(f"Black Hair, Not Smiling: ", len(train_set_df[(train_set_df["Black_Hair"] == 1) & (train_set_df["Smiling"] == 0)]))
    print(f"Not black Hair, Not Smiling: ", len(train_set_df[(train_set_df["Black_Hair"] == 0) & (train_set_df["Smiling"] == 0)]))

    train_set_df.to_csv(unbiased_celeba_df_source_file, sep=" ", index=False)