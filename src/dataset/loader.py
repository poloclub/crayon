import sys 
sys.path.append("./..")
import os 

from dataset.biased_celeba import BiasedCelebADataset
from dataset.waterbirds import WaterbirdsDataset
from dataset.background import BACKGROUND_TRANSFORM_TRAIN, BACKGROUND_TRANSFORM_TEST, BACKGROUND_TRANSFORM_SEG, ImageFolder

import pandas as pd
import numpy as np
import torch 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class DatasetWithHumanFB(Dataset):
    def __init__(self, X, y, E, fb, data_name="waterbirds", img_transform=None, root_dir=None):
        self.data_name = data_name
        self.root_dir = root_dir        
        self.X = X 
        self.y = y
        self.E = E
        self.fb = -np.ones_like(np.array(fb=="no", dtype=np.float32)) # fb: -1 by default, 1 for no, 0 for yes, 0.5 for maybe
        self.fb = self.fb + np.array(fb=="no", dtype=np.float32) * 2 + np.array(fb=="yes", dtype=np.float32) + np.array(fb=="maybe", dtype=np.float32) * 1.5
        self.img_transform = img_transform
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = dict()
        if type(self.X[idx]) in [np.ndarray, torch.Tensor]: img = self.X[idx]
        else:
            # if self.X[idx] is str, read the image and transform
            # assert self.root_dir is not None 
            img_name = str(self.X[idx])
            if self.root_dir is not None: img_path = os.path.join(self.root_dir, img_name)
            else: img_path = img_name
            img = Image.open(img_path)
            if self.data_name=="background": img = img.convert("RGB")
            if self.img_transform is not None: img = self.img_transform(img)

        expl = self.E[idx]
        sample = {"img": img, "label": self.y[idx], "expl": expl, "fb": self.fb[idx]}

        return sample


def getLoader(data_name="waterbirds", data_dir=".", batch_size=256, num_workers=1, model_type=None, verbose=False, expl_upscale=False, seg_size=(7,7), reshape_seg=True):
    train_set, val_set, test_set = None, None, None
    transform_size=224
    
    if data_name == "biased_celeba":
        transform=transforms.Compose([transforms.Resize(transform_size), transforms.ToTensor()])
        biased_df_train_source_file = f"{data_dir}/biased_celeba_black_hair_1_100_train.csv"
        biased_df_test_source_file = f"{data_dir}/biased_celeba_black_hair_test.csv"
        train_set = BiasedCelebADataset(biased_df_train_source_file, f"{data_dir}/celeba/img_align_celeba", transform)
        test_set = BiasedCelebADataset(biased_df_test_source_file, f"{data_dir}/celeba/img_align_celeba", transform)
        print(f"Data size of BiasedCelebA --- Train: {len(train_set)}, Test: {len(test_set)}")

    elif data_name == "waterbirds":
        img_dir = os.path.join(data_dir, "images")
        seg_dir = os.path.join(data_dir, "segmentations")
        metadata_df = pd.read_csv(os.path.join(data_dir, "metadata.csv"))

        datasets = {}
        counts = {}
        
        for split, split_label in [(0, 'train'), (1, 'val'), (2, 'test')]:
            split_df = metadata_df.loc[metadata_df["split"]==split, :]
            labels = split_df["y"].values 
            places = split_df["place"].values 

            img_filenames = split_df["img_filename"].values 
            seg_filenames = [f.replace("jpg","png") for f in split_df["img_filename"].values]
            img_filenames = [os.path.join(img_dir,f) for f in img_filenames]
            seg_filenames = [os.path.join(seg_dir,f) for f in seg_filenames]
            
            datasets[split_label] = WaterbirdsDataset(img_filenames, seg_filenames, labels, places, model_type, expl_upscale=expl_upscale, seg_size=seg_size)

            counts[split_label] = [
                np.sum(split_df.loc[split_df["y"]==1, "place"]==1),
                np.sum(split_df.loc[split_df["y"]==1, "place"]==0),
                np.sum(split_df.loc[split_df["y"]==0, "place"]==1),
                np.sum(split_df.loc[split_df["y"]==0, "place"]==0)
            ]

        train_set = datasets["train"]
        val_set = datasets["val"]
        test_set = datasets["test"]

        print(f"Data distribution of Train Set --- w_waterbird: {counts['train'][0]}, l_waterbird: {counts['train'][1]}, w_landbird: {counts['train'][2]}, l_landbird: {counts['train'][3]}")
        print(f"Data distribution of Val Set --- w_waterbird: {counts['val'][0]}, l_waterbird: {counts['val'][1]}, w_landbird: {counts['val'][2]}, l_landbird: {counts['val'][3]}")
        print(f"Data distribution of Test Set --- w_waterbird: {counts['test'][0]}, l_waterbird: {counts['test'][1]}, w_landbird: {counts['test'][2]}, l_landbird: {counts['test'][3]}")

    elif data_name=="background":
        train_transform = BACKGROUND_TRANSFORM_TRAIN
        test_transform = BACKGROUND_TRANSFORM_TEST 
        seg_transform = BACKGROUND_TRANSFORM_SEG
        original_train_set = ImageFolder(os.path.join(data_dir, "original", "train"), train_transform, seg_transform=seg_transform, reshape_seg=reshape_seg)
        original_test_set = ImageFolder(os.path.join(data_dir, "original", "val"), test_transform, seg_transform=seg_transform)
        mixed_same_test_set = ImageFolder(os.path.join(data_dir, "mixed_same", "val"), test_transform, seg_transform=seg_transform)
        mixed_rand_test_set = ImageFolder(os.path.join(data_dir, "mixed_rand", "val"), test_transform, seg_transform=seg_transform)
        only_fg_test_set = ImageFolder(os.path.join(data_dir, "only_fg", "val"), test_transform, seg_transform=seg_transform)
        original_train_loader = DataLoader(original_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        original_test_loader = DataLoader(original_test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        mixed_same_test_loader = DataLoader(mixed_same_test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        mixed_rand_test_loader = DataLoader(mixed_rand_test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        only_fg_test_loader = DataLoader(only_fg_test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return (original_train_set, original_test_set, mixed_same_test_set, mixed_rand_test_set, only_fg_test_set), (original_train_loader, original_test_loader, mixed_same_test_loader, mixed_rand_test_loader, only_fg_test_loader)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_set is not None else None
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return (train_set, val_set, test_set), (train_loader, val_loader, test_loader) 


def getLoaderWithHumanFB(Xs, ys, expls, fbs, batch_size, data_name="waterbirds", img_transform=None, root_dir=None):
    dataset_humanfb = DatasetWithHumanFB(Xs, ys, expls, fbs, data_name, img_transform, root_dir)
    return DataLoader(dataset_humanfb, batch_size=min(batch_size, len(Xs)), shuffle=True)

def getUnshuffledLoader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

if __name__=="__main__":
    getLoader("mnli")