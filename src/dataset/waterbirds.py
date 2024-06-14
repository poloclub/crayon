import os

from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

from PIL import Image
import torchvision.transforms as transforms



def get_transform_cub(model_type, train, augment_data):
    scale = 256.0/224.0
    target_resolution = model_attributes[model_type]['target_resolution']  # (224,224) for resnset50, (299,299) for inceptionv3
    assert target_resolution is not None

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform

class WaterbirdsDataset(Dataset):
    """CelebA dataset."""

    def __init__(self, img_files, seg_files, labels, places, model_type, train=None, E=None, expl_upscale=False, seg_size=(7,7)):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_files = np.array(img_files)
        self.seg_files = np.array(seg_files)
        self.labels = np.array(labels)   # 1 for waterbird
        self.places = np.array(places)   # 1 for water
        self.transform = None
        self.seg_transform = None
        self.model_type = model_type

        self.transform = transforms.Compose([transforms.Resize((256,256)), transforms.CenterCrop((224,224)), transforms.ToTensor()])#, transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.seg_transform = transforms.Compose([transforms.Resize((256,256)), transforms.CenterCrop((224,224)), transforms.ToTensor(),])
        self.seg_size=seg_size

        self.E = E
        self.correct = None 
        self.expl_upscale = expl_upscale
        self.img_size = (224, 224)
        self.resize_to_img_sized_tensor = transforms.Compose([transforms.ToPILImage(), transforms.Resize(self.img_size), transforms.ToTensor()])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file = self.img_files[idx]
        seg_file = self.seg_files[idx]
        label = self.labels[idx]
        place = self.places[idx]
        img = Image.open(img_file)
        seg_img = Image.open(seg_file)
        box = np.array(Image.open(seg_file))
        box_pos_idx = np.where(box>0)
        box_l, box_r, box_t, box_b = np.min(box_pos_idx[0]), np.max(box_pos_idx[0]), np.min(box_pos_idx[1]), np.max(box_pos_idx[1])
        box = np.zeros_like(box)
        box[box_l:box_r+1, box_t:box_b+1] = 255
        box = Image.fromarray(box)
        expl = -1
        corr = -1
        orig_seg = -1
        reshaped_seg = -1
        if self.E is not None:
            expl = self.E[idx]
            if self.expl_upscale: expl = self.resize_to_img_sized_tensor(expl.astype(np.float32))
        if self.correct is not None:
            corr = self.correct[idx]
        if self.transform is not None:
            img = self.transform(img)
        if self.seg_transform is not None:
            orig_seg = self.seg_transform(seg_img)
            box = self.seg_transform(box)
            h, w = int(orig_seg.shape[1]/self.seg_size[0]), int(orig_seg.shape[2]/self.seg_size[1])
            try:
                reshaped_seg = orig_seg.reshape(self.seg_size[0], h, self.seg_size[1], w).mean(axis=(1,3))
                box = box.reshape(self.seg_size[0], h, self.seg_size[1], w).mean(axis=(1,3))
            except:
                orig_seg = orig_seg[0].unsqueeze(0)
                reshaped_seg = orig_seg.reshape(self.seg_size[0], h, self.seg_size[1], w).mean(axis=(1,3))
                box = box[0].reshape(self.seg_size[0], h, self.seg_size[1], w).mean(axis=(1,3))

        sample = {"idx": idx, "img_file": img_file, "img": img, "seg": reshaped_seg, "label": label, "place": place, "expl": expl, "corr": corr, "box": box, "orig_seg": orig_seg}
        return sample

    def slice_dataset(self, idx):
        self.img_files = self.img_files[idx]
        self.seg_files = self.seg_files[idx]
        self.labels = self.labels[idx]
        self.places = self.places[idx]
        if self.E is not None: self.E = self.E[idx]
        if self.correct is not None: self.correct = self.correct[idx]

# Check whether the segmentation map works well
if __name__=="__main__":
    segmentation_dir = "./../../data/waterbirds/segmentations"
    images_dir = "./../../data/waterbirds/images"
    vis_dir = "./../../vis/waterbirds"
    
    files = [
        "086.Pacific_Loon/Pacific_Loon_0034_75438",
        "012.Yellow_headed_Blackbird/Yellow_Headed_Blackbird_0031_8456",
        "059.California_Gull/California_Gull_0028_40666",
        "128.Seaside_Sparrow/Seaside_Sparrow_0003_796539",
        "189.Red_bellied_Woodpecker/Red_Bellied_Woodpecker_0020_182335"
        ]
    transform = transforms.Compose([
        transforms.Resize((224,224))
    ])

    np.set_printoptions(precision=3)
    dataset = WaterbirdsDataset([os.path.join(images_dir,f+".jpg") for f in files], [os.path.join(segmentation_dir,f+".png") for f in files], [""]*5, [""]*5, "resnet50")
    sample = dataset[1]
    exit()
    for i in range(5):
        sample = dataset[i]
        name = files[i].split("/")[0].split(".")[1]
        # sample["img"].save(os.path.join(vis_dir, name+".jpg"))
        # sample["seg"].save(os.path.join(vis_dir, name+"_seg.jpg"))
        if i < 2:
            print(name,sample["seg"])
    exit()


    for f in files:
        name = f.split("/")[0].split(".")[1]
        img_path = os.path.join(images_dir, f+".jpg")
        seg_path = os.path.join(segmentation_dir, f+".png")
        # img_np = np.asarray(Image.open(img_path))
        seg_np = np.asarray(Image.open(seg_path)) / 255
        Image.open(seg_path).save(os.path.join(vis_dir, name+"_seg_orig.png"))
        seg_np_reshaped = np.asarray(transform(Image.open(seg_path))) / 255
        Image.fromarray((seg_np_reshaped*255).astype(np.uint8)).save(os.path.join(vis_dir, name+"_seg.png"))
        continue
        # img_black = Image.fromarray(np.around(img_np * seg_np).astype(np.uint8))
        img_black = Image.open(img_path)
        img_black.save(os.path.join(vis_dir, name+"_.png"))
