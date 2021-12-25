import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class SantaDataset(Dataset):
    def __init__(self, txt_file="datasets/train_dataset.txt", root_dir="datasets/train/", transform=None):
        """
        :param txt_file: Path to the txt file with annotations (image filename, encoding of class).
        :type txt_file: str
        :param root_dir: Path to the folder where the images are.
        :type root_dir: str
        :param transform: callable object, optional transform to be applied to the data
        """
        df = pd.read_csv(txt_file)
        self.img_filename = df["filename"].tolist()
        self.img_label = df["class_label"].tolist()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_filename)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.img_filename[idx])
        image = Image.open(img_name)
        image = image.convert('RGB')
        label = self.img_label[idx]

        sample = {'image': image, "label": label}

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample