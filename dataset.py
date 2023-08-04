from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pandas as pd
import torch
from PIL import Image
import os

# Dataset class
class CustomDataset(Dataset):
    def __init__(self, img_folder, metadata_file,split="train"):
        self.img_folder = img_folder
        self.metadata = pd.read_csv(metadata_file)
        self.metadata['gender'] = self.metadata['gender'].map({'Female': 0, 'Male': 1})
        if split == "train":
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.metadata.iloc[idx]['file'])
        img = Image.open(img_path)
        img = self.transform(img)
        gender = torch.tensor(self.metadata.iloc[idx]['gender']).long()
        return img, gender

if __name__=="__main__":
    dataset = CustomDataset('fairface', 'fairface_label_train.csv')
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True,num_workers=10)
    for img,gender in loader:
        print(img.shape)
