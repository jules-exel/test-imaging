import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class AnimalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes

dataset = AnimalDataset(
    data_dir='./dataset/'
)

image, label = dataset[400]
