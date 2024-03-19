import torch
import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# 사용자 정의 데이터셋 클래스 정의
class ImageDataset(Dataset):
    def __init__(self, data_root, transform=transform):
        self.data_root = data_root
        self.transform = transform
        self.image_paths = glob(os.path.join(data_root, 'batch*', '*', '*','*.jpg'))
        # sort
        self.image_paths = sorted(self.image_paths)
        # half
        self.image_paths = self.image_paths[:len(self.image_paths)//2+1]
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        data = {}
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        data['image'] = image
        data['image_path'] = image_path
        return data