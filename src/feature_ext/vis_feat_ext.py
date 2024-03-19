from transformers import ViTImageProcessor, ViTModel
from data import ImageDataset
from torch.utils.data import DataLoader
import numpy as np
import os
import torch
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 사전 훈련된 모델과 특징 추출기 로드
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)
model.eval()

# dataloader
dataset = ImageDataset(data_root='/home/minseongjae/Downloads/AffWild2')
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

# 데이터 순회하며 특징 추출 및 저장
with torch.no_grad():
    for data in tqdm(dataloader):
        # 특징 추출
        inputs = image_processor(images=data['image'], return_tensors="pt", do_rescale=False, do_resize=False).to(device)
        outputs = model(**inputs)
        features = outputs.pooler_output
        for i, path in enumerate(data['image_path']):
            path = path.replace('cropped_aligned', 'Features').replace('.jpg', '.npy')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, features[i].to('cpu').numpy())