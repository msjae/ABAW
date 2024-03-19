import torch
from torch import nn
from dataset_msj import MyDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import os
import argparse
from utils import get_config
import yaml

# Load the configuration
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, help='Path to the configuration file')
args = parser.parse_args()

# Load the configuration
cfg = get_config(args.config_path)  # Update this path as necessary
# save the config yaml
# if same dir exists, add number to the dir
if os.path.exists(cfg['dir']):
    i = 1
    while True:
        if not os.path.exists(cfg['dir'] + f"_{i}"):
            dir = cfg['dir'] + f"_{i}"
            break
        i += 1
else:
    dir = cfg['dir']
os.makedirs(dir, exist_ok=True)
with open(dir + '/config.yaml', 'w') as f:
    yaml.dump(cfg, f)

# Initialize the dataset
if cfg['task'] == 'VA':
    train_anno_path = os.path.join(cfg['anno_path'], f"VA_Estimation_Challenge_Train.csv")
    val_anno_path = os.path.join(cfg['anno_path'], f"VA_Estimation_Challenge_Validation.csv")
elif cfg['task'] == 'EXPR':
    train_anno_path = os.path.join(cfg['anno_path'], f"EXPR_Recognition_Challenge_Train.csv")
    val_anno_path = os.path.join(cfg['anno_path'], f"EXPR_Recognition_Challenge_Validation.csv")
else:
    train_anno_path = os.path.join(cfg['anno_path'], f"AU_Detection_Challenge_Train.csv")
    val_anno_path = os.path.join(cfg['anno_path'], f"AU_Detection_Challenge_Validation.csv")
train_dataset = MyDataset(data_root = cfg['data_path'], csv_file=train_anno_path, 
                            return_img=cfg['return_img'], return_aud=cfg['return_aud'], return_vis=cfg['return_vis'],
                            return_seq=cfg['return_seq'], seq_size=cfg['seq_size'], aug=cfg['augmentation'])
val_dataset = MyDataset(data_root = cfg['data_path'], csv_file=val_anno_path, 
                            return_img=cfg['return_img'], return_aud=cfg['return_aud'], return_vis=cfg['return_vis'],
                            return_seq=cfg['return_seq'], seq_size=cfg['seq_size'])

# Initialize the data loaders
train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])


# Load a pre-trained ViT model
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')

# Modify the model for your number of classes (e.g., 8)
model.classifier = nn.Linear(model.classifier.in_features, 8)

# Define your optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train the model (here simplified for brevity)
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # forward pass
        outputs = model(images).logits
        loss = criterion(outputs, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
