import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import yaml
import argparse
from tqdm import tqdm

from src.data.image_feature_dataset import ImageFeatureDataset
from src.models.lstm_emotion import LSTMEmotionModel
from src.models.transformer_emotion import TransformerEmotionModel
from src.loss import get_loss
from src.metrics import ExprMetric, VAMetric, AUMetric

def train_one_epoch(model, dataloader, criterion, optimizer, device, max_iter=None):
    model.train()
    running_loss = 0.0
    for i, (features, labels) in enumerate(tqdm(dataloader, desc="Training")):
        if max_iter and i >= max_iter:
            break
        features, labels = features.to(device), labels.to(device)

        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
    return running_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion, device, task, max_iter=None):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for i, (features, labels) in enumerate(tqdm(dataloader, desc="Validation")):
            if max_iter and i >= max_iter:
                break
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * features.size(0)

            if task == "expr":
                preds = torch.softmax(outputs, dim=1).cpu().numpy()
                labels = labels.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
            elif task == "va":
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            elif task == "au":
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)

    if task == "expr":
        metric = ExprMetric()(np.array(all_preds), np.array(all_labels))
    elif task == "va":
        metric = VAMetric()(np.array(all_preds), np.array(all_labels))
    elif task == "au":
        metric = AUMetric()(np.array(all_preds), np.array(all_labels))
    else:
        metric = {}

    return avg_loss, metric

def main(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = config["device"]
    max_iter = config.get("train", {}).get("max_iter", None)

    # Dataset
    train_dataset = ImageFeatureDataset(
        feature_root=config["data"]["feature_root"],
        label_root=config["data"]["label_root"],
        task=config["task"],
        split="train",
        seq_len=config["data"]["seq_len"]
    )
    val_dataset = ImageFeatureDataset(
        feature_root=config["data"]["feature_root"],
        label_root=config["data"]["label_root"],
        task=config["task"],
        split="val",
        seq_len=config["data"]["seq_len"]
    )
    train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["train"]["batch_size"], shuffle=False)

    # Model
    if config["model_type"] == "transformer":
        model = TransformerEmotionModel(
            input_dim=config["model"]["input_dim"],
            model_dim=config["model"]["model_dim"],
            num_heads=config["model"]["num_heads"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
            task=config["task"]
        )
    elif config["model_type"] == "lstm":
        model = LSTMEmotionModel(
            input_dim=config["model"]["input_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
            task=config["task"]
        )
    else:
        raise ValueError("Invalid model type")
    model.to(device)

    # Loss
    criterion = get_loss(task=config["task"], device=device)

    optimizer = optim.Adam(model.parameters(), lr=config["train"]["lr"])

    # wandb init
    wandb.init(project="ABAW", name=f"{config['task']}_run", config=config)
    wandb.watch(model)

    for epoch in range(config["train"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, max_iter)
        val_loss, val_metric = validate(model, val_loader, criterion, device, config["task"], max_iter)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metric
        })

        print(f"Epoch [{epoch+1}/{config['train']['epochs']}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)