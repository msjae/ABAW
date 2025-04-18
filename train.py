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

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for features, labels in tqdm(dataloader, desc="Training"):
        features, labels = features.to(device), labels.to(device)

        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
    return running_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validation"):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * features.size(0)
    return running_loss / len(dataloader.dataset)

def main(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    if config["task"] == "expr":
        criterion = nn.CrossEntropyLoss()
    elif config["task"] == "va":
        criterion = nn.MSELoss()
    elif config["task"] == "au":
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=config["train"]["lr"])

    # wandb init
    wandb.init(project="ABAW", config=config)
    wandb.watch(model)

    for epoch in range(config["train"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        print(f"Epoch [{epoch+1}/{config['train']['epochs']}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)