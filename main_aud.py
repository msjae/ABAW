from utils import get_config
from transformer_model import TransformerModel
# from dataset_yjs import MyDataset
from dataset_msj import MyDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import os
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    # Load the configuration
    cfg = get_config(args.config_path)  # Update this path as necessary
    # save the config
    torch.save(cfg, cfg['dir'] + 'config.pt')
    # Initialize the model
    model = TransformerModel(cfg)

    # Initialize the dataset
    ## train_dataset = MyDataset(cfg['data_path'], cfg['anno_path'], cfg['task'], 'Train', cfg['num_classes'], cfg['window_size'], cfg['stride'])
    ## val_dataset = MyDataset(cfg['data_path'], cfg['anno_path'], cfg['task'], 'Validation', cfg['num_classes'], cfg['window_size'], cfg['stride'])
    if cfg['task'] == 'VA':
        train_anno_path = os.path.join(cfg['anno_path'], f"VA_Estimation_Challenge_Train.csv")
        val_anno_path = os.path.join(cfg['anno_path'], f"VA_Estimation_Challenge_Validation.csv")
    elif cfg['task'] == 'EXPR':
        train_anno_path = os.path.join(cfg['anno_path'], f"EXPR_Recognition_Challenge_Train.csv")
        val_anno_path = os.path.join(cfg['anno_path'], f"EXPR_Recognition_Challenge_Validation.csv")
    else:
        train_anno_path = os.path.join(cfg['anno_path'], f"AU_Detection_Challenge_Train.csv")
        val_anno_path = os.path.join(cfg['anno_path'], f"AU_Detection_Challenge_Validation.csv")

    train_dataset = MyDataset(data_root = "/home/minseongjae/ABAW_Audio/2022_ABAW_imlab/data/features/train_audio", csv_file=train_anno_path)
    val_dataset = MyDataset(data_root = "/home/minseongjae/ABAW_Audio/2022_ABAW_imlab/data/features/val_audio", csv_file=val_anno_path)

    # Initialize the data loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

    # Initialize the trainer
    trainer = Trainer(model, train_loader, val_loader, cfg, dir=cfg['dir'], device=cfg['device'])
    trainer.train_epochs()

if __name__ == '__main__':
    main()