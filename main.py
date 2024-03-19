from utils import get_config
from transformer_model import TransformerModel, TransformerModel_decoder
from model import linear_model, lstm_model, resnet_model, ImageFeatureTransformer, ImageTransformer
# from dataset_yjs import MyDataset
from dataset_msj import MyDataset
from torch.utils.data import DataLoader
from trainer_yjs import Trainer
import torch
import torch.nn as nn
import os
import argparse
import yaml
import torch



    
def main():
    # Parse command line arguments
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
    # Initialize the model
    if cfg['model'] == 'transformer':
        model = TransformerModel(seq_len=cfg['seq_size'], embedding_size=cfg['embedding_size'],
                                    nhead=cfg['n_head'], num_encoder_layers=cfg['n_layers'],
                                    num_classes=cfg['num_classes'], cfg=cfg)
    elif cfg['model'] == 'transformer_decoder':
        model = TransformerModel_decoder(seq_len=cfg['seq_size'], nhead=cfg['n_head'], 
                                         embedding_size=cfg['embedding_size'], num_encoder_layers=cfg['n_layers'],
                                         num_decoder_layers=cfg['n_layers'], cfg=cfg, feature_dim=cfg['feature_dim'])
    elif cfg['model'] == 'linear':
        # Linear model
        model = linear_model(cfg)
    elif cfg['model'] == 'lstm':
        # LSTM model
        model = lstm_model(cfg)
    elif cfg['model'] == 'resnet':
        # ResNet model
        model = resnet_model(cfg)
    elif cfg['model'] == 'vit':
        model = ImageTransformer('google/vit-base-patch16-224-in21k', cfg, device=cfg['device'])
    else:
        # Vision Transformer
        model = ImageFeatureTransformer('google/vit-base-patch16-224-in21k', cfg, device=cfg['device'])
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
                              return_seq=cfg['return_seq'], seq_size=cfg['seq_size'], aug=cfg['augmentation'], p=cfg['p'])
    val_dataset = MyDataset(data_root = cfg['data_path'], csv_file=val_anno_path, 
                              return_img=cfg['return_img'], return_aud=cfg['return_aud'], return_vis=cfg['return_vis'],
                              return_seq=cfg['return_seq'], return_mask=False, seq_size=cfg['seq_size'])

    # Initialize the data loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

    # # Initialize the trainer
    # if os.path.exists(cfg['dir']):
    #     dir = cfg['dir']+
        
    trainer = Trainer(model, train_loader, val_loader, cfg, dir=dir, device=cfg['device'])
    trainer.train_epochs()
    #trainer.train_decoder()
    

if __name__ == '__main__':
    main()