import os
import numpy as np
from sklearn.metrics import f1_score
import torch
import time
from tqdm import tqdm
import torch.optim as optim
from loss import FocalLoss  # Ensure correct import path
from loss import CCCLoss, VALoss, ExprLoss  # Ensure correct import path
from metrics import ExprMetric, VAMetric  # Ensure correct import path
from torch.utils.tensorboard import SummaryWriter
from transformers import ViTImageProcessor, ViTModel  # Ensure correct import path
import pandas as pd
class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg, dir="abaw/", device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.current_epoch = 0
        self.epoch = cfg['epoch']
        self.save_dir = dir
        self.device = device
        self.writer = SummaryWriter(os.path.join(dir, 'logs'))  # Add this line
        self.optimizer, self.criterion, self.scheduler = self._setting_hyperparameters()
        if self.cfg['return_img']:
            self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
            self.feat_extmodel = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)
            self.feat_extmodel.eval()
    def extract_time_series_features(self, image_seq):
        # 특징을 저장할 리스트 초기화
        batch_features = []
        # 데이터 순회
        with torch.no_grad():
            for image in image_seq:
                batch_images = image
                inputs = self.image_processor(images=batch_images, return_tensors="pt", do_rescale=False, do_resize=False).to(self.device)
                outputs = self.feat_extmodel(**inputs)
                features = outputs.pooler_output
                batch_features.append(features.to('cpu').numpy())  # 특징을 CPU로 이동 후 리스트에 추가
        batch_features = np.array(batch_features)
        return batch_features

    def _setting_hyperparameters(self):
        optimizer = self._get_optimizer()
        criterion = self._get_criterion()
        scheduler = self._get_scheduler(optimizer)
        return optimizer, criterion, scheduler

    def _get_optimizer(self):
        if self.cfg['optimizer'] == "Adam":
            return optim.Adam(self.model.parameters(), lr=self.cfg['lr'], weight_decay=self.cfg['decay_rate'] if self.cfg['use_weight_decay'] else 0)
        elif self.cfg['optimizer'] == "AdamW":
            return optim.AdamW(self.model.parameters(), lr=self.cfg['lr'], weight_decay=self.cfg['decay_rate'] if self.cfg['use_weight_decay'] else 0)
        else:  # Default to SGD
            return optim.SGD(self.model.parameters(), lr=self.cfg['lr'], weight_decay=self.cfg['decay_rate'] if self.cfg['use_weight_decay'] else 0, momentum=0.9, nesterov=True)

    def _get_criterion(self):
        if self.cfg['task'] == "VA":
            return VALoss(ccc=self.cfg['ccc'], mae=self.cfg['mae'], mse=self.cfg['mse'])
        elif self.cfg['task'] == "AU":
            if self.cfg['weighted']:
                nSamples = [160261,  67513, 209982, 362236, 538960, 467113, 332788, 38163, 42787, 34746, 851020, 100978]
                weights = [1 - (x / sum(nSamples)) for x in nSamples]
            else:
                weights = np.ones(self.cfg['num_classes'])
            if self.cfg['loss'] == "focal":
                return FocalLoss(weights=weights, device=self.device)
            return ExprLoss(weights, self.device)
        else:  # Default to Expression task
            # nSamples = [341714, 28825, 21363, 25896, 164485, 129065, 56279, 378754]
            if self.cfg['weighted']:
                nSamples = [177375, 16575, 10810, 9079, 95632, 79864, 31631, 165847]
                weights = [1 - (x / sum(nSamples)) for x in nSamples]
            else:
                weights = np.ones(self.cfg['num_classes'])
            if self.cfg['loss'] == "focal":
                return FocalLoss(weights=weights, device=self.device)
            return ExprLoss(weights, self.device)
        
    def _get_scheduler(self, optimizer):
        if self.cfg['use_scheduler']:
            return optim.lr_scheduler.StepLR(optimizer, self.cfg['step_size'], self.cfg['gamma'])
        return None

    def compute_metric(self, outputs, labels):
        outputs, labels = np.array(outputs), np.array(labels)
        if self.cfg['task'] == "VA":
            return VAMetric()(outputs, labels)
        else:
            return ExprMetric()(outputs, labels, self.cfg['num_classes'])
            
        
    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        outs = []
        targets = []
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch} Training')
        for batch_idx, data in enumerate(pbar):
            if self.cfg['return_vis'] and self.cfg['return_aud']:
                input = torch.cat((data['vis_feat'], data['aud_feat']), dim=2).to(self.device)
            elif self.cfg['return_vis']:
                input = data['vis_feat'].to(self.device)
            elif self.cfg['return_aud']:
                input = data['aud_feat'].to(self.device)
            elif self.cfg['return_img']:
                input = data['img'].to(self.device)
                if self.cfg['model'] == "transformer":
                    input = self.extract_time_series_features(input)
                    input = torch.tensor(input).to(self.device)
                    mask = torch.rand(input.size(1)) > 0.5  # 30개 프레임에 대한 마스크 생성
                    # 마스킹될 프레임의 수를 기반으로 새 텐서 생성
                    masked_tensor = torch.zeros(input.size(0), (~mask).sum(), input.size(2)).to(input.device)
                    input[:, ~mask, :] = masked_tensor  # 마스킹된 위치에 새 텐서 할당
            if not self.cfg['multi']:
                if data['anno'].dim() == 3:
                    label = data['anno'][:,-1].to(self.device)
                else:
                    label = data['anno'].to(self.device)
            else:
                label = data['anno'].to(self.device)


            self.optimizer.zero_grad()
            output = self.model(input)
            loss = self.criterion(output.view(-1, output.size(-1)), label.view(-1, label.size(-1)))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            outs.append(output.detach().cpu().numpy())
            targets.append(label.detach().cpu().numpy())
            pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
            
        avg_loss = total_loss / len(self.train_loader)
        outs = np.concatenate(outs)
        targets = np.concatenate(targets)
        metric = self.compute_metric(outs, targets)
        pbar.set_postfix({'loss': avg_loss, 'metric': metric})
        return avg_loss, metric 

    
    def validate_one_epoch(self):
        self.model.eval()
        total_loss = 0
        feature_outputs = {}
        feature_targets = {}
        pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch} Validation')
        
        with torch.no_grad():
            for batch_idx, data in enumerate(pbar):
                # Input 데이터 준비
                input, label = self.prepare_batch(data)
                feature_path = data['Feature_path']
                output = self.model(input)
                loss = self.criterion(output, label)
                
                # 로스 업데이트
                total_loss += loss.item()

                # 결과 및 타겟 저장
                for fp, out, tar in zip(feature_path, output.detach().cpu().numpy(), label.detach().cpu().numpy()):
                    if fp not in feature_outputs:
                        feature_outputs[fp] = []
                        feature_targets[fp] = []
                    feature_outputs[fp].append(out)
                    feature_targets[fp].append(tar)

                # 진행 상태 업데이트
                pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})

        # 중복 결과 합산 및 스무딩
        outs = []
        targets = []
        for fp in feature_outputs:
            outs.append(np.mean(feature_outputs[fp], axis=0))
            targets.append(np.mean(feature_targets[fp], axis=0))

        # 최종 손실 및 메트릭 계산
        avg_loss = total_loss / len(self.val_loader)
        outs = np.array(outs)
        targets = np.array(targets)
        metric = self.compute_metric(outs, targets)

        pbar.set_postfix({'loss': avg_loss, 'metric': metric})
        return avg_loss, metric
    # def validate_one_epoch(self):
    #     self.model.eval()
    #     total_loss = 0
    #     outs = []
    #     targets = []
    #     pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch} Validation')
    #     with torch.no_grad():
    #         for batch_idx, data in enumerate(pbar):
    #             if self.cfg['return_vis'] and self.cfg['return_aud']:
    #                 input = torch.cat((data['vis_feat'], data['aud_feat']), dim=2).to(self.device)
    #             elif self.cfg['return_vis']:
    #                 input = data['vis_feat'].to(self.device)
    #             elif self.cfg['return_aud']:
    #                 input = data['aud_feat'].to(self.device)
    #             elif self.cfg['return_img']:
    #                 input = data['img'].to(self.device)
    #                 if self.cfg['model'] == "transformer":
    #                     input = self.extract_time_series_features(input)
    #                     input = torch.tensor(input).to(self.device)
    #             if not self.cfg['multi']:
    #                 if data['anno'].dim() == 3:
    #                     label = data['anno'][:,-1].to(self.device)
    #                 else:
    #                     label = data['anno'].to(self.device)
    #             else:
    #                 label = data['anno'].to(self.device)
    #             feature_path = data['Feature_path']
    #             output = self.model(input)
    #             loss = self.criterion(output, label)
    #             total_loss += loss.item()
    #             outs.append(output.detach().cpu().numpy())
    #             targets.append(label.detach().cpu().numpy())
    #             pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
    #             # If you have other things to track per batch, do here
    #     avg_loss = total_loss / len(self.val_loader)
    #     outs = np.concatenate(outs)
    #     targets = np.concatenate(targets)
    #     metric = self.compute_metric(outs, targets)
    #     pbar.set_postfix({'loss': avg_loss, 'metric': metric})
    #     return avg_loss, metric
    
    def test(self):
        self.model.eval()
        total_loss = 0
        outs = []
        targets = []
        data_paths = []
        pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch} Validation')
        with torch.no_grad():
            for batch_idx, data in enumerate(pbar):
                if self.cfg['return_vis'] and self.cfg['return_aud']:
                    input = torch.cat((data['vis_feat'], data['aud_feat']), dim=2).to(self.device)
                elif self.cfg['return_vis']:
                    input = data['vis_feat'].to(self.device)
                elif self.cfg['return_aud']:
                    input = data['aud_feat'].to(self.device)
                elif self.cfg['return_img']:
                    input = data['img'].to(self.device)
                output = self.model(input)
                outs.append(output.detach().cpu().numpy())
                data_paths.append(data['Feature_path'])
                # If you have other things to track per batch, do here
        avg_loss = total_loss / len(self.val_loader)
        df = pd.DataFrame({'Feature_path': data_paths, 'Predicted': outs})
        #print(outs)
        outs = np.concatenate(outs)
        #print(outs)
        df.to_csv('result.csv', index=False)
        np.save('result.npy', outs)
        return outs, data_paths
    
    def log_metrics(self, phase, loss, metrics, epoch):
        self.writer.add_scalar(f'Loss/{phase}', loss, epoch)
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'Metric/{phase}_{metric_name}', metric_value, epoch)

    def save_model(self, validation_loss, epoch):
        # Implement model saving logic based on validation loss or other criteria
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'best_model_{epoch}_{validation_loss}.pt'))
        pass

    def train_epochs(self):
        best_val_metirc = 0
        counter = 0
        patience = 3
        for epoch in range(self.epoch):
            self.current_epoch = epoch
            train_loss, train_metric = self.train_one_epoch()
            self.log_metrics('train', train_loss, train_metric, epoch)
            val_loss, val_metric = self.validate_one_epoch()
            self.log_metrics('validation', val_loss, val_metric, epoch)
            print("Train results")
            print(f"loss:{train_loss}, metric:{train_metric}")
            print("Val results")
            print(f"loss:{val_loss}, metric:{val_metric}")
            if self.scheduler:
                self.scheduler.step()
            
            # dict to float
            if self.cfg['task'] == "VA":
                val_metric = val_metric['va_metric']
            else:
                val_metric = val_metric['F1_macro']

            # Early stopping logic
            if val_metric > best_val_metirc:
                best_val_metirc = val_metric
                counter = 0  # Reset counter if validation loss improves
                # Save best model if validation loss improves
                self.save_model(val_loss, epoch)
            else:
                counter += 1  # Increment counter if validation loss does not improve
            if counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break  # Break out of the loop if early stopping condition is met
            
        self.writer.close()
    def train_decoder(self, learning_rate=0.001):
        best_val_metirc = 0
        counter = 0
        patience = 3
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        # Example data loader (replace with actual data)

        # Training loop

        for epoch in range(self.epoch):
            total_val_loss = 0
            id  = 0
            self.model.train()
            for data in self.train_loader:
                
                src = data["vis_feat"].to(self.device)  # Source data
                tgt = data["vis_feat"].to(self.device)  # Target data, in actual use, this could be shifted versions for forecasting
                
                # Forward pass
                output = self.model(src, tgt)
                
                # Compute loss
                loss = criterion(output, tgt)  # Measure how well we are predicting the future
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                
            for data in self.val_loader:
                src = data["vis_feat"].to(self.device)
                tgt = data["vis_feat"].to(self.device)
                output = self.model(src, tgt)
                loss = criterion(output, tgt)
                val_loss = loss.item()
                total_val_loss += val_loss
                id += 1
            val_loss = total_val_loss / id
            if val_loss > best_val_metirc:
                best_val_metirc = val_loss
                counter = 0  # Reset counter if validation loss improves
                # Save best model if validation loss improves
                self.save_model(val_loss, epoch)
            else:
                counter += 1  # Increment counter if validation loss does not improve
            if counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break  # Break out of the loop if early stopping condition is met
        print(f'Epoch [{epoch+1}/{self.epoch}], Loss: {loss.item():.4f}')