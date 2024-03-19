import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from PIL import Image
import cv2
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MyDataset(Dataset):
    def __init__(self, data_root, csv_file, return_vis=True, return_aud=False, return_img=False, return_seq=False, aug=False, return_mask=True, seq_size=10, p =0.5):
        self.data_root = data_root
        self.data_frame = pd.read_csv(csv_file, low_memory=False)
        self.seq_size = seq_size
        self.p = p
        if self.data_frame['Annotation'].dtype == 'int64':
            self.annotations = [np.eye(8)[x] for x in self.data_frame['Annotation']]
        else:
            self.annotations = [np.array(x.strip('[]').split(', '), dtype=float) for x in self.data_frame['Annotation']]
        
        self.features_paths = self.data_frame['Feature_path'].values
        self.directories = self.data_frame['Directory'].values

        self.return_seq = return_seq
        self.return_vis = return_vis
        self.return_aud = return_aud
        self.return_img = return_img
        self.return_mask = return_mask
        if return_aud:
            self.features_paths = [x for x in self.features_paths if os.path.exists(os.path.join(data_root, 'aud_feat', x))]
        if aug:
            self.transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.HorizontalFlip(p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
                A.Resize(224, 224),
                A.Perspective(scale=(0.05, 0.1), p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, always_apply=True),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, always_apply=True),
                ToTensorV2()
            ])
    def __len__(self):
        return len(self.features_paths)
    
    def __getitem__(self, idx):
        data = {}
        if self.return_seq:
            start_idx = max(idx - self.seq_size + 1, 0)
            sequence_range = range(start_idx, idx + 1)
            current_dir = self.directories[idx]
            # if len(sequence_range) < self.seq_size:
            #     sequence_range = [start_idx] * (self.seq_size - len(sequence_range)) + list(sequence_range)
            idxs = []
            for i in sequence_range:
                if self.directories[i] != current_dir:
                    # Repeat the first valid entry to fill the sequence if the directory changes
                    continue
                else:
                    idxs.append(i)

            if len(idxs) < self.seq_size:
                idxs = [idxs[0]] * (self.seq_size - len(idxs)) + idxs

            # Adjust for the sequence of feature paths
            feature_paths_seq = [self.features_paths[i] for i in idxs]
            annotations_seq = np.array([self.annotations[i] for i in idxs])
            data['anno'] = torch.tensor(annotations_seq, dtype=torch.float)
            
            if self.return_vis:
                vis_feat_seq = np.concatenate([np.load(os.path.join(self.data_root, 'vis_feat', fp)) for fp in feature_paths_seq], axis=0)
                data['vis_feat'] = torch.tensor(vis_feat_seq, dtype=torch.float)
                # masked some data for transformer
                if self.return_mask:
                    mask = np.random.choice([0, 1], size=vis_feat_seq.shape, p=[self.p, 1-self.p])
                    data['vis_feat'] = torch.tensor(vis_feat_seq * mask, dtype=torch.float)
            if self.return_aud:
                aud_feat_seq = np.concatenate([np.load(os.path.join(self.data_root, 'aud_feat', fp)) for fp in feature_paths_seq], axis=0)
                data['aud_feat'] = torch.tensor(aud_feat_seq, dtype=torch.float)
            if self.return_img:
                img_seq = [self.transform(image=cv2.cvtColor(cv2.imread(os.path.join(self.data_root, 'imgs', fp.replace('.npy','.jpg'))), cv2.COLOR_BGR2RGB))['image'] for fp in feature_paths_seq]
                data['img'] = torch.stack(img_seq)

        else:
            annotations = self.annotations[idx]
            feature_path = self.features_paths[idx]
            
            data['Feature_path'] = feature_path
            data['anno'] = torch.tensor(annotations, dtype=torch.float)
            
            if self.return_vis:
                vis_feat = np.load(os.path.join(self.data_root, 'vis_feat', feature_path))
                data['vis_feat'] = torch.tensor(vis_feat, dtype=torch.float)
            if self.return_aud:
                aud_feat = np.load(os.path.join(self.data_root, 'aud_feat', feature_path))
                data['aud_feat'] = torch.tensor(aud_feat, dtype=torch.float)
            if self.return_img:
                img_path = os.path.join(self.data_root, 'imgs', feature_path.replace('.npy','.jpg'))
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                img = self.transform(image=img)['image']
                ## PIL
                # img = Image.open(os.path.join(self.data_root, 'imgs', feature_path.replace('.npy','.jpg'))).convert('RGB')
                # img = self.transform(img)
                data['img'] = img
        return data

class MyDataset_test(Dataset):
    def __init__(self, data_root, csv_file='/home/minseongjae/ABAW/0_data/test/test_set_examples/test_set_examples/CVPR_6th_ABAW_Expr_test_set_example.txt', return_vis=True, return_aud=False, return_img=False, return_seq=False, seq_size=10):
        self.data_root = data_root
        self.data_frame = pd.read_csv(csv_file, low_memory=False)
        self.seq_size = seq_size
        self.features_paths = self.data_frame['Feature_path'].values
        self.directories = self.data_frame['Directory'].values

        self.return_seq = return_seq
        self.return_vis = return_vis
        self.return_aud = return_aud
        self.return_img = return_img
        if return_aud:
            self.features_paths = [x for x in self.features_paths if os.path.exists(os.path.join(data_root, 'aud_feat', x))]
        if return_img:            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])        

    def __len__(self):
        return len(self.features_paths)
    
    def __getitem__(self, idx):
        data = {}
        if self.return_seq:
            start_idx = max(idx - self.seq_size + 1, 0)
            sequence_range = range(start_idx, idx + 1)
            current_dir = self.directories[idx]
            # if len(sequence_range) < self.seq_size:
            #     sequence_range = [start_idx] * (self.seq_size - len(sequence_range)) + list(sequence_range)
            idxs = []
            for i in sequence_range:
                if self.directories[i] != current_dir:
                    # Repeat the first valid entry to fill the sequence if the directory changes
                    continue
                else:
                    idxs.append(i)

            if len(idxs) < self.seq_size:
                idxs = [idxs[0]] * (self.seq_size - len(idxs)) + idxs

            # Adjust for the sequence of feature paths
            feature_paths_seq = [self.features_paths[i] for i in idxs]
            annotations_seq = np.array([self.annotations[i] for i in idxs])
            data['anno'] = torch.tensor(annotations_seq, dtype=torch.float)
            
            if self.return_vis:
                vis_feat_seq = np.concatenate([np.load(os.path.join(self.data_root, 'vis_feat', fp)) for fp in feature_paths_seq], axis=0)
                data['vis_feat'] = torch.tensor(vis_feat_seq, dtype=torch.float)
                
            if self.return_aud:
                aud_feat_seq = np.concatenate([np.load(os.path.join(self.data_root, 'aud_feat', fp)) for fp in feature_paths_seq], axis=0)
                data['aud_feat'] = torch.tensor(aud_feat_seq, dtype=torch.float)
            if self.return_img:
                img_seq = [self.transform(image=cv2.cvtColor(cv2.imread(os.path.join(self.data_root, 'imgs', fp.replace('.npy','.jpg'))), cv2.COLOR_BGR2RGB))['image'] for fp in feature_paths_seq]
                data['img'] = torch.stack(img_seq)

        else:
            annotations = self.annotations[idx]
            feature_path = self.features_paths[idx]
            
            data['Feature_path'] = feature_path
            data['anno'] = torch.tensor(annotations, dtype=torch.float)
            
            if self.return_vis:
                vis_feat = np.load(os.path.join(self.data_root, 'vis_feat', feature_path))
                data['vis_feat'] = torch.tensor(vis_feat, dtype=torch.float)
            if self.return_aud:
                aud_feat = np.load(os.path.join(self.data_root, 'aud_feat', feature_path))
                data['aud_feat'] = torch.tensor(aud_feat, dtype=torch.float)
            if self.return_img:
                img_path = os.path.join(self.data_root, 'imgs', feature_path.replace('.npy','.jpg'))
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                img = self.transform(image=img)['image']
                ## PIL
                # img = Image.open(os.path.join(self.data_root, 'imgs', feature_path.replace('.npy','.jpg'))).convert('RGB')
                # img = self.transform(img)
                data['img'] = img
        return data
    
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
def main():
    # csv
    task = "AU"
    split = "Train"
    data_path = "./0_data"
    anno_path = "/media/minseongjae/HDD/data/AffWild2/6th ABAW Annotations"
    if task == 'VA':
        anno_path = os.path.join(anno_path, f"VA_Estimation_Challenge_{split}.csv")
    elif task == 'EXPR':
        anno_path = os.path.join(anno_path, f"EXPR_Recognition_Challenge_{split}.csv")
    else:
        anno_path = os.path.join(anno_path, f"AU_Detection_Challenge_{split}.csv")
    print(anno_path)
    # dataset
    dataset = MyDataset(data_root = data_path, csv_file=anno_path, return_img=True, return_aud=False, return_vis=True, return_seq=False)
    dataset_seq = MyDataset(data_root = data_path, csv_file=anno_path, return_img=True, return_aud=False, return_vis=True, return_seq=True)
    
    # dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    dataloader_seq = DataLoader(dataset_seq, batch_size=4, shuffle=False, num_workers=4)
    
    # class 비율 확인 정렬
    print(dataset.data_frame['Annotation'].value_counts().sort_index())
    print(dataset.annotations)
    # time check
    print(f"dataset length: {len(dataset)}")
    start = time.time()
    for i_batch, sample_batched in enumerate(tqdm(dataloader)):
        print(i_batch, sample_batched['img'].size(),
              sample_batched['vis_feat'].size(),
            #   sample_batched['aud_feat'].size(),
              sample_batched['anno'].size(),
              sample_batched['anno'][:,-1].size())
        if i_batch == 3:
            break
        # if i_batch == 1000:
        #     break
        pass
    print(f"elapsed time: {time.time() - start}")

    print(f"dataset_seq length: {len(dataset_seq)}")
    start = time.time()
    for i_batch, sample_batched in enumerate(tqdm(dataloader_seq)):
        print(i_batch, sample_batched['img'].size(),
              sample_batched['vis_feat'].size(),
            #   sample_batched['aud_feat'].size(),
              sample_batched['anno'].size(),
              sample_batched['anno'][:,-1].size())
        if i_batch == 3:
            break
        # if i_batch == 1000:
        #     break
        pass
    print(f"elapsed time: {time.time() - start}")
if __name__ == '__main__':
    main()