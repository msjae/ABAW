from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch

class ImageFeatureDataset(Dataset):
    task_folder = {
        "expr": "EXPR_Recognition_Challenge",
        "va": "VA_Estimation_Challenge",
        "au": "AU_Detection_Challenge"
    }
    split_folder = {
        "train": "Train_Set",
        "val": "Validation_Set",
    }

    def __init__(self, feature_root, label_root, task="expr", split="train", seq_len=30, stride=1):
        self.task = task
        self.split = split
        self.seq_len = seq_len
        self.stride = stride

        self.feature_root = feature_root
        self.label_root = os.path.join(label_root,
            self.task_folder[task],
            self.split_folder[split.lower()]
        )

        self.samples = []
        self._build_index()

    def _build_index(self):
        for label_file in sorted(os.listdir(self.label_root)):
            if not label_file.endswith(".txt"):
                continue
            video = os.path.splitext(label_file)[0]
            feature_dir = os.path.join(self.feature_root, video)
            label_path = os.path.join(self.label_root, label_file)

            if not os.path.isdir(feature_dir) or not os.path.exists(label_path):
                continue

            feature_files = sorted([
                f for f in os.listdir(feature_dir) if f.endswith(".npy")
            ])
            available_frames = set(int(f.split('_')[-1].split('.')[0]) for f in feature_files)

            with open(label_path, "r") as f:
                header = f.readline()
                lines = f.readlines()

            delimiter = ',' if self.task in ("va", "au") else None
            labels = [list(map(float, line.strip().split(delimiter))) for line in lines]

            for start in range(0, len(labels) - self.seq_len + 1, self.stride):
                valid = True
                for offset in range(self.seq_len):
                    idx = start + offset
                    if idx + 1 not in available_frames:
                        valid = False
                        break
                    if self._is_invalid(labels[idx]):
                        valid = False
                        break
                if valid:
                    paths = [os.path.join(feature_dir, f"{start + offset + 1:05d}.npy") for offset in range(self.seq_len)]
                    if self.task == "expr":
                        label = torch.tensor(labels[start + self.seq_len - 1][0], dtype=torch.long)
                    elif self.task == "va":
                        label = torch.tensor(labels[start + self.seq_len - 1][:2], dtype=torch.float32)
                    elif self.task == "au":
                        label = torch.tensor(labels[start + self.seq_len - 1], dtype=torch.float32)
                    self.samples.append((paths, label))

    def _is_invalid(self, label):
        if self.task == "expr":
            return label[0] == -1
        elif self.task == "va":
            return label[0] == -5 or label[1] == -5
        elif self.task == "au":
            return any(l == -1 for l in label)
        return True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, label = self.samples[idx]
        feature_seq = [torch.from_numpy(np.load(p)).float() for p in paths]
        feature_seq = torch.cat(feature_seq, dim=0)  # [seq_len, 768]
        return feature_seq, label

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # feature_root = "data/features/image_mean"
    label_root = 'data/labels'
    split = "val"  # train or val
    feature_root = "data/features/features_mean"
    
    for task in ["expr", "va", "au"]:
        print(f"\n[INFO] Loading task: {task}")
        dataset = ImageFeatureDataset(
            feature_root=feature_root,
            label_root=label_root,
            task=task,
            split=split,
            seq_len=30,
            stride=1
        )
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
        print(f"[INFO] Number of samples for task '{task}': {len(dataset)}")
    
        for features, labels in dataloader:
            print(f"[INFO] Task '{task}' Feature batch shape: {features.shape}")
            print(f"[INFO] Task '{task}' Label batch shape: {labels.shape}")
            break