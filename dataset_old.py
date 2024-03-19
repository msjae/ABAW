import os
import numpy as np
import glob
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, dataset_root, anno_path, task, split, num_classes, window_size=1, stride=1):
        self.dataset_root = dataset_root
        self.task = task
        self.split = split
        self.num_classes = num_classes
        if self.task == 'VA':
            self.anno_path = os.path.join(anno_path, f"VA_Estimation_Challenge", f"{self.split}_Set")
        elif self.task == 'EXPR':
            self.anno_path = os.path.join(anno_path, f"EXPR_Recognition_Challenge", f"{self.split}_Set")
        else:
            self.anno_path = os.path.join(anno_path, f"AU_Detection_Challenge", f"{self.split}_Set")
        self.anno_list = glob.glob(os.path.join(self.anno_path, "*.txt"))
        self.anno_data = self._process_annotations()
        self.window_size = window_size
        self.stride = stride
        self.sequence_data = self._process_sequence()

    def _process_annotations(self):
        anno_data = []
        for anno_file in self.anno_list:
            file_name = os.path.basename(anno_file)[:-4]
            np_files = sorted(glob.glob(os.path.join(self.dataset_root, file_name, "*.npy")))
            with open(anno_file, 'r') as file:
                anno_lines = file.readlines()[1:]
                for anno, npy in zip(anno_lines, np_files):
                    valid_data, processed_anno = self._validate_and_process_anno(anno)
                    if valid_data:
                        anno_data.append([processed_anno, npy])
        return anno_data
    
    def _process_sequence(self):
        # 숫자 부분 추출
        sequence_numbers = np.array([int(path[1].split("/")[-1][:-4]) for path in self.anno_data])

        # 시작과 끝 시퀀스 번호 계산
        start_indices = np.arange(0, len(self.anno_data) - self.window_size + 1, self.stride)
        end_indices = start_indices + self.window_size - 1
        
        # 유효한 시퀀스 필터링
        valid_indices = np.where(sequence_numbers[end_indices] >= sequence_numbers[start_indices])[0]
        valid_start_indices = start_indices[valid_indices]

        # 유효한 시퀀스 데이터 생성
        sequence_data = [self.anno_data[i: i + self.window_size] for i in valid_start_indices]

        return sequence_data

    def _validate_and_process_anno(self, anno):
        if self.task == 'AU':
            anno = [int(i) for i in anno.split(',')]
            valid_data = all(num >= 0 for num in anno)
        elif self.task == 'EXPR':
            try:
                num = int(anno)
                anno = [0.0] * self.num_classes
                if num >= 0:
                    anno[num] = 1.0
                    valid_data = True
                else:
                    valid_data = False
            except ValueError:
                valid_data = False
        else:  # VA Task
            anno = [float(i) for i in anno.split(',')]
            valid_data = all(-1 <= num <= 1 for num in anno)
        return valid_data, anno

    def __getitem__(self, index):
        data = {}
        if self.window_size == 1:
            anno, feat = self.anno_data[index]
            data['vis_feat'] = np.load(feat)
            data['anno'] = np.array([anno], dtype=np.float32)  # Single element in array for consistency
        else:
            sequence = self.sequence_data[index]
            # np.stack or np.array can be used to convert list of arrays into a 3D numpy array
            data['vis_feat'] = np.array([np.load(seq[1]) for seq in sequence])
            data['anno'] = np.array(sequence[-1][0], dtype=np.float32)  # Single element in array for consistency
        return data

    def __len__(self):
        return len(self.sequence_data)

from utils import get_config
from torch.utils.data import DataLoader

def main():
    cfg = get_config('configs/config.yaml')
    data_path = "/media/minseongjae/HDD/data/AffWild2/batch1/Features"
    anno_path = "/media/minseongjae/HDD/data/AffWild2/6th ABAW Annotations"
    task = 'EXPR'
    if task == 'VA':
        num_classes = 2
    elif task == 'EXPR':
        num_classes = 8
    elif task == 'AU':
        num_classes = 12

    train_set = MyDataset(dataset_root=data_path, anno_path=anno_path, task=task, split='Train', num_classes=num_classes)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    
    for data in train_loader:
        print(data['vis_feat'].shape)
        print(data['anno'].shape)
        break  # Remove or modify for full dataset iteration

if __name__ == '__main__':
    main()
