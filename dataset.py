import os
import numpy as np
import glob
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, dataset_root, anno_path, task, split, num_classes, window_size=32, stride=16):
        """
        Initialize the dataset.
        
        Args:
            dataset_root (str): The root directory of the dataset.
            anno_path (str): The directory containing the annotation files.
            task (str): The type of task ('VA', 'EXPR', or 'AU').
            split (str): The dataset split ('train', 'val', or 'test').
            num_classes (int): The number of classes for the task.
            window_size (int, optional): The size of the sliding window for sequences. Defaults to 32.
            stride (int, optional): The stride of the sliding window. Defaults to 16.
        """
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
        #self.sequence_data = self._process_sequence()

    def _process_annotations(self):
        """
        Process all annotation files to extract and validate annotations.
        
        Returns:
            List of tuples containing (processed_annotation, corresponding_npy_file).
        """
        anno_data = []
        for anno_file in self.anno_list:
            file_name = os.path.basename(anno_file)[:-4]
            with open(anno_file, 'r') as file:
                anno_lines = file.readlines()[1:]
                for idx, anno in enumerate(anno_lines):
                    valid_data, processed_anno = self._validate_and_process_anno(anno)
                    if valid_data:
                        anno_data.append([processed_anno, file_name ,idx])
        return anno_data
        # + annodata, folder + frame number(=idx)
    '''def _process_sequence(self):
        """
        Process the annotations to create a list of valid sequences based on the window size and stride.
        
        Returns:
            List of valid sequences, where each sequence is a list of (annotation, file path) pairs.
        """
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

        return sequence_data'''

    def _validate_and_process_anno(self, anno):
        """
        Validate and process a single annotation line based on the task.
        
        Args:
            anno (str): The annotation line as a string.
            
        Returns:
            tuple: (valid_data, processed_anno) where valid_data is a boolean indicating 
                   whether the annotation is valid, and processed_anno is the processed annotation.
        """
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
        """
        Retrieve a dataset item by index.
        
        Args:
            index (int): The index of the item.
        
        Returns:
            dict: A dictionary containing the visual features and the annotation.
        """
        data = {}
        if self.window_size == 1:
            processed_anno, file_name, idx = self.anno_data[index]
            # np.load feature_folder
            # current idx + window size - 1
            data['vis_feat'] = np.load(os.path.join(self.dataset_root, file_name + '.npy'))[idx]
            data['anno'] = np.array([processed_anno], dtype=np.float32)  # Single element in array for consistency
        else:
            processed_anno, file_name, idx = self.anno_data[index]
            if idx >= self.window_size - 1:
                data["vis_feat"] = np.load(os.path.join(self.dataset_root, file_name + '.npy'))[idx - self.window_size + 1: idx + 1]
                # data["anno"] = np.array([x[0] for x in self.anno_data[index - self.window_size + 1: index + 1]], dtype=np.float32)
                print(processed_anno, file_name, data["vis_feat"].shape, idx)
            else:
                vis_feat = np.load(os.path.join(self.dataset_root, file_name + '.npy'))[:idx + 1]
                feat_num = vis_feat.shape[0]
                last_layer = vis_feat[-1].reshape(1, 768)
                expanded_last_layer = np.tile(last_layer, (self.window_size - feat_num, 1))
                vis_feat = np.concatenate((vis_feat, expanded_last_layer), axis=0)
                data["vis_feat"] = vis_feat
                # anno = np.array([x[0] for x in self.anno_data[:index + 1]], dtype=np.float32)
                # last_anno = anno[-1].reshape(1, anno.shape[1])
                # expanded_last_anno = np.tile(last_anno, (self.window_size - feat_num, 1))
                # data["anno"] = np.concatenate((anno, expanded_last_anno), axis=0)
            data['anno'] = np.array(processed_anno, dtype=np.float32)  # Single element in array for consistency
        return data

    def __len__(self):
        return len(self.anno_data)

from utils import get_config
from torch.utils.data import DataLoader

def main():
    cfg = get_config('configs/config.yaml')
    data_path = "/media/minseongjae/HDD/data/AffWild2/Features_folder"
    anno_path = "/media/minseongjae/HDD/data/AffWild2/6th ABAW Annotations"
    task = cfg['task']
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
