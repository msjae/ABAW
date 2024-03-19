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
                np_files = np.load(os.path.join(self.dataset_root, file_name + '.npy'))
                frame_name = np_files[:, 0]
                for idx, anno in enumerate(anno_lines):
                    if np.where(frame_name == idx + 1)[0].size != 0:
                        valid_data, processed_anno = self._validate_and_process_anno(anno)
                        if valid_data:
                            anno_data.append([processed_anno, file_name ,idx + 1])
        return anno_data
        # + annodata, folder + frame number(=idx)

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
        processed_anno, file_name, idx = self.anno_data[index]
        # print(file_name, idx)
        vis_feat = np.load(os.path.join(self.dataset_root, file_name + '.npy'))
        frame_name = vis_feat[:, 0]
        vis_feat = vis_feat[:, 1:]
        frame_num = np.where(frame_name == idx)[0][0]
        if self.window_size == 1:
            # np.load feature_folder
            # current idx + window size - 1
            data['vis_feat'] = vis_feat[frame_num]
            data['anno_folder_frame'] = np.array([processed_anno], dtype=np.float32)  # Single element in array for consistency
        else:
            x = 0 if frame_num - self.window_size + 1  < 0 else frame_num - self.window_size + 1
            data["vis_feat"] = vis_feat[x : frame_num + 1]
            if data["vis_feat"].shape[0] < self.window_size:
                feat_num = data["vis_feat"].shape[0]
                last_layer = data["vis_feat"][-1].reshape(1, 768)
                expanded_last_layer = np.tile(last_layer, (self.window_size - feat_num, 1))
                data["vis_feat"] = np.concatenate((data["vis_feat"], expanded_last_layer), axis=0)
            data['anno'] = np.array(processed_anno, dtype=np.float32)  # Single element in array for consistency
            # print(data["vis_feat"].shape, frame_name[frame_num], frame_num)
        return data

    def __len__(self):
        return len(self.anno_data)

from utils import get_config
from torch.utils.data import DataLoader

def main():
    cfg = get_config('configs/config.yaml')
    data_path = "/media/minseongjae/HDD/data/AffWild2/Features_folder_frame"
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
