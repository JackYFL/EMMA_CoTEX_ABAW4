import pathlib
import numpy as np
import math
import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
import scipy.io as sio
import os
import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ABAWMTDataset(Dataset):
    def __init__(self, annotation_path, test=False, transform=None, data_root='/data/yfli/AffWild2/cropped_aligned/'):
        self.data_root = data_root
        self.test = test
        if not test:
            self.img_names, self.labels_list = self.read_file(annotation_path)
        else:
            self.img_names = self.read_file(annotation_path)
        self.transform = transform
        self.img_num = len(self.img_names)
        
    def __getitem__(self, index):
        if not self.test:
            label = np.array(self.labels_list[index])
            image_path = self.data_root + self.img_names[index]
            img = Image.open(image_path)
            img = self.transform(img)
            return img, label
        else:
            img_name = self.img_names[index]
            image_path = self.data_root + img_name
            img = Image.open(image_path)
            img = self.transform(img)
            return img, img_name
        
    def __len__(self):
        data_len = len(self.img_names)
        return data_len

    def read_file(self, filename):
        if not self.test:
            img_names, labels_list = [], []
            with open(filename, 'r') as f:
                for i, line in tqdm.tqdm(enumerate(f.readlines())):
                    if i != 0:
                        line_splits = line.split(',')
                        img_path = line_splits[0]
                        if os.path.exists(self.data_root + '/' + img_path):
                            labels = [float(entry) for entry in line_splits[1:]]
                            if labels[0]==-5 or labels[1]==-5:
                                pass
                            img_names.append(img_path)
                            labels_list.append(labels)
                            
            print('Data has been loaded!!!')
            return img_names, labels_list
        else:
            img_names = []
            with open(filename, 'r') as f:
                for i, line in tqdm.tqdm(enumerate(f.readlines())):
                    if i != 0:
                        line_splits = line.split(',')
                        img_path = line_splits[0].strip('\n')
                        path = self.data_root + '/' + img_path
                        if os.path.exists(path):
                            try:
                                img = Image.open(path)
                                img_names.append(img_path)
                            except:
                                print(img_path)
                            
            print('Data has been loaded!!!')
            return img_names
