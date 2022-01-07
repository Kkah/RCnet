import os
import numpy as np
import cv2
import torch
from torch.utils import data

class RC_Dataset(data.Dataset): 
    def __init__(self, dataset, transform, args):
        self.image = dataset.drop(['target'], axis = 1)
        self.target = dataset['target']
        self.transform = transform
        self.mode = False
        self.seq_len = args.seq_len
        self.target_len = args.target_len
        
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        image_seq = []

        for frame in self.image.iloc[idx]:
            image = cv2.imread(frame)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            image = self.transform(image = image)['image']
            image_seq.append(image)
            
        image = torch.stack(image_seq)
        target = self.target.iloc[idx]
        
        return image, target
    
