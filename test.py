import torch
import torch.nn as nn
from torch.utils.data import Dataset
from make_dataset import *

def test_preprocessing():
    None

class test_transform(self_transform):
    def __init__(self):
        super(self,test_transform)
    
    def __call__():
        None

class test_dataset(Dataset):
    def __init__(self, sources, conditions, transfrom=None):
        self.sources = sources
        self.conditions = conditions
        self.transform = transform
    
    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):

        sample = {}

        if self.transform:
            sample = self.transform(sample)

#틀만 잡아놓음


