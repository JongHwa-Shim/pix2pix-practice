import os
import torch
from torchvision import transforms
from PIL import Image
import csv
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler

def PreProcessing(source_path, target_path=None, mode=None):
    sources = []
    targets = []

    if mode==None:
        print("please select mode")

    elif mode=="csv":
        with open(source_path) as f:
            rdr = csv.reader(f)
            next(rdr)
            for line in rdr:
                
                target = line[0]
                source = line[1:]
                target = [int(num) for num in target]
                source = [int(num) for num in source]
                targets.append(target)
                sources.append(source)
            
    elif mode=='individual':
        source_list = os.listdir(source_path)

        for source_name in source_list:
            file_path = source_path + source_name
            source = Image.open(file_path)
            sources.append(source)

        target_list = os.listdir(target_path)

        for target_name in target_list:
            file_path = target_path + target_name
            target = 3
            targets.append(target)

    elif mode=='other':
        None

    return sources, targets

