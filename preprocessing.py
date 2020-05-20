import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import image
import csv
import torch
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

    elif mode=='jpg':
        root_path = source_path
        data_list = os.listdir(source_path)
        for file_name in data_list:
            file_path = root_path + '/' + file_name
            img = image.imread(file_path)

            height = img.shape[0]
            width = img.shape[1]

            source = img[:,0:int(width/2)]
            source_r = source[:,:,0:1]
            source_g = source[:,:,1:2]
            source_b = source[:,:,2:3]

            source = source_r/3 + source_g/3 + source_b/3 # convert to gray scale
            sources.append(source)

            target = img[:,int(width/2):]
            targets.append(target)

    return sources, targets
    # source = condition, target = real

