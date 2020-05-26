import random
import matplotlib.pyplot as plt
from matplotlib import image
import torch
import os

from preprocessing import PreProcessing
from make_dataset import transform_processing
from postprocessing import visualization
def pixel_analysis(path):
    img = image.imread(path)
    plt.imshow(img)
    plt.show()

#pixel_analysis("./data/pix2pix-dataset/edges2shoes/edges2shoes/train/1_AB.jpg")

def sample_condition_generate(data_path, num, condition_process, device):
    fixed_condition_list = []

    data_list = os.listdir(data_path)
    sample_list = random.sample(data_list, num)
    for sample in sample_list:
        fixed_condition = data_path + '/' + sample

        for process in condition_process:
            fixed_condition = process(fixed_condition)
        fixed_condition_list.append(torch.unsqueeze(fixed_condition,0))
    
    fixed_condition = torch.cat(fixed_condition_list, dim=0).to(device)

    return fixed_condition
"""
data_path =  r'C:/USING_DATA/pix2pix-dataset/edges2shoes/edges2shoes/val'
filter = transform_processing(real_mode='jpg', condition_mode='jpg')
condition_process = [filter.first_processing_condition, filter.to_FloatTensor, filter.Scaling, filter.final_processing]
fixed_condition = sample_condition_generate(data_path, 36, condition_process, torch.device("cuda:0"))
a=1
"""