#len(dataset) 에서 호출되는 __len__ 은 데이터셋의 크기를 리턴
#dataset[i] 에서 호출되는 __getitem__ 은 i번째 샘플을 찾는데 사용
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from matplotlib import image
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class transform_processing(object):
    def __init__(self, real_mode=None, condition_mode=None):
        self.real_mode = real_mode
        self.condition_mode = condition_mode

    def first_processing_real(self, data):
        if self.real_mode==None:
            print('please select real mode')
            return None
        
        elif self.real_mode=='jpg':
            img = image.imread(data)

            height = img.shape[0]
            width = img.shape[1]

            real = img[:,int(width/2):]
        
        return real
            
    
    def first_processing_condition(self, data):
        if self.condition_mode==None:
            print('please select condition mode')
            return None

        elif self.condition_mode=='jpg':
            img = image.imread(data)

            height = img.shape[0]
            width = img.shape[1]

            condition = img[:, 0:int(width/2)]
            condition_r = condition[:,:,0:1]
            condition_g = condition[:,:,1:2]
            condition_b = condition[:,:,2:3]

            condition = condition_r/3 + condition_g/3 + condition_b/3 # conver to gray scale

        return condition

    def to_FloatTensor(self,data):
        return torch.FloatTensor(data)
    
    def to_LongTensor(self,data):
        return torch.LongTensor(data)
    
    def Scaling(self, data, range=[-1,1], data_min=0, data_max=255):
        if data_min == None:
            if data_max == None:
                data_min = data.min()
                data_max = data.max()
            else:
                print("please set min max value properly (data_min is None)")
        else:
            if data_max == None:
                print("please set min max value properly (data_max is None)")
        
        data = ((data - data_min) / (data_max - data_min)) - 0.5 #scale to -0.5~0.5
        scale = range[1] - range[0]
        middle = range[0] + (scale / 2)
        data = data * scale + middle
        
        return data

    def MinMaxScale(self,data):
        scalar=MinMaxScaler(feature_range=(0,1))
        scalar.fit(data)
        scaled_data=scalar.transform(data)
        return scaled_data
    
    def image_pixel_scale(self,data): #scale 0~1
        #type(data) = list
        data = torch.FloatTensor(data)
        data = data.view(1,-1)
        filter1 = transforms.ToPILImage()
        filter2 = transforms.ToTensor()
        data = filter1(data)
        data = filter2(data)
        data = data.view(-1)
        return data
    
    def final_processing(self,data):
        data = data.transpose(0,2).transpose(1,2)
        return data

class my_transform (object):
    def __init__(self, real_process, condition_process=None):
        self.real_process = real_process
        self.condition_process = condition_process
    
    def __call__(self, sample):
        # real image processing
        if self.real_process:
            for process in self.real_process:
                sample['real'] = process(sample['real'])

        # condition processing
        if self.condition_process:
            for process in self.condition_process:
                sample['condition'] = process(sample['condition'])

        return sample
        

class Mydataset(Dataset):
    def __init__(self, conditions, reals, transform=None, root_dir=None):
        self.conditions = conditions
        self.reals = reals
        self.transform = transform

    def __len__(self):
        return len(self.reals)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}
        sample['real'] = self.reals[idx]

        if self.conditions == None: # if model don't use condition
            sample['condition'] = None
        else:
            sample['condition'] = self.conditions[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

