#len(dataset) 에서 호출되는 __len__ 은 데이터셋의 크기를 리턴
#dataset[i] 에서 호출되는 __getitem__ 은 i번째 샘플을 찾는데 사용
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

class transform_processing(object):
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

class my_transform (object):
    def __init__(self, real_process, condition_process=None):
        self.real_process = real_process
        self.condition_process = condition_process
    
    def __call__(self, sample):
        real = sample['real']
        condition = sample['condition']

        # real image processing
        if self.real_process:
            for process in self.real_process:
                real = process(real)

        # condition processing
        if self.condition_process:
            for process in self.condition_process:
                condition = process(condition)
            

        

        sample['real'] = real
        sample['condition'] = condition
        return sample
        

class Mydataset(Dataset):
    def __init__(self, sources, targets, transform=None, root_dir=None):
        self.sources = sources
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}
        sample['real'] = self.sources[idx]

        if self.targets == None: # if model don't use condition
            sample['condition'] = None
        else:
            sample['condition'] = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

