from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

def visualization (data, path, mode='gray'):

    # batch will be n^2
    num_dim = data.ndim
    height = data.size(num_dim-2)
    width = data.size(num_dim-1)
    """
    if height > width:
        pad = height
    else:
        pad = width
    """
    #pad = int(pad/20) #padding ratio
    pad = 1
    n = int(np.sqrt(data.size(0))) # e.g batch = 25 >> n = 5


    if mode=='gray':

        # data normalizing for visualization *can be deprecated
        """
        for i in range(data.size(0)):
            data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())
        """
        data = data.cpu().data.numpy()

        # shape: (batch, height, width)
        #data = data.view(data.size(0), height, -1) 
        padding = ((0, 0), (pad, pad), (pad, pad))
        data = np.pad(data, padding, mode='constant', constant_values=1)

        #generate sample image
        ####################################
        sample_image = [] 

        height = data.shape[1]
        width = data.shape[2]

        start = 0
        end = n
        for j in range(n):
            for i in range(height):
                row = [ element for one_image in data[start:end] for element in one_image[i]]
                sample_image.append(row)

            start = start + n
            end = end + n
        ######################################
         
        plt.imsave(path, sample_image, cmap="gray", vmin=-1, vmax=1)

    elif mode=='RGB':

        data = data.view(data.size(0), 3, height, -1)

        # data normalizing for visualization *can be deprecated
        """
        for i in range(data.size(0)):
            data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())
        """
        data = data.cpu().data.numpy()

        # shape: (batch, channel=3, height, width)
        padding = ((0, 0), (0, 0), (pad, pad), (pad, pad))
        data = np.pad(data, padding, mode='constant', constant_values=1)

        #generate sample image (3, height, width)
        ##################################
        sample_image = [[],[],[]] 

        height = data.shape[2]
        width = data.shape[3]

        for k in range(3):
            start = 0
            end = n
            for j in range(n):
                for i in range(height):
                    row = [ element for one_image in data[start:end,k] for element in one_image[i]]
                    sample_image[k].append(row)

            start = start + n
            end = end + n
        sample_image = np.array(sample_image) # shape(3 x H x W)
      
        sample_height = sample_image.shape[1]
        sample_width = sample_image.shape[2]

        sample_image = np.transpose(sample_image,(1,2,0))
        #############################################

        #sample_image = sample_image.reshape((2560,1206,3))
        plt.imsave(path, sample_image, vmin=-1, vmax=1) # image shape should be (H x W x 3)
        

    else:
        print("Error: this function only apply gray and RGB mode")

#test code
if __name__=="__main__":
    im = Image.open('./sample/image_2.jpg')
    trans1 = transforms.ToTensor()
    im = trans1(im)
    im = im.reshape((1,3,im.shape[1],im.shape[2]))
    im = [im for _ in range(25)]
    im = torch.cat(im,0)
    x = im
    visualization(x,'./result/sample.jpg', mode='RGB')
