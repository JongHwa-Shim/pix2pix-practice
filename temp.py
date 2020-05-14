from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

def square_plot(data, path):
    """Take an array of shape (n, height, width) or (n, height, width , 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    if type(data) == list:
	    data = np.concatenate(data)
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))

    padding = (((0, n ** 2 - data.shape[0]) ,
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data , padding, mode='constant' , constant_values=1)  # pad with ones (white)

    # tilethe filters into an image
    data = data.reshape((n , n) + data.shape[1:]).transpose((0 , 2 , 1 , 3) + tuple(range(4 , data.ndim + 1)))

    data = data.reshape((n * data.shape[1] , n * data.shape[3]) + data.shape[4:])

    plt.imsave(path, data, cmap='gray')
"""
x = torch.randn(25,28,28)
square_plot(x,None)
im = Image.open('./sample/image_4.jpg')
trans1 = transforms.ToTensor()
im = trans1(im)
im = im.reshape((1,3,im.shape[1],im.shape[2]))
im = [im for _ in range(25)]
im = torch.cat(im,0)
x = im
"""
def asdf (data, path, mode):

   data = (data - data.min()) / (data.max() - data.min())
   
   # batch will be n^2
   num_dim = data.ndim
   height = data.size(num_dim-2)
   width = data.size(num_dim-1)

   if height > width:
      pad = height
   else:
      pad = width
   #pad = int(pad/20) #padding ratio
   pad = 4
   n = int(np.sqrt(data.size(0))) # e.g batch = 25 >> n = 5

   if mode=='gray':
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
         
      plt.imsave(path, sample_image, cmap="gray")

   elif mode=='RGB':
      # shape: (batch, channel=3, height, width)
      data = data.view(data.size(0), 3, height, -1) 
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
      plt.imsave(path, sample_image) # image shape should be (H x W x 3)

   else:
      print("Error: this function only apply gray and RGB mode")

#asdf(x,'./result/sample.jpg', mode='RGB')
#square_plot(x,'./result/sample.jpg')
