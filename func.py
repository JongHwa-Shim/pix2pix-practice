import matplotlib.pyplot as plt
from matplotlib import image

def pixel_analysis(path):
    img = image.imread(path)
    plt.imshow(img)
    plt.show()

#pixel_analysis("./data/pix2pix-dataset/edges2shoes/edges2shoes/train/1_AB.jpg")