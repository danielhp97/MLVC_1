from utils.mnist_reader import *
import matplotlib.pyplot as plt
import numpy as np
import pandas
import skimage
import skimage.io
import skimage.measure

# method for converting data to a 28x28 image
def create_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    # two_d = np.subtract(255,two_d)
    plt.imshow(two_d, interpolation='nearest',cmap='gray')
    return two_d

# Read data
[images, labels] = load_mnist("./fashion")
print('Images:', images.shape)
print('Labels:', labels.shape)

img = create_image(images[0])