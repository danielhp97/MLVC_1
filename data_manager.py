from utils.mnist_reader import *
import matplotlib.pyplot as plt
import numpy as np
import pandas
import skimage
import skimage.io
import skimage.measure
import os


### Read data
def read_mnist_data():
    x_train, y_train = load_mnist('./fashion', kind='train')
    x_test, y_test = load_mnist('./fashion', kind='t10k') 
    # print('Images_train:', x_train.shape)
    # print('Labels_train:', y_train.shape)
    # print('Images_test:', x_test.shape)
    # print('Labels_test:', y_test.shape)
    return x_train, y_train, x_test, y_test

def save_images_to_jpgs(path, images, labels):
    for (i, image), label in zip(enumerate(images), labels):
        image_filename = str(label)+'_'+str(i)+'.jpg'
        skimage.io.imsave(os.path.join(path, image_filename), image.reshape(28, 28))

def extract_features(dataset: np.array):
    features = pandas.DataFrame()
    for i,img in enumerate (dataset):
        region = skimage.measure.regionprops(img)[0]
        features = features.append (
            {
                'image'  : str(i),    # do not change this line, it is the image ID
                #'class'   : TOOL_NAMES [i // 5],   # do not change this line, it is the image tool name
                'cx'   : region.centroid[1],    # x and y centroids are useless for out tools ...
                'cy'   : region.centroid[0],    # ... but demonstrate how to append your 4 features 
                'area' : region.area,
                'convex_area' : region.convex_area,
                'eccentricity' : region.eccentricity,
                'perimeter' : region.perimeter,
            },
            ignore_index = True
        )
    features = features.set_index ('image')
    return features





# not working in VSC ?!
# plt.imshow (features_subset[0], cmap='gray')

# hint for subset:
# seed, ratio, indices


