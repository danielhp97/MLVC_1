from utils.mnist_reader import *
import matplotlib.pyplot as plt
import numpy as np
import pandas
import skimage
import skimage.io
import skimage.measure
import os

# method for converting data to a 28x28 image
def create_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    # two_d = np.subtract(255,two_d)
    plt.imshow(two_d, interpolation='nearest',cmap='gray')
    return two_d

def make_images(path, images, labels):
    image_list = []
    for (i, image), label in zip(enumerate(images), labels):
        #filepath = path / '{}_{}.jpg'.format(label, i)
        #image_list[label,i] = image.reshape(28, 28)
        image_filename = str(label)+'_'+str(i)+'.jpg'
        skimage.io.imsave(os.path.join(path, image_filename), image.reshape(28, 28))
        #skimage.io.imsave(os.path.join(path, image_filename), image_list[label, i])

# def make_labellist(path, kind, labels):
#     path.mkdir(parents=True, exist_ok=True)
#     filepaths = [
#         '{}_{}.jpg'.format(label, i) for i, label in enumerate(labels)
#     ]
#     df = pd.DataFrame({'name': filepaths, 'target': labels.tolist()})
#     df.to_csv(path / '{}.csv'.format(kind), index=False, header=False)

# Read data
x_train, y_train = load_mnist('./fashion', kind='train')
x_test, y_test = load_mnist('./fashion', kind='t10k')

print('Images_train:', x_train.shape)
print('Labels_train:', y_train.shape)
print('Images_test:', x_test.shape)
print('Labels_test:', y_test.shape)

#img = create_image(images[0])
make_images('./images/train2', x_train, y_train)
#make_labellist(x_train, y_train)
make_images('./images/test2', x_test, y_test)