from packages import *


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

def extract_features(dataset: np.array, labels: np.array):
    features = pandas.DataFrame()
    for i,img in enumerate (dataset):
        region = skimage.measure.regionprops(img)[0]
        
        features = features.append (
            {
                'image'  : str(i),    # ID of image
                'class'  : 'Ankle boot' if labels[i] == int(9) else 'Trouser', # Class
                'cx'   : region.centroid[1],
                'cy'   : region.centroid[0],
                'area' : region.area,
                'convex_area' : region.convex_area,
                'eccentricity' : region.eccentricity,
                'perimeter' : region.perimeter,
                'extent'  : region.extent
            },
            ignore_index = True
        )
    features = features.set_index ('image')
    return features



# hints for improved subset selection:
# seed, ratio, indices


