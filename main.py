from packages import *

import data_manager
# import perceptron

### This is the starting point of the program ###
def main():

    ### Read data
    x_train, y_train, x_test, y_test = data_manager.read_mnist_data()

    ### In case the images should be saved to jpg files
    # save_images_to_jpgs('./images/train2', x_train, y_train)
    # save_images_to_jpgs('./images/test2', x_test, y_test)

    ### Extracting subset of classes Dress(3) and Sneaker(7)
    indices = np.where((y_train == 3) | (y_train == 7))
    mnist_subset_rows = x_train[indices]
    mnist_subset = mnist_subset_rows.reshape(mnist_subset_rows.shape[0],28,28)

    features_subset = data_manager.extract_features(mnist_subset[:20])
    print(features_subset)
    
    plt.imshow(mnist_subset[0], cmap='gray')
    plt.show()
     

    #plt.plot(X,Y, 'o', color='red' if X == -1 else 'blue')

if __name__ == "__main__":
    main()