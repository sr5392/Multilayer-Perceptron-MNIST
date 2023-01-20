from keras.datasets import mnist
from keras.utils import to_categorical

def get_image_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2], 1)
    train_images = train_images.astype('float32')
    train_images /= 255

    train_labels = to_categorical(train_labels)
    train_labels = train_labels.reshape(train_labels.shape[0], train_labels.shape[1], 1)
                                        
    test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2], 1)
    test_images = test_images.astype('float32')
    test_images /= 255

    test_labels = to_categorical(test_labels)
    test_labels = test_labels.reshape(test_labels.shape[0], test_labels.shape[1], 1)

    return (train_images, train_labels), (test_images, test_labels)
