import tensorflow as tf

def load_dataset(name):
    if name == 'mnist':
        data = tf.keras.datasets.mnist
    elif name == 'fashion_mnist':
        data = tf.keras.datasets.fashion_mnist
    elif name == 'cifar10':
        data = tf.keras.datasets.cifar10
    else:
        raise ValueError("Unsupported dataset")

    (x_train, y_train), (x_test, y_test) = data.load_data()

    if name == 'cifar10':
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
    else:
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    return (x_train, y_train), (x_test, y_test)
