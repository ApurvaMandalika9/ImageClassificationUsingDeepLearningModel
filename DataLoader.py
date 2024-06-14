import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""
def load_from_pickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE

    # train data
    file_prefix = f"{data_dir}/data_batch_"
    for i in range(1,6):
        data = load_from_pickle(file_prefix+str(i))
        if i==1:
            x_train = data[b"data"]
            y_train = data[b"labels"]
        else:
            x_train = np.concatenate((x_train,data[b"data"]),axis=0)
            y_train.extend(data[b'labels'])
    y_train = np.array(y_train)

    # test data
    file = f"{data_dir}/test_batch"
    data = load_from_pickle(file)
    x_test = data[b"data"]
    y_test = np.array(data[b"labels"])
    print(f"x_train: {x_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"x_test: {x_test.shape}")
    print(f"y_test: {y_test.shape}")

    ### END CODE HERE

    return x_train, y_train, x_test, y_test


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 3072].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    file_name = f"{data_dir}/private_test_images_2024.npy"
    x_test = np.load(file_name)

    print("Verification: Shape of the test set is", x_test.shape)

    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """

    ### YOUR CODE HERE
    split_index = int(np.multiply(x_train.shape[0], train_ratio))

    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid
