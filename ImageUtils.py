import numpy as np
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    ### YOUR CODE HERE

    # Reshape from [depth * height * width] to [depth, height, width].
    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(record.reshape((3, 32, 32)), [1, 2, 0])

    ### END CODE HERE

    image = preprocess_image(image, training)

    return image


def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    ** IMPORTANT_NOTE: Start **
    - The above notes: "a single image of shape [height, width, depth]"
    - But the below Args noted that the shape of the array is [3, 32, 32]? - this does not make sense.
    - I think you meant [32, 32, 3]?
    - Therefore, I changed it - please see the below.
    ** IMPORTANT_NOTE: End **

    Args:
        (Original) image: An array of shape [3, 32, 32].
        (New) image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32]. The processed image.
    """
    ### YOUR CODE HERE

    transform_image = None

    if training:
        transform_image = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomVerticalFlip(p=0.5),
            #transforms.ToTensor(),

            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),  # Using AutoAugment for CIFAR-10
            transforms.ToTensor(),
            # normalizing constant - https://github.com/kuangliu/pytorch-cifar/issues/19
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
        ])
    else:
        transform_image = transforms.Compose([
            transforms.ToTensor(),
            # normalizing constant - https://github.com/kuangliu/pytorch-cifar/issues/19
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
        ])

    image = transform_image(Image.fromarray(image))
    ### END CODE HERE

    return image


def visualize(image, save_name='test.png'):
    """Visualize a single test image.

    Args:
        image: An array of shape [3072]
        save_name: An file name to save your visualization.

    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    print("-- VISUALIZATION --")
    image = np.transpose(image.reshape((3, 32, 32)), [1, 2, 0])
    ### YOUR CODE HERE

    plt.imshow(image)
    plt.savefig(save_name)
    return image


# Other functions
### YOUR CODE HERE
def preprocess_for_testing(image):

    image = np.transpose(image.reshape((3, 32, 32)), [1, 2, 0])

    transform_image = transforms.Compose([
        transforms.ToTensor(),
        # normalizing constant from https://github.com/kuangliu/pytorch-cifar/issues/19
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
    ])
    image = transform_image(Image.fromarray(image))
    return image
### END CODE HERE
