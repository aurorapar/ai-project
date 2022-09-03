import os
import time

import numpy as np
from mlxtend.data import mnist_data
from numba import njit

import config


def debug(msg):
    if True:
        print(msg)


def set_configuration_options():
    custom_config = config.settings
    for k, v in custom_config.items():
        print(f"Unit testing config\n\tkey: {k}")
        if k.startswith('training'):
            assert (v or mnist_data)
        else:
            assert v
    return custom_config


@njit
def create_activation_layer(layer_size: int):
    activation_layer = np.array((layer_size))
    return activation_layer


@njit
def create_hidden_layer(layer_size: int):
    x = np.random.randint(low=-10, high=10, size=(layer_size, 3))
    y = np.random.rand(layer_size, 3)
    hidden_layer = np.add(x, y)
    return hidden_layer


@njit
def create_output_layer(layer_size: int):
    output_layer = np.array((layer_size))
    return output_layer


@njit
def forward_propagate(previous_layer, current_layer):
    current_layer[0] = np.mult(previous_layer, current_layer[1]


@njit
def find_loss(batch_images, batch_labels):
    loss = 0
    for test_index in range(0, len(batch_images)):
        net[0] = batch_images[test_index]


def process_batches(network):
    images, labels = mnist_data()
    for batch_index in range(0, len(images), CONFIG['batch_size']):

        batch = images[batch_index: min(len(images), batch_index+CONFIG['batch_size']) ]
        error = 0

        for image in batch:
            network[0] = image
            for layer_index in range(1, len(network)):
                forward_propogate(network[layer_index-1], network[layer_index])


if __name__ == '__main__':
    CONFIG = set_configuration_options()

    # for custom data set
    training_data_path = CONFIG['training_data_path'] or mnist_data
    training_label_file = CONFIG['training_data_label_file'] or mnist_data
    # ignore if using MNIST training set
    if training_data_path != mnist_data:
        assert os.path.exists(training_label_file)

    if training_label_file != mnist_data:
        assert os.path.exists(training_label_file)

    net = [create_activation_layer(CONFIG['layer_sizes'][0])]
    for layer_size_index in range(1, len(CONFIG['layer_sizes']) - 1):
        debug(f"Creating layer of size {CONFIG['layer_sizes'][layer_size_index]}")
        net.append(create_hidden_layer(CONFIG['layer_sizes'][layer_size_index]))
    net.append(create_output_layer(CONFIG['layer_sizes'][-1]))

    process_batches(net)
