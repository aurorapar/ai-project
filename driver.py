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
def create_layer(layer_size: int, weights_needed: int):
    x = np.zeros((layer_size, 1))
    w = np.random.rand(weights_needed, 1)
    b = np.add(np.random.rand(weights_needed), np.random.randint(low=-10, high=10, size=(1)))
    return x, w, b


@njit
def create_output_layer(layer_size: int):
    return np.zeros((layer_size))


@njit
def forward_propagate(a, w, b):
    # debug(f"Propagating to layer {index+1}")
    # in order to fit into dimension of next layer, use next layer input as first parameter
    # z = np.sum(previous_layer[1].reshape(-1, 1) @ previous_layer[0].reshape(1, -1), axis=1)
    z = np.sum(w @ a, axis=1)
    next_layer = np.add(z, b)
    return next_layer


@njit
def find_loss(batch_images, batch_labels):
    loss = 0
    for test_index in range(0, len(batch_images)):
        net[0] = batch_images[test_index]


@njit
def relu_fn(rel_input):
    return np.array([max(0, x) for x in rel_input])


# @njit
def softmax_fn(softmax_input):
    softmax_input -= np.max(softmax_input)
    softmax_input = np.exp(softmax_input)
    if np.inf in softmax_input:
        raise RuntimeError(f"Inf/Overflow detected after softmax\n\t{softmax_input}")
    softmax_input /= np.sum(np.exp(softmax_input))
    return softmax_input


def process_batches(network):
    images, labels = mnist_data()
    print(f"Beginning batch processing")

    processed_images = 0
    loss = 0
    accuracy = 0
    for batch_index in range(0, len(images), CONFIG['batch_size']):

        batch = images[batch_index: min(len(images), batch_index+CONFIG['batch_size']) ]

        for image_index in range(0, len(batch)):
            # network[l][n=0, w=1, b=2]
            network[0][0] = batch[image_index] / 255
            for layer_index in range(1, len(network)):

                inputs = network[layer_index-1][0].reshape(1, -1)
                weights = network[layer_index-1][1].reshape(-1, 1)
                biases = network[layer_index-1][2]
                new_output = forward_propagate(inputs, weights, biases)
                if layer_index < len(network) - 1:
                    network[layer_index][0] = relu_fn(new_output)
                else:
                    network[layer_index] = softmax_fn(new_output)

            label_index = np.argmax(network[-1])
            if int(label_index == labels[batch_index + image_index]):
                accuracy += 1
                print(f"Image {batch_index + image_index + 1} classified correctly")
            processed_images += 1

        print(f"Processed {batch_index + CONFIG['batch_size']} images")
        print(f"Accuracy: {accuracy / processed_images * 100:.2f}%")

    print(f"Total matches: {accuracy}")


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

    net = []
    for layer_size_index in range(0, len(CONFIG['layer_sizes']) - 1):
        new_layer = np.array(
            (create_layer(CONFIG['layer_sizes'][layer_size_index], CONFIG['layer_sizes'][layer_size_index+1])),
            dtype=object
        )
        net.append(new_layer)
    net.append(create_output_layer(CONFIG['layer_sizes'][-1]))

    process_batches(net)
