#!/usr/bin/env python3

import json
import os

from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers

from src.Places import MODELS_FOLDER

OP_SCALE = 'scale'
OP_NOISE = 'noise'
OP_NOISE_SCALE = 'noise_scale'

LEAKY_ALPHA = tf.constant(0.1)
def leaky_relu(x):
    return tf.where(tf.greater(0.0, x), tf.multiply(x, LEAKY_ALPHA), x)
    #return LEAKY_ALPHA * x if x < 0 else x


class Waifu2x:

    def __init__(self, input_path):
        self.img = Image.open(input_path)
        self._operation = None

    def _get_model_path(self):
        return os.path.join(MODELS_FOLDER, 'vgg_7/art/noise0_model.json') # TODO

    def _load_layers(self):
        model_path = self._get_model_path()
        decoder = json.JSONDecoder()
        with open(model_path, 'r') as hand:
            data = hand.read().strip()
        return decoder.decode(data)

    def _create_conv_layer(self, data, activation=None):
        weights = np.asarray(data["weight"], dtype=np.float32).transpose(2, 3, 1, 0)
        bias = np.asarray(data["bias"], dtype=np.float32)
        layer = layers.Conv2D(data["nOutputPlane"],
                              strides=(data["dH"], data["dW"]),
                              kernel_size=(data["kH"], data["kW"]),
                              activation=activation,
                              weights=[weights, bias])
        return layer

    def _create_conv_transpose_layer(self, data):
        weights = np.asarray(data["weight"], dtype=np.float32) #.transpose(2, 3, 0, 1)
        bias = np.asarray(data["bias"], dtype=np.float32)
        layer = layers.Conv2DTranspose(data["nOutputPlane"],
                                       strides=(data["dH"], data["dW"]),
                                       kernel_size=(data["kH"], data["kW"]),
                                       weights=[weights, bias])
        return layer

    def _build_vgg7(self):
        layers = self._load_layers()
        model = Sequential()
        model.add(self._create_conv_layer(layers[0], activation=leaky_relu))
        model.add(self._create_conv_layer(layers[1], activation=leaky_relu))
        model.add(self._create_conv_layer(layers[2], activation=leaky_relu))
        model.add(self._create_conv_layer(layers[3], activation=leaky_relu))
        model.add(self._create_conv_layer(layers[4], activation=leaky_relu))
        model.add(self._create_conv_layer(layers[5], activation=leaky_relu))
        model.add(self._create_conv_layer(layers[6]))
        return model

    def _save_image_to(self, data, path):
        data = np.minimum(np.maximum(0., data[0]), 1.)
        data = np.uint8(np.round(data * 255.))
        print(data.min())
        print(data.max())
        print("Output", data.shape)
        image = Image.fromarray(data)
        image.save(path)

    def _get_input_tensor(self):
        data = np.asarray(self.img, dtype=np.float32) / 255.
        return np.expand_dims(data, axis=0)

    def scale(self, output_path):
        self._operation = OP_SCALE
        model = self._build_vgg7()
        # TODO: add padding to input
        input_data = self._get_input_tensor()
        print(input_data.shape)
        result = model.predict(input_data)
        print(result.shape)
        self._save_image_to(result, output_path)

    def noise(self, input_path, output_path, noise_level):
        self._operation = OP_NOISE
        pass

    def noise_scale(self, input_path, output_path, noise_level):
        self._operation = OP_NOISE_SCALE
        pass


def scale(input_path, output_path):
    waifu2x = Waifu2x(input_path)
    waifu2x.scale(output_path)

def denoise(input_path, output_path, noise_level):
    print("Not implemented")
    pass

def denoise_scale(input_path, output_path, noise_level):
    print("Not implemented")
    pass
