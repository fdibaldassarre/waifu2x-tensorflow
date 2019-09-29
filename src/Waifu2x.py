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


def save_image_to(data, path):
    data = np.minimum(np.maximum(0., data[0]), 1.)
    data = np.uint8(np.round(data * 255.))
    image = Image.fromarray(data)
    image.save(path)


def load_weights(config):
    weights = np.asarray(config["weight"], dtype=np.float32).transpose(2, 3, 1, 0)
    bias = np.asarray(config["bias"], dtype=np.float32)
    return [weights, bias]


def create_conv2D_layer(config, activation=None):
    weights = load_weights(config)
    layer = layers.Conv2D(config["nOutputPlane"],
                          strides=(config["dH"], config["dW"]),
                          kernel_size=(config["kH"], config["kW"]),
                          activation=activation,
                          padding='same',
                          weights=weights)
    return layer


def create_conv2Dtranspose_layer(config):
    weights = load_weights(config)
    layer = layers.Conv2DTranspose(config["nOutputPlane"],
                                   strides=(config["dH"], config["dW"]),
                                   kernel_size=(config["kH"], config["kW"]),
                                   padding='same',
                                   weights=weights)
    return layer


class Waifu2x:

    def __init__(self, input_path):
        self.img = Image.open(input_path)
        self._operation = None
        self._noise_level = 0

    def _get_model_path(self):
        print(self._noise_level)
        if self._operation == OP_NOISE:
            model_name = 'vgg_7/art/noise%d_model.json' % self._noise_level
        elif self._operation == OP_SCALE:
            model_name = 'upconv_7/art/scale2.0x_model.json'
        elif self._operation == OP_NOISE_SCALE:
            model_name = 'upconv_7/art/noise%d_scale2.0x_model.json' % self._noise_level
        return os.path.join(MODELS_FOLDER, model_name)

    def _load_layers(self):
        model_path = self._get_model_path()
        decoder = json.JSONDecoder()
        with open(model_path, 'r') as hand:
            data = hand.read().strip()
        return decoder.decode(data)

    def _build_model(self):
        if self._operation == OP_NOISE:
            return self._build_vgg7()
        else:
            return self._build_upconv()

    def _build_vgg7(self):
        layers = self._load_layers()
        model = Sequential()
        for i in range(0, 6):
            model.add(create_conv2D_layer(layers[i], activation=leaky_relu))
        model.add(create_conv2D_layer(layers[6]))
        return model

    def _build_upconv(self):
        layers = self._load_layers()
        model = Sequential()
        for i in range(0, 6):
            model.add(create_conv2D_layer(layers[i], activation=leaky_relu))
        model.add(create_conv2Dtranspose_layer(layers[6]))
        return model

    def _get_input_tensor(self):
        data = np.asarray(self.img, dtype=np.float32) / 255.
        return np.expand_dims(data, axis=0)

    def _compute_result_and_save(self, output_path):
        model = self._build_model()
        input_data = self._get_input_tensor()
        result = model.predict(input_data)
        save_image_to(result, output_path)

    def noise(self, output_path, noise_level):
        self._operation = OP_NOISE
        self._noise_level = noise_level
        self._compute_result_and_save(output_path)

    def scale(self,  output_path):
        self._operation = OP_SCALE
        self._noise_level = 0
        self._compute_result_and_save(output_path)

    def noise_scale(self, output_path, noise_level):
        self._operation = OP_NOISE_SCALE
        self._noise_level = noise_level
        self._compute_result_and_save(output_path)


def scale(input_path, output_path):
    waifu2x = Waifu2x(input_path)
    waifu2x.scale(output_path)


def denoise(input_path, output_path, noise_level):
    waifu2x = Waifu2x(input_path)
    waifu2x.noise(output_path, noise_level)


def denoise_scale(input_path, output_path, noise_level):
    waifu2x = Waifu2x(input_path)
    waifu2x.noise_scale(output_path, noise_level)
