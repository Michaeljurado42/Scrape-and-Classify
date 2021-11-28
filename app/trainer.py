import numpy as np
import cv2
import os
import random
import matplotlib.pylab as plt

import tensorflow as tf
from numpy import asarray
import pickle
import pathlib
from sklearn.model_selection import train_test_split
from utils.keras_vis_helper import flatten_model

data_dir = "./dataset/"

def preprocess(data_classes, scale=True):
    """
    Retrieves images for given classes
    Performs any necessary image processing
    Additional augmentations can be integrated here
    """
    images = []
    labels = []

    for i, class_dir in enumerate(data_classes):
        if os.path.isdir(data_dir + class_dir):
            for image in os.listdir(data_dir + class_dir):
                images.append(cv2.resize(cv2.imread(data_dir + class_dir + "/" + image), (224, 224)))
                labels.append(i)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.30)

    X_train = np.array(X_train).astype('float16',casting='same_kind')
    y_train = np.array(y_train).astype('float16',casting='same_kind')
    X_test = np.array(X_test).astype('float16',casting='same_kind')
    y_test = np.array(y_test).astype('float16',casting='same_kind')

    if scale:
        X_train = X_train / 255
        X_test = X_test / 255

    return X_train, y_train, X_test, y_test

def train(id, model_name, classes, lr, epochs, batch_size):
    print("Loading Data ...")
    X_train, y_train, X_test, y_test = preprocess(os.listdir(data_dir))

    print("Building Architecture ...")
    if model_name == "mnv2":
        pretrained_model_without_top_layer = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
        pretrained_model_without_top_layer.trainable = False

    input = tf.keras.layers.Input((224, 224, 3))
    model_out = pretrained_model_without_top_layer(input)
    flat_out = tf.keras.layers.Flatten()(model_out)
    output = tf.keras.layers.Dense(len(classes))(flat_out)
    model = tf.keras.Model(inputs = input, outputs = output)
    model = flatten_model(model)
    print(model.summary())

    model.compile(
        optimizer= tf.keras.optimizers.Adam(learning_rate=lr), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['acc']
        )
  
    print("Training Model ...")
    train_history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    print("Testing Model ...")
    test_history = model.evaluate(X_test, y_test)

    print("Saving Model ...")
    model.save(f'models/{id}_{model_name}.h5')

    return f'models/{id}_{model_name}.h5', train_history, test_history

