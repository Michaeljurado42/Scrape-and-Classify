from scipy.sparse.construct import random
import tensorflow as tf
from tensorflow.keras.applications import MobileNet, ResNet50, MobileNetV2
from tensorflow.keras.layers import Input, Dense, Flatten
import tensorflow.keras.preprocessing
import argparse
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.python.ops.gen_math_ops import mod

def deep_learning_wrapper(image_size: tuple, MODEL, num_logits: int):
    """
    image_size: size of input (h, w)
    MODEL: ResNet50, MobileNet, or some other network from tensorflow.keras.applications
    num_logits: number of classes
    """
    input_ = Input(image_size)

    deep_model = MODEL(include_top=False, input_shape=image_size)
    deep_model.trainable = True

    flat_out = Flatten()(deep_model(input_))
    logits_out = Dense(num_logits)(flat_out)
    model = tf.keras.Model(inputs = input_, outputs = logits_out)
    return model
    


import sklearn
import sklearn.model_selection
import numpy as np

from shutil import copyfile
def create_training_testing_split(test_split_percent = .15, validation_split_percent = .1, classes = []):
    dirs = os.listdir("dataset")
    if not os.path.isdir("dataset/train"):
        os.mkdir("dataset/train")
        os.mkdir("dataset/validation")
        os.mkdir("dataset/test")
    else:
        dirs.remove("train")
        dirs.remove("validation")
        dirs.remove("test")
    dirs = [i for i in dirs if i in classes]
    for dir_ in dirs:
        if not os.path.isdir("dataset/train/%s" % dir_):
            os.mkdir("dataset/train/%s" % dir_)
            os.mkdir("dataset/validation/%s" % dir_)
            os.mkdir("dataset/test/%s" % dir_)

        files = os.listdir("dataset/%s" % dir_)
        train_files, test_files = sklearn.model_selection.train_test_split(np.array(files), test_size = test_split_percent, random_state=42)

        train_files, val_files = sklearn.model_selection.train_test_split(np.array(train_files), test_size = test_split_percent, random_state=42)
        for i in train_files:
            copyfile("dataset/%s/%s" % (dir_, i), "dataset/train/%s/%s" % (dir_, i))
        for i in test_files:
            print("dataset/%s/%s" % (dir_, i))
            print("dataset/test/%s/%s" % (dir_, i))
            copyfile("dataset/%s/%s" % (dir_, i), "dataset/test/%s/%s" % (dir_, i))        
        for i in val_files:
            copyfile("dataset/%s/%s" % (dir_, i), "dataset/validation/%s/%s" % (dir_, i))        


import os
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="scrapes google images using google chrome and puts the images into a dataset folder")
    p.add_argument('classes',  nargs='*', type = str, help = "classes to include in the classification problem. (ie dog vs cat or dog vs random")    
    p.add_argument("--model_type", type = str, help = "model type. Supports mobilenet, resnet50, mobilenetv2", default = "mobilenet")
    args = p.parse_args()
    model_type = args.model_type
    classes = args.classes

    # create valid dataset partitions
    create_training_testing_split(classes = classes)  # split data into train test split

    if model_type == "resnet":
        preprocessing_function = tensorflow.keras.applications.resnet.preprocess_input
        MODEL = ResNet50
    elif model_type == "mobilenet":
        preprocessing_function = tensorflow.keras.applications.mobilenet.preprocess_input
        MODEL = MobileNet
    elif model_type == "mobilenetv2":
        preprocessing_function = tensorflow.keras.applications.mobilenet_v2.preprocess_input
        MODEL = MobileNetV2
    
    # apply numerous augmentations
    train_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=10,
        brightness_range=[.3, 1],
        width_shift_range=.1,
        height_shift_range=.1,
        preprocessing_function=preprocessing_function)

    validation_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocessing_function)    

    # create data generators
    training_generator = train_datagen.flow_from_directory("dataset/train", target_size=(256, 256), class_mode="sparse")
    validation_generator = validation_datagen.flow_from_directory("dataset/validation", target_size=(256, 256), class_mode="sparse")
    test_generator = validation_datagen.flow_from_directory("dataset/test", target_size=(256, 256), class_mode="sparse")
    
    # create transfer learn model
    model = deep_learning_wrapper((256, 256, 3), MobileNet, np.unique(training_generator.classes).shape[0])
    loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="Adam", loss=loss, metrics=["accuracy"])

    # define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience = 3)
    mcp_save = tf.keras.callbacks.ModelCheckpoint('trained_model.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    
    # fit and save best model
    history = model.fit(training_generator, validation_data = validation_generator, epochs = 30, callbacks = [early_stopping, mcp_save])
    
    # make learning curve and save
    plt.figure()
    plt.plot(np.arange(len(history.history["val_loss"])), history.history["val_loss"], label  = "validation")
    plt.plot(np.arange(len(history.history["val_loss"])), history.history["loss"], label  = "training")
    plt.title("Learning Curve")
    plt.xlabel("epochs")
    plt.ylabel("Categorical Crossentropy loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("learning_curve.png")

    # get testing metrics
    score = model.evaluate_generator(generator=test_generator)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    