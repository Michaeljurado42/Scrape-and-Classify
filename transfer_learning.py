from scipy.sparse.construct import random
import tensorflow as tf
from tensorflow.keras.applications import MobileNet, ResNet50, MobileNetV2
from tensorflow.keras.layers import Input, Dense, Flatten
import tensorflow.keras.preprocessing
import argparse
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.python.ops.gen_math_ops import mod

def deep_learning_wrapper(image_size: tuple, MODEL, num_logits: int, trainable = False):
    """
    image_size: size of input (h, w)
    MODEL: ResNet50, MobileNet, or some other network from tensorflow.keras.applications
    num_logits: number of classes
    trainable: frozen weights or not
    """
    input_ = Input(image_size)

    deep_model = MODEL(include_top=False, input_shape=image_size)
    deep_model.trainable = trainable

    flat_out = Flatten()(deep_model(input_))
    logits_out = Dense(num_logits)(flat_out)
    model = tf.keras.Model(inputs = input_, outputs = logits_out)
    return model
    


import sklearn
import sklearn.model_selection
import numpy as np

from shutil import copyfile
def create_training_testing_split(test_split_percent = .15, validation_split_percent = .1, classes = [], train_dir = "dataset/train", test_dir = "dataset/test", val_dir = "dataset/val"):

    dirs = os.listdir("dataset")
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        os.mkdir(val_dir)
    else:
        os.remove(train_dir)
        os.remove(val_dir)
        os.remove(test_dir)
    dirs = [i for i in dirs if i in classes]
    image_id = 0
    for dir_ in dirs: # iterate over classes
        if not os.path.isdir("%s/%s" % (train_dir, dir_)):  # make subdirectories
            os.mkdir("%s/%s" % (train_dir, dir_)) 
            os.mkdir("%s/%s" % (test_dir, dir_))
            os.mkdir("%s/%s" % (val_dir, dir_))

        files = os.listdir("dataset/%s" % dir_)
        train_files, test_files = sklearn.model_selection.train_test_split(np.array(files), test_size = test_split_percent, random_state=42)
        train_files, val_files = sklearn.model_selection.train_test_split(np.array(train_files), test_size = test_split_percent, random_state=42)

        for i in train_files:
            image_id += 1
            print("dataset/%s/%s" % (dir_, i))
            copyfile("dataset/%s/%s" % (dir_, i), "%s/%s/%d.jpg" % (train_dir, dir_, image_id))
        for i in test_files:
            image_id += 1
            print("dataset/%s/%s" % (dir_, i))
            copyfile("dataset/%s/%s" % (dir_, i), "%s/%s/%d.jpg" % (test_dir, dir_, image_id))        
        for i in val_files:
            image_id += 1
            print("dataset/%s/%s" % (dir_, i))
            copyfile("dataset/%s/%s" % (dir_, i), "%s/%s/%d.jpg" % (val_dir, dir_, image_id))        


import os
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="scrapes google images using google chrome and puts the images into a dataset folder")
    p.add_argument('classes',  nargs='*', type = str, help = "classes to include in the classification problem. (ie dog vs cat or dog vs random")    
    p.add_argument("--model_type", type = str, help = "model type. Supports mobilenet, resnet50, mobilenetv2", default = "mobilenet")
    args = p.parse_args()
    model_type = args.model_type
    classes = args.classes
    target_size = (224, 224)

    # create valid dataset partitions
    problem_string = "_".join(classes)
    train_dir = "dataset/train_" + problem_string
    test_dir = "dataset/test_" + problem_string
    val_dir = "dataset/val_" + problem_string
    #create_training_testing_split(classes = classes,train_dir=train_dir, test_dir=test_dir, val_dir=val_dir)  # split data into train test split

    if model_type == "resnet":
        preprocessing_function = tensorflow.keras.applications.resnet.preprocess_input
        MODEL = ResNet50
    elif model_type == "mobilenet":
        preprocessing_function = tensorflow.keras.applications.mobilenet.preprocess_input
        MODEL = MobileNet
    elif model_type == "mobilenetv2":
        preprocessing_function =  tensorflow.keras.applications.mobilenet_v2.preprocess_input
        MODEL = MobileNetV2
    
    # apply numerous augmentations
    train_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
        #shear_range=0.2,
        #zoom_range=0.2,
        horizontal_flip=True,
        # vertical_flip=True,
#        rotation_range=10,
        #brightness_range=[.3, 1],
        # width_shift_range=.1,
        # height_shift_range=.1,
        preprocessing_function=preprocessing_function)
    
    validation_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocessing_function)    
    testing_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocessing_function)    

    # create data generators
    training_generator = train_datagen.flow_from_directory(train_dir, target_size=target_size, class_mode="sparse", batch_size = 128, shuffle = True)
    validation_generator = validation_datagen.flow_from_directory(val_dir, target_size=target_size, class_mode="sparse",  batch_size = 128, shuffle = False)
    test_generator = testing_datagen.flow_from_directory(test_dir, target_size=target_size, class_mode="sparse",  batch_size = 128, shuffle = False)

    labels = (training_generator.class_indices)
    print(labels)
    # create transfer learn model
    image_size = tuple(list(target_size) + [3])
    model = deep_learning_wrapper(image_size, MODEL, np.unique(training_generator.classes).shape[0])
    loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="Adam", loss=loss)

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
    plt.xlabel("Epochs")
    plt.ylabel("Categorical Crossentropy loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("learning_curve.png")

    # get testing metrics
    predictions = model.predict(test_generator)
    argmax_predictions = np.argmax(predictions, axis = -1)
    accuracy = np.sum(argmax_predictions == validation_generator.classes)/ len(argmax_predictions)
    print("Testing accuracy", accuracy)
    print("Test evaluate", model.evaluate(test_generator))
    print("Train evaluate", model.evaluate(training_generator))
    print("val evaluate", model.evaluate(validation_generator))



    print("saving model")
    model.save(problem_string + "_model.h5")
    