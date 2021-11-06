from tensorflow.keras import layers, models, datasets
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2

from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.scorecam import Scorecam
from matplotlib import cm
from tensorflow import keras 
from tensorflow.keras.models import Model 
from tensorflow.keras import backend as K

import tensorflow


def flatten_model(model_nested:tf.keras.Model):
    """Takes a keras model and automatically flattens it so it can be used with grad cam

    Args:
        model_nested (tf.keras.Model): model with Functional layers
    """
    def get_layers(layers):
        layers_flat = []
        for layer in layers:
            try:
                layers_flat.extend(get_layers(layer.layers))
            except AttributeError:
                layers_flat.append(layer)
        return layers_flat

    model_flat = tf.keras.Sequential(
        get_layers(model_nested.layers)
    )
    return model_flat

def generate_cam_map(model:tf.keras.Model, image:np.array, label:int, type = "gradcam"):
    """Generates cam visualization with identical dimensions to target image

    Args:
        model (tf.keras.Model): [description]
        image (np.array): [description]
        label (int): [description]
        type (str, optional): [description]. Defaults to "gradcam".

    Returns:
        [type]: [description]
    """
    #model = flatten_model(model)
    conv_layers = np.array(["conv" in str(l) for l in model.layers])
    replace2linear = ReplaceToLinear()

    # you can also define the function from scratch as follows:
    def model_modifier_function(cloned_model):
        cloned_model.layers[-1].activation = tf.keras.activations.linear

    model_modifier_function(model)

    def score_function(output):
        score_list = []
        for i in range(output.shape[0]):
            score_list.append(output[i, label])
        return tuple(score_list)

    if type == "gradcam":
        gradcam = Gradcam(model,
                        model_modifier=replace2linear,
                        clone=True)

        # Generate heatmap with GradCAM
        last_layer_idx = int(np.where(conv_layers)[0][-1]) - len(conv_layers)
        cam = gradcam(score_function,
                    np.array([image]),
                    penultimate_layer = last_layer_idx)

    elif type == "saliency":
        saliency = Saliency(model, replace2linear, clone = True)
        cam = saliency(score_function, np.array([image]), smooth_samples = 20, smooth_noise = .2)

    elif type == "scorecam":
        # Generate heatmap with Faster-ScoreCAM
        scorecam = Scorecam(model, model_modifier=replace2linear)
        cam = scorecam(score_function,
                    np.array([image]),
                    penultimate_layer=-1,
                    max_N=10) 
    return np.uint8(cm.jet(cam[0])[..., :3] * 255)


def generate_grad_cam_map(model:tf.keras.Model, image:np.array, label):
    return generate_cam_map(model, image, label, "gradcam")

def generate_saliency_map(model:tf.keras.Model, image:np.array, label):
    return generate_cam_map(model, image, label, "saliency")

def generate_scorecam_map(model:tf.keras.Model, image:np.array, label):
    return generate_cam_map(model, image, label, "scorecam")


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    images = x_test

    def normalize(X_train, X_test):
        return X_train/255, X_test/255
    x_train, x_test = normalize(x_train, x_test)
    x_test, y_test = x_test[:6], y_test[:6]

    image_dict = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }
    #pretrained_model_without_top_layer = tf.keras.applications.MobileNet(input_shape=(224, 224, 3), include_top=False)
    image_titles = [image_dict[int(i[0])] for i in y_test]
    model_path = "airplane_automobile_bird_cat_deer_dog_frog_horse_ship_truck_mobilenet_model.h5"
    model = tf.keras.models.load_model(model_path,   compile = False)

    methods = [generate_cam_map, generate_scorecam_map, generate_saliency_map]
    f, ax = plt.subplots(nrows=2, ncols=len(methods), figsize=(12, 4))
    for method_idx in range(len(methods)): # iterate over score methods
        vis_method = methods[method_idx]
        
        for i, title in enumerate(image_titles[:2]):
            heatmap = vis_method(model, cv2.resize(x_test[i], (224, 224)), y_test[i][0]) # generate cam vis
            if vis_method == generate_saliency_map:
                ax[i][method_idx].imshow(heatmap, cmap='jet') # do not overlay
            else:
                ax[i][method_idx].imshow(cv2.resize(images[i],(224,224)))
                ax[i][method_idx].imshow(heatmap, cmap='jet', alpha=0.5) # overlay

            if vis_method == generate_cam_map:
                ax[i][method_idx].set_title("Grad Cam for %s" % title)
            elif vis_method == generate_scorecam_map:
                ax[i][method_idx].set_title("Score Cam %s" % title)
            elif vis_method ==  generate_saliency_map:
                ax[i][method_idx].set_title("Saliency Map %s" % title)

            ax[i][method_idx].axis('off')
    plt.tight_layout()
    plt.show()


