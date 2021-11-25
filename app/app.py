import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import os
import sys
sys.path.insert(0, '.')

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tensorflow.keras.applications import mobilenet
from utils.keras_vis_helper import generate_grad_cam_map, generate_saliency_map, generate_scorecam_map
import tensorflow as tf
from PIL import Image
import numpy as np
from scraping_script import *
from trainer import train
import uuid

app = dash.Dash(__name__)

matrix_df = px.data.medals_wide(indexed=True)

#Global variables
classes = None
classifier = None

# Global variables for grad cam
background = Image.open("graphics/machine_learning.png")
model_path_wrapper = ["airplane_automobile_bird_cat_deer_dog_frog_horse_ship_truck_mobilenet_model.h5"]
# model_wrapper = [tf.keras.models.load_model(model_path_wrapper[0])]
images = ["dataset/banana/0000.jpg", "dataset/banana/0001.jpg"]
image_idx_wrapper = [0]


app.layout = html.Div(children=[
    # Header
    html.H1(children='Sample App with Plotly'),

    # Read in classes
    html.Div(children='''
        Choose classes to build a model around:
    '''),
    dcc.Input(
        id="classes",
        type="text",
        placeholder="name of classes seperated by comma"
    ),
    dcc.Input(
        id="number_of_images",
        type="text",
        placeholder="number of images per class",
        style={'marginLeft':'20px'}
    ),
    html.Button('Create', id='submit-classes', n_clicks=0),
    html.Div(
        id="class_out",
        children="",
    ),

    # Select which model
    html.Div(children='''
        Choose which model to train:
    '''),
    html.Div([html.Div(dcc.Dropdown(
        id="model",
        options=[
            {'label': 'MobileNetV2', 'value': 'mnv2'},
        ],
        value='mnv2',
        clearable=False,
        searchable=False
    ), style={'width': '30%'}),

    # Select Hyperparameters
    html.Div(children='''
        Set hyperparameter values:
    '''),
    html.Div([
        dcc.Input(
            id="lr",
            type="number",
            placeholder="learning rate",
        ),
        dcc.Input(
            id="epochs",
            type="number",
            placeholder="number of epochs",
        ),
        dcc.Input(
            id="batch_size",
            type="number",
            placeholder="batch size",
        )
        ]),
    html.Div(id="model_out", children=""),
    dcc.Loading(
        id="model_loading",
        type="default",
        children=html.Div(id="loading-output")
    ),
    html.Button('Train', id='submit-model', n_clicks=0)]),
    html.Div(id="chosen_model",
        children="",
    ),
    html.Div(id="train_loss",
        children="",
    ),
    html.Div(id="train_acc",
        children="",
    ),
    html.Div(id="test_loss",
        children="",
    ),
    html.Div(id="test_acc",
        children="",
    ),

    # Confusion Matrix
    html.P("Medals included:"),
    dcc.Checklist(
        id='medals',
        options=[{'label': x, 'value': x} 
                 for x in matrix_df.columns],
        value=matrix_df.columns.tolist(),
    ),
    dcc.Graph(id="matrix"),

    # Matrix Element
    html.Div(id="cm_element",
        children="",
    ),
    

    # The rest of the code is dedicated to grad-cam related functions
    html.Div(children='''
        Grad Cam Visualizations For Selected Square of Confusion Matrix:
    '''),    

    dcc.Graph(id="grad_cam", figure = go.Figure(go.Image(z=background)), style={"width": "75%", "display": "inline-block"}),
    

    html.Div(children='''
        Choose class to visualize:
    '''),
    html.Div(dcc.Dropdown(
        id="choose_class",
        options=[
            {'label': 'car', 'value': 'car'},
            {'label': 'plane', 'value': 'plane'}
        ],
        clearable=False,
        searchable=False
    ), style={'width': '10%'}),
    html.Div(children='''
        Choose Grad Cam Method:
    '''),   

    html.Div(dcc.Dropdown(
        id="choose_method",
        options=[
            {'label' : 'Grad Cam', 'value': 'grad_cam'},
            {'label': 'Score Cam', 'value': 'score_cam'},
            {'label': 'Saliency map', 'value': 'saliency_map'}
        ],
        clearable=False,
        searchable=False
    ), style={'width': '10%'}),  
    html.Div(children='''
        Choose Grad Cam Method:
    '''),
    # TODO implement this. We 
    html.Button('Next', id='next', n_clicks=0),  
])


# ******************* CALLBACK FUNCTIONS **********************************
@app.callback(
    Output("class_out", "children"),
    Input('submit-classes', 'n_clicks'),
    State("classes", "value"),
    State("number_of_images", "value")
)
def output_classes(clicks, class_str, num_image):
    if class_str != None:
        global classes
        classes = class_str.split(",")
        number_of_image=int(num_image)
        class_str=str(classes)
        # ****************fetching data*****************
        print("Creating dataset for the following classes: " + class_str + " with "+ num_image + " images per class ")
        scrape_data(classes, number_of_image)
        return "Dataset creation finished"
    else:
        return ""

@app.callback(
    Output('chosen_model', 'children'),
    Output('train_loss', 'children'),
    Output('train_acc', 'children'),
    Output('test_loss', 'children'),
    Output('test_acc', 'children'),
    Output("loading-output", "children"),
    Input('submit-model', 'n_clicks'),
    State("model_loading", "children"),
    State('model', 'value'),
    State('lr', 'value'),
    State('epochs', 'value'),
    State('batch_size', 'value')
)
def fetch_model(n_clicks, loading, model, lr, epochs, batch_size):

    # ****************model setup*****************
    inputs = [model, classes, lr, epochs, batch_size]
    print(inputs)
    if None not in inputs:
        id = uuid.uuid4().hex
        global classifier
        classifier, train_history, test_history = train(id, model, classes, lr, int(epochs), int(batch_size))
        return ("Model ID: " + id, 
                "Train losses over epochs: " + str([round(loss, 3) for loss in train_history.history['loss']]),
                "Train accuracy over epochs: " + str([round(acc, 3) for acc in train_history.history['acc']]),
                "Test loss: " + str(round(test_history[0], 3)),
                "Test accuracy: " + str(round(test_history[1], 3)),
                loading)

    return "", "", "", "", "", None

@app.callback(
    Output("matrix", "figure"), 
    [Input("medals", "value")]
    )
def filter_heatmap(cols):
    fig = px.imshow(matrix_df[cols])
    return fig

@app.callback(
    Output("cm_element", 'children'),
    [Input('matrix', 'clickData')],
    )
def display_element(element):
    print(app.get_asset_url("machine_learning.png"))
    print(os.getcwd())
    if element is not None:
        return u'{} have {} {} medals'.format(element['points'][0]['y'], element['points'][0]['z'], element['points'][0]['x'])
    else:
        return "no element selected"

@app.callback(
    Output(component_id="grad_cam", component_property='figure'),
    Input(component_id="choose_class", component_property='value'),
    Input(component_id="choose_method", component_property = 'value')
)
def display_grad_cam_image(classification_class, vis_method):
    if classification_class == 'nothing' or vis_method == 'nothing':
        return go.Figure(go.Image(z=background))

    # unpack variables
    model_path = model_path_wrapper[0]
    # model = model_wrapper[0]
    if "mobilenet" in model_path:
        preprocessing_function = tf.keras.applications.mobilenet.preprocess_input
    else:
        raise(Exception("Not implemented error"))

    #import pdb; pdb.set_trace()
    if vis_method == "grad_cam":
        cam_method = generate_grad_cam_map
    else:
        raise(Exception("Not implemented error"))

    image_path = images[image_idx_wrapper[0]]
    print(image_path)
    original_image = Image.open(image_path)
    image = np.array(original_image.resize((224, 224)))
    print(image.shape)
    image = preprocessing_function(image)

    #vis = cam_method(model, image, 0)
    return go.Figure(go.Image(z=original_image))

if __name__ == '__main__':
    app.run_server(debug=True)