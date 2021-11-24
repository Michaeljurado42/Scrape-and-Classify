import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tensorflow.python.keras.engine.training_utils import prepare_loss_functions
from utils.keras_vis_helper import generate_grad_cam_map, generate_saliency_map, generate_scorecam_map
import tensorflow as tf
from PIL import Image
import numpy as np

app = dash.Dash(__name__)

# Global variables to update
matrix_df = px.data.medals_wide(indexed=True)
background = plt.imread("graphics/machine_learning.png")
model = tf.keras.models.load_model("airplane_automobile_bird_cat_deer_dog_frog_horse_ship_truck_mobilenet_model.h5")
preprocessing_function =  tf.keras.applications.mobilenet.preprocess_input
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
    html.Button('Create', id='submit-classes', n_clicks=0),
    html.Div(
        id="class_out",
        children="",
    ),

    # Select which model
    html.Div(children='''
        Choose which model to train:
    '''),
    html.Div(dcc.Dropdown(
        id="model",
        options=[
            {'label': 'MobileNetV2', 'value': 'mnv2'},
        ],
        value='mnv2',
        clearable=False,
        searchable=False
    ), style={'width': '30%'}),
    html.Div(id="chosen_model",
        children="",
    ),

    # Select Hyperparameters
    html.Div(children='''
        Set hyperparameter values:
    '''),
    html.Div(dcc.Slider(
        id="lr",
        min=0,
        max=1,
        step=.005,
        tooltip={"placement": "bottom", "always_visible": True},
        value=0.01,
    ), style={'width': '30%'}),

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
    dcc.Graph(id="grad_cam", figure = go.Figure(go.Image(z=background)), style={"width": "75%", "display": "inline-block"}, config = {"staticPlot": True}),
    

    html.Div(children='''
        Choose class to visualize:
    '''),
    html.Div(dcc.Dropdown(
        id="choose_class",
        options=[
            {'label': 'car', 'value': 'car'},
            {'label': 'plane', 'value': 'plane'}
        ],
        value='car',
        clearable=False,
        searchable=False
    ), style={'width': '10%'}),
    html.Div(children='''
        Choose Grad Cam Method:
    '''),   

    html.Div(dcc.Dropdown(
        id="choose_method",
        options=[
            {'label': '', 'value': "nothing"},
            {'label' : 'Grad Cam', 'value': 'grad_cam'},
            {'label': 'Score Cam', 'value': 'score_cam'},
            {'label': 'Saliency map', 'value': 'saliency_map'}
        ],
        value = 'nothing',
        clearable=False,
        searchable=False
    ), style={'width': '10%'}),  
    html.Div(children='''
        Choose Grad Cam Method:
    '''),
    # TODO implement this. We 
    html.Button('Next', id='next', n_clicks=0),  
    
    # TODO Next has to update the image path
    dcc.Store(id="image_path", data = "airplane_automobile_bird_cat_deer_dog_frog_horse_ship_truck_mobilenet_model.h5"),
])


# ******************* CALLBACK FUNCTIONS **********************************
@app.callback(
    Output("class_out", "children"),
    Input('submit-classes', 'n_clicks'),
    State("classes", "value")
)
def output_classes(clicks, class_str):
    if class_str != None:
        classes = class_str.split(",")

        # ****************implement scraper fetching data*****************
        # scraper(classes)

        return "Creating dataset for the following classes: " + str(classes)
    else:
        return ""

@app.callback(
    Output('chosen_model', 'children'),
    Input('model', 'value'),
    Input('lr', 'value')
)
def fetch_model(model, epochs, lr, batch_size):

    # ****************implement model setup for chosen model*****************
    # model.train()
    # for epoch

    return ""

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
    Input(component_id="choose_method", component_property = 'value'),
    Input(component_id="image_path", component_property = 'data')
)
def display_grad_cam_image(classification_class, classification_method, image_path):
    if classification_class == 'nothing' or not classification_method:
        return
    #if classification_method == "grad_cam":
    cam_method = generate_grad_cam_map
    #elif classification_method == "score_cam":
    #    cam_method = generate_scorecam_map
    #elif classification_method == "saliency_map":
    #    cam_method = generate_saliency_map
    #model = tf.keras.models.load_model(model_path)
    original_image = Image.open(image_path)
    image = np.array(original_image.resize((224, 224)))    
    image = preprocessing_function(image)

    vis = cam_method(model, image, 0)
    return go.Figure(go.Image(z=original_image))

if __name__ == '__main__':
    app.run_server(debug=True)