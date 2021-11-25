import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tensorflow.keras.applications import mobilenet
from tensorflow.python.keras.engine.training_utils import prepare_loss_functions
from utils.keras_vis_helper import generate_grad_cam_map, generate_saliency_map, generate_scorecam_map
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import gc

# This line allows it to work on windows
import sys
sys.path.insert(0, ".")

app = dash.Dash(__name__)


matrix_df = px.data.medals_wide(indexed=True)
background = Image.open("graphics/grad_cam_place_holder.jpg") # This is a placeholder image. We dont have to change this


# Global variables for grad cam. TODO THESE CANNOT BE HARDCODED
model_path_wrapper = ["airplane_automobile_bird_cat_deer_dog_frog_horse_ship_truck_mobilenet_model.h5"]  # model path has to be declared within list structure
model_wrapper = [tf.keras.models.load_model(model_path_wrapper[0])] # model has to be declared in a list structure
# TODO these MUST point to the test or preferably the validation set
images = ["dataset/airplane/0000.jpg", "dataset/airplane/0001.jpg", "dataset/deer/0000.jpg"] # subset of images found clicking on square of conf. matrix
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

    dcc.Graph(id="grad_cam", style={"width": "100%", "display": "inline-block"}),
    

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

    html.Button('Next', id='next', n_clicks=0),  
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
    Input(component_id="next", component_property='n_clicks')
)
def display_grad_cam_image(classification_class:int, vis_method:str, n_clicks:int):
    """Updates the grad cam image based on the chosen classification_class and vis_method

    The chosen image depends on what square of the confusion matrix the user clicks on as well
    as how many times the user has pressed next.

    TODO
    -Stop hardcoding model_path, model, images, 

    Args:
        classification_class (int): 
        vis_method (str): [description]
        n_clicks (int): [description]

    Returns:
        go.Figure: An updated figure showing an image with cam visualization.
    """
    # Do not run this method if the text boxes have not been populated
    if classification_class is None or vis_method is None:
        return go.Figure(go.Image(z=background))
    
    image_idx_wrapper[0] = n_clicks % len(images)
    #Unpack model 
    model_path = model_path_wrapper[0]
    model = model_wrapper[0]

    # Use model path name to determine type of model
    if "mobilenet" in model_path:
        preprocessing_function = tf.keras.applications.mobilenet.preprocess_input
    elif "vgg16" in model_path:
       preprocessing_function = tf.keras.applications.vgg16.preprocess_input
    else:
        raise(Exception(model_path, "does not have a registered preprocessing_function"))

    # Unpack visualization method for cam
    if vis_method == "grad_cam":
        cam_method = generate_grad_cam_map
    elif vis_method == "score_cam":
        cam_method = generate_scorecam_map
    elif vis_method == "saliency_map":
        cam_method = generate_saliency_map
    else:
        raise(Exception("Vis mehod", vis_method, "not recognized"))

    # determine image path and open image
    image_path = images[image_idx_wrapper[0]]
    original_image = Image.open(image_path)
    image = np.array(original_image.resize((224, 224)))
    image = preprocessing_function(image)

    # apply grad cam
    heatmap = cam_method(model, image, classification_class )
    
    # Create heatmap using matplotlib
    fig = plt.figure()
    if vis_method == "saliency_map":
        plt.imshow(cv2.resize(heatmap, original_image.size), cmap='jet')
    else:
        plt.imshow(original_image)
        plt.imshow(cv2.resize(heatmap, original_image.size), cmap='jet', alpha=.5)  # usa a heatmap but with original res

    # extract heatmap from matplotlib
    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    grad_cam_image = data.reshape(canvas.get_width_height()[::-1] + (3,))  

    # we do not actually want to plot anything using matplotlib. So we clear the figure
    fig.clear()
    plt.close()
    plt.cla()
    plt.clf()
    del(fig)
    gc.collect()

    fig =  go.Figure(go.Image(z=grad_cam_image))
    fig.update_layout(transition_duration=500)
    return fig

@app.callback(
    Output(component_id="choose_class", component_property='options'),
    Input(component_id = "submit-classes", component_property="n_clicks"),
    State('classes', 'value')
)
def set_class_options(n_clicks:int, class_string:str):
    """When the user chooses classes to build this method will be called. It automatically 
    populates the drop-down menu in the grad cam section. 
    
    Args:
        n_clicks int: How many clicks were registered. The program just verifies that n_clicks != 0 before executing
        class_string str: This is the raw string of the classes in comma format

    Returns:
        list: drop down menu options. Each element is of the form {label: str: value: str}
    """
    if n_clicks == 0:
        return [] # if there are no clicks an options menu should not be returned

    # process the class options into a valid drop down meno
    class_options = []
    clean_class_string = class_string.lower().replace(" ", "")  # remove whitecase and lowercase
    classes = clean_class_string.split(",")
    for class_idx, class_ in enumerate(classes):
        class_options.append({"label": class_, 'value': class_idx})
    return class_options

if __name__ == '__main__':
    app.run_server(debug=True)