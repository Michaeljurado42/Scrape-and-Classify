from operator import truth
import pandas as pd
import numpy as np
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import cv2
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
from transfer_learning import full_pipeline
import uuid
from textwrap import dedent
import base64
import os
import gc
import dash_core_components as dc
from urllib.parse import quote as urlquote
from flask import Flask, send_from_directory
from zipfile import ZipFile
import shutil


UPLOAD_DIRECTORY = "app_uploaded_zip"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

server = Flask(__name__)
app = dash.Dash(server=server,external_stylesheets=[dbc.themes.BOOTSTRAP])
#

@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)

matrix_df = px.data.medals_wide(indexed=True)

#Global variables
classes = os.listdir("./dataset/") if os.path.isdir("./dataset/") else []
classes = [name for name in classes if not name.startswith(('train', 'val', 'test'))]
list_of_images = [] 
num_imgs = [0]
curr_num_img = 0


# Global variables for grad cam
background = Image.open("graphics/stand_in_grad.jpg")
model_path_wrapper = []
model_wrapper = []
confusion_matrix_image_subset = [] # TODO: Set to confusion matrix images
confusion_matrix_image_mapping = {}
image_idx_wrapper = [0]


app.layout = html.Div(children=[
    # Header
    html.H1(children='Confused Guide To Image Classification'),

    # Read in classes
    html.Div(children='''
        Choose classes to build a model around:
    '''),
    dcc.Input(
        id="classes",
        type="text", #"Found these current classes:" + ",".join(classes) if len(classes) > 0 else 
        placeholder="Input comma seperated class names"
    ),
    dcc.Input(
        id="number_of_images",
        type="text",
        placeholder="number of images per class",
        style={'marginLeft':'20px'}
    ),
    html.Button('Create', id='submit-classes', n_clicks=0),
    dc.Upload(
        id="upload-data", 
        children = html.Div(html.Button('Upload a Dataset zip file', id='submit-file', n_clicks=0)),multiple=True,
    ),
    html.Div(
        id="class_out",
        children="",
    ),
    html.Div(id="file_out"),
    dbc.Button("Clean Data", id="button"),
    dbc.Modal([
        dbc.ModalHeader("Would you like to keep this image?"),
        dbc.ModalBody(html.Img(src=None, id = 'curr_img', style={"width": "100%"})),
        dbc.ModalFooter([
            dbc.Button("No", id="no", className="ms-auto", n_clicks=0),
            dbc.Button("Yes", id="yes", className="ms-auto", n_clicks=0)
            ]),
        ],
        id="modal",
        is_open=False,
    ),

    # Select which model
    html.Div(children='''
        Choose which model to train:
    '''),
    html.Div([html.Div(dcc.Dropdown(
        id="model",
        options=[
            {'label': 'MobileNet', 'value': 'mn'},
            {'label': 'VGG16', 'value': 'vgg16'},
        ],
        value='mn',
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
    html.P("Sample Confusion Matrix:", id = "confusion_matrix_title"),
    dcc.Checklist(
        id='class_labels',
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
        options=[{"label": class_, 'value': class_idx} for class_idx, class_ in enumerate(classes)] if classes is not None else None ,
        clearable=False,
        searchable=False
    ), style={'width': '10%'}),
    html.Div(children='''
        Choose Grad Cam Method:
    '''),   

    html.Div(dcc.Dropdown(
        id="choose_method",
        options=[
            {'label': 'None', 'value': 'none'},
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


def add_dataset_to_list_of_images():
    for subdir, dirs, files in os.walk("dataset"):
        subdir= subdir.replace("\\", "/")
        for file in files:
            list_of_images.append("/".join([subdir, file]))
            num_imgs[0] +=1

# ******************* CALLBACK FUNCTIONS **********************************
@app.callback(
    Output("class_out", "children"),
    Input('submit-classes', 'n_clicks'),
    State("classes", "value"),
    State("number_of_images", "value")
)
def output_classes(clicks, class_str, num_image):
    if class_str != None:
        # clean class str
        class_str = class_str.lower()

        global classes
        classes = []
        classes = class_str.split(",")
        classes = [c.strip() for c in classes] # remove whitespace
        number_of_image=int(num_image)
        class_str=str(classes)
        # ****************fetching data*****************
        print("Creating dataset for the following classes: " + class_str + " with "+ num_image + " images per class ")
        scrape_data(classes, number_of_image)

        add_dataset_to_list_of_images()
        return "Dataset created with the following classes: " + ",".join(classes)
    else:
        return ""

def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def uploaded_files():
    """List the files in the upload directory."""
    print("Fetching Files ...")
    files = []
    count = 0
    global classes
    classes = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
            with ZipFile(path, 'r') as zipObj:
                #  zipObj.extractall('Dataset')
                for zip_info in zipObj.infolist():
                    count = count +1 

                    # if count != 1:                  
                    if zip_info.filename[-1] == '/':
                        continue
                    # zip_info.filename = os.path.basename(zip_info.filename)
                    zip_filename = os.path.split(zip_info.filename)
                    subfolder = zip_filename[0].split('/')
                    if subfolder[1] not in classes:
                        classes.append(subfolder[1])
                    zip_info.filename= os.path.join(subfolder[-1],zip_filename[1])
                    # input(zip_info.filename)
                    zipObj.extract(zip_info, 'dataset')

    return files


@app.callback(
    Output("file_out", "children"),
    Output("classes", "value"),
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
def update_output(uploaded_filenames, uploaded_file_contents):
    # """Save uploaded files and regenerate the file list."""
    # files = uploaded_files()

    files = []
    if uploaded_filenames is not None and uploaded_file_contents is not None:
        os.path.isdir("dataset")
        shutil.rmtree("dataset")
        os.mkdir("dataset")
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            save_file(name, data)
        files = uploaded_files()
    if len(files) != 0:
        add_dataset_to_list_of_images()
        return "Zip file successfully uploaded and extracted with given classes: " + ",".join(classes), ",".join(classes)
    return "", ""

# *********************************************************************

@app.callback(
    Output("modal", "is_open"),
    Output("curr_img", "src"),
    [Input("button", "n_clicks"), Input("yes", "n_clicks"), Input("no", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, n3, is_open):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if is_open:
        if 'no' in changed_id:
            if (n2+n3) < num_imgs[0]:
                print(n2+n3)
                current_img1 = list_of_images[n2+n3]
                test_base641 = base64.b64encode(open(current_img1, 'rb').read()).decode('ascii')
                print(list_of_images[n2+n3-1])
                os.remove(list_of_images[n2+n3-1])
                return True, 'data:image/png;base64,{}'.format(test_base641)
        elif 'yes' in changed_id:
            if (n2+n3) < num_imgs[0]:
                print(n2+n3)
                current_img2 = list_of_images[n2+n3]
                test_base642 = base64.b64encode(open(current_img2, 'rb').read()).decode('ascii')
                return True, 'data:image/png;base64,{}'.format(test_base642)
    elif n1:
        if (n2+n3+1 == num_imgs):
            return False, None
        print(n2+n3)
        current_img = list_of_images[n2+n3]
        test_base640 = base64.b64encode(open(current_img, 'rb').read()).decode('ascii')
        return not is_open, 'data:image/png;base64,{}'.format(test_base640)
    return is_open, None

@app.callback(
    Output('chosen_model', 'children'),
    Output('train_loss', 'children'),
    Output('train_acc', 'children'),
    Output('test_loss', 'children'),
    Output('test_acc', 'children'),
    Output("loading-output", "children"),
    Output("matrix", "figure"),
    Output("class_labels", "value"),
    Output("confusion_matrix_title", "children"),
    Input('submit-model', 'n_clicks'),
    State("model_loading", "children"),
    State('model', 'value'),
    State('lr', 'value'),
    State('epochs', 'value'),
    State('batch_size', 'value'),
)
def fetch_model(n_clicks, loading, model, lr, epochs, batch_size):

    # ****************model setup*****************
    inputs = [model, classes, lr, epochs, batch_size]
    print("Is it really prining this?", inputs)
    if None not in inputs:
        classifier, train_history, test_history, test_df, conf_matrix_mapping_update = full_pipeline(model, classes, lr, int(epochs), int(batch_size))
        
        confusion_matrix_image_mapping.update(conf_matrix_mapping_update) # this contains mappings between imagenames and squares of conf matrix
        print(confusion_matrix_image_mapping) 

        cols = test_df.columns.tolist()
        class_counts = test_df.sum(axis=1).tolist()
        test_df = test_df.T.div(class_counts, 1).T
        print(test_df)
        fig = px.imshow(test_df[cols], 0, 1, labels={"x": "Prediction", "y": "Ground Truth", "color": "Proportion of Ground Truth"})
        print("Displaying Model Results")
        
        confusion_matrix_image_subset.extend(list_of_images)

        if len(model_path_wrapper) == 0:
            model_path_wrapper.append(classifier)
            model_wrapper.append(tf.keras.models.load_model(classifier))
        else:
            model_path_wrapper[0] = classifier
            model_wrapper[0] = tf.keras.models.load_model(classifier)
        return ("Model ID: " + classifier.split(".")[0], 
                "Train losses over epochs: " + str([round(loss, 3) for loss in train_history.history['loss']]),
                "Train accuracy over epochs: " + str([round(acc, 3) for acc in train_history.history['accuracy']]),
                "Test loss: " + str(round(test_history[0], 3)),
                "Test accuracy: " + str(round(test_history[1], 3)),
                loading,
                fig,
                cols,
                "Confusion Matrix With Classes: " + " ".join(classes))
    
    values = matrix_df.columns.tolist()
    fig = px.imshow(matrix_df[values])

    return "", "", "", "", "", None, fig, values, "Sample Confusion Matrix:"

@app.callback(
    Output("cm_element", 'children'),
    Output("choose_class", "value"),
    Output("choose_method", "value"),
    [Input('matrix', 'clickData')],
    )
def display_element(element):
    if element is not None:
        prediction = element['points'][0]['x']
        truth_value = element['points'][0]['y']

        # now 
        confusion_matrix_image_subset.clear()
        confusion_matrix_image_subset.extend(confusion_matrix_image_mapping[(prediction, truth_value)])
        return u'{}, {}, {}'.format(prediction, truth_value, element['points'][0]['z']), truth_value, "none"
    else:
        return "", "", ""

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
    Args:
        classification_class (int): 
        vis_method (str): [description]
        n_clicks (int): [description]
    Returns:
        go.Figure: An updated figure showing an image with cam visualization.
    """
    images = confusion_matrix_image_subset # this should be set to a square of the confusion matrix
    # Do not run this method if the text boxes have not been populated
    if classification_class is None or vis_method is None or len(images) == 0:
        return go.Figure(go.Image(z=background))
    
    image_idx_wrapper[0] = n_clicks % len(images)
    #Unpack model
    model_path = model_path_wrapper[0]
    model = model_wrapper[0]

    # determine image path and open image
    image_path = images[image_idx_wrapper[0]]
    original_image = Image.open(image_path)
    image = np.array(original_image.resize((224, 224)))
    if vis_method == "none":
        fig =  go.Figure(go.Image(z=original_image))
        fig.update_layout(transition_duration=100)
        return fig

    # Use model path name to determine type of model
    if "mn" in model_path:
        preprocessing_function = tf.keras.applications.mobilenet.preprocess_input
    elif "vgg16" in model_path:
       preprocessing_function = tf.keras.applications.vgg16.preprocess_input
    else:
        raise(Exception(model_path, "does not have a registered preprocessing_function"))

    image = preprocessing_function(image)

    # Unpack visualization method for cam
    if vis_method == "grad_cam":
        cam_method = generate_grad_cam_map
    elif vis_method == "score_cam":
        cam_method = generate_scorecam_map
    elif vis_method == "saliency_map":
        cam_method = generate_saliency_map
    else:
        raise(Exception("Vis mehod", vis_method, "not recognized"))


    # apply grad cam
    heatmap = cam_method(model, image, classification_class)
    
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
    fig.update_layout(transition_duration=100)
    return fig

@app.callback(
    Output(component_id="choose_class", component_property='options'),
    Input(component_id = "submit-classes", component_property="n_clicks"),
    Input('file_out', 'children')
)
def set_class_options(n_clicks:int, file_clicks:int):
    """When the user chooses classes to build this method will be called. It automatically 
    populates the drop-down menu in the grad cam section. 
    
    Args:
        n_clicks int: How many clicks were registered. The program just verifies that n_clicks != 0 before executing
        class_string str: This is the raw string of the classes in comma format
    Returns:
        list: drop down menu options. Each element is of the form {label: str: value: str}
    """

    # process the class options into a valid drop down meno
    class_options = []
    global classes
    for class_idx, class_ in enumerate(classes):
        class_options.append({"label": class_, 'value': class_idx})
    return class_options

if __name__ == '__main__':
    app.run_server(debug=True)