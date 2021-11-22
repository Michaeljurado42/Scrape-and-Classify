import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)

matrix_df = px.data.medals_wide(indexed=True)

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
    )
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
    if element is not None:
        return u'{} have {} {} medals'.format(element['points'][0]['y'], element['points'][0]['z'], element['points'][0]['x'])
    else:
        return "no element selected"

if __name__ == '__main__':
    app.run_server(debug=True)