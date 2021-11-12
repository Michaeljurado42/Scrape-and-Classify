import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)

bar_df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

matrix_df = px.data.medals_wide(indexed=True)

fig = px.bar(bar_df, x="Fruit", y="Amount", color="City", barmode="group")

app.layout = html.Div(children=[
    # Header
    html.H1(children='Sample App with Plotly'),

    html.Div(children='''
        Visualize our data
    '''),

    # Bar Graph
    dcc.Graph(
        id='example-graph',
        figure=fig
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
    )
])

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