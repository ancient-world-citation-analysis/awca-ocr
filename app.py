# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from logging import fatal
import os
from sys import byteorder
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from tesseract_manager import Text
import tempfile
import base64

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'style.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def getSlider(language: str, minimum=0, maximum=4):
    return html.Div(children=[
        html.Label(language), 
        dcc.Slider(
            id='{}-slider'.format(language),
            min=minimum,
            max=maximum,
            value=1,
            marks={str(i): str(i) for i in range(minimum, maximum + 1)}
        )
    ], className='lang-slider')

def getOptions():
    return html.Div(children=[
        *[
            getSlider(language)
            for language in Text.global_possible_languages
        ]
    ], className='ocr-options')

app.layout = html.Div(children=[
    getOptions(),
    dcc.Upload(
        id='ocr-upload',
        children=html.Div([
            'Drag and drop or ',
            html.A('select a PDF')
        ])
    ),
    html.Div(id='ocr-output')
])

@app.callback(
    Output('ocr-output', 'children'),
    Input('ocr-upload', 'contents'),
    State('ocr-upload', 'filename'),
    State('ocr-upload', 'last_modified')
)
def update_output(contents, names, dates):
    if contents:
        root = tempfile.gettempdir()
        src = os.path.join(root, 'src.pdf')
        if os.path.exists(src):
            os.remove(src)
        print(src)
        print(contents)
        print(base64.b64decode(contents))
        with open(src, 'wb') as f:
            f.write(base64.b64decode(contents))
        out = os.path.join(root, 'out')
        text = Text(src, out)
        text.save_ocr()
        return ', '.join(os.listdir(out))

if __name__ == '__main__':
    app.run_server(debug=True)