#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install jupyter_dash
# !pip install dash_dangerously_set_inner_html


# In[17]:


from jupyter_dash import JupyterDash
from dash import Dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_dangerously_set_inner_html
import requests
import json
import visdcc
import torch
import pandas as pd
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification

external_scripts = [
    'https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js',
    'https://code.jquery.com/ui/1.12.1/jquery-ui.min.js']

app = JupyterDash(__name__, suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.MINTY],
                external_scripts=external_scripts)

app.layout = html.Div([html.Div([
     html.Img(src=app.get_asset_url('logo.jpg'),
                       style={'height':'4%', 'width':'4%',
                              'textAlign': 'center',
                              'padding': '5px',
                              'display': 'inline-block',
                              'margin': '0 0.2em'}),
     html.Div('深度學習判定日文句型：動詞普通型、たり型、のに型、ある/いる型',
                 style={'color': '#454d26',
                       'fontSize': '20px',
                       'textAlign': 'left',
                       'font-weight' :'bold',
                       'display': 'inline-block'})
                        ], id= 'header'),

     html.Div(dbc.Container([
     html.Br(),
        dcc.Textarea(
            id='textarea-example',
            value='春休みは友達と、桜を見たり、ピクニックをしたりするつもりです。',
#            placeholder='在此處輸入要分析的文本。',
            style={'width': '100%', 'height': '300px','text-align': 'left','font-family': 'UD デジタル 教科書体 NK-R'}
        )],id='main')),

    html.Hr(),
                       
    html.Div( 
        dbc.FormGroup([
            html.Span(html.H4(dbc.Badge("句型分類",color="#73BABA", className="mr-2"))), 
            
    html.Div(
        dbc.Checklist(id='pattern',
         options=[{'label': d, 'value': c}
                  for d,c in zip(['動詞普通型','たり型','のに型','ある/いる型'],
                                 ['V','Tari','Noni','IruAru']) ],
             inline=True,
             value=[])
            )
                        ]) ),
                     
    html.Div([       
    html.Button('進行分析', id='submit-val', n_clicks=0),
             
    html.Br(),  
    html.Br(), 
     
    dbc.Card(
        [dbc.CardHeader("分析結果",
                       id='cardheader'),
         dbc.CardBody([html.P("", id='table_header_body')]),
        ],id="cards",
    ),
    
    ],id='main2'),

                       
     html.Div('©2022 Tzu-Yu, Ning', id='footer'),                  
                       
], id='container')


@app.callback(
    Output('table_header_body','children'),
    Input('submit-val', 'n_clicks'),
    State('textarea-example', 'value'),
    State('pattern', 'value'),
)


def update_output(n_clicks, article, pattern_all):
    pattern_id = pattern_all
    print(article,pattern_all)
    data = {"article": article, "pattern_id": pattern_id}
    headers = {'Content-Type': 'application/json'}
    res = requests.post(url='http://163.13.202.232:30501/api/filter_single', headers=headers,
                        data=json.dumps(data))  
    res_json = res.json()
    
    result_V = res_json.get('html_V')
    print(result_V)
    #===========================
    result_Tari = res_json.get('html_Tari')
    print(result_Tari)
    #===========================
    result_Noni = res_json.get('html_Noni')
    print(result_Noni)
    #===========================
    result_IruAru = res_json.get('html_IruAru')
    print(result_IruAru)


    return dash_dangerously_set_inner_html.DangerouslySetInnerHTML(result_V),\
     dash_dangerously_set_inner_html.DangerouslySetInnerHTML(result_Tari),\
     dash_dangerously_set_inner_html.DangerouslySetInnerHTML(result_Noni),\
     dash_dangerously_set_inner_html.DangerouslySetInnerHTML(result_IruAru)


if __name__ == '__main__':
    #app.run_server(debug=True)
    #app.run_server(port=1236)
#     app.run_server(mode='inline', debug=True, port=3033)
#    app.run_server(debug=True, use_reloader=False, port=3033)
     app.run_server(debug=False, use_reloader=False, host='0.0.0.0', port=3050)


# In[ ]:




