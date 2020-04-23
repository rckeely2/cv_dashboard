import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_table
from dash.dependencies import Input, Output

import fetch_data

import json
import urllib.request
import sys
import os
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bisect
import math
from bs4 import BeautifulSoup

# def create_layout():
#     return

full_df, cv_merged_df, iso_codes_df, indicator_df = fetch_data.fetch_all(purge=False)
#snapshot = fetch_data.get_latest(full_df)

app = dash.Dash(__name__)
# app.layout = html.Div(className="container",
#                                 children=[
#                                     #snapshot.shape[0],
#                                     html.Div(className="render_div", children=
#                                             dbc.Table.from_dataframe(df=snapshot,
#                                             id="main_table"))])

#column_set = ["deaths_total", "confirmed_total",
#             "deaths_total_norm", "confirmed_total_norm", ]

# full_df.head()

# print(full_df.loc[full_df["Country/Region"]=='United States'])

# def map_columns(column_names):
#     pass

app.layout = html.Div(className="container",children=
    [
        dbc.Row(html.Div(className="masthead", children=["masthead here"])),
        dbc.Row(html.Div(className="bodyDiv", children=
            [
                # Sidepanel
                dbc.Col(html.Div(className="sidepanel",children=[
                    #"testchild"
                    html.Div(className="subcomponent", children=[
                        html.H3('Selected Columns:', className="subcomponent"),
                        dcc.Dropdown(
                            id='selected_columns',
                            options=[{'label': colname, 'value': idx} for idx, colname in enumerate(full_df.columns)],
                            #value=[{'label':'deaths' 'value':1} for idx, colname],
                            value=[5,3,7,8,11,12,16,22,13,17,23,14,24,15,25],
                            placeholder='Select Columns',
                            multi=True,
                            className="side_controls"
                        ),
                    ]),
                    #html.Div(id="testList", children=[", ".join(full_df.columns)])
                    #html.Div(id="testList", children=[" ".join([f"{idx}:{name}," for idx, name in enumerate(full_df.columns)])])
                    # html.Div(className="subcomponent", id="testList", children=
                    # [
                    #     html.Br(),
                    #     ", ".join(column_set)
                    # ]
                    # )
                    #column_set

                    #[" ]
                ])),

                # Mainpanel
                dbc.Col(html.Div(className="mainpanel",children=
                [
                    html.Div(className="graphCont", children=
                    [
                        "this is where the graphs go"
                    ]),
                    html.Div(className="tableCont",children=
                    [
                        dash_table.DataTable(
                        id='table',
                        columns=[{"name": i, "id": i} for i in full_df.columns],
                        #columns=[],
                        data=full_df.to_dict('records'),
                        editable=True,
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        column_selectable="single",
                        row_selectable="multi",
                        selected_columns=[],
                        selected_rows=[],
                        page_action="native",
                        page_size=10,
                        style_table={'overflowX': 'scroll', 'padding': '10px'},
                        ),
                        html.Div(id="datatable-interactivity-container")
                    ]),
                    html.Div()#[full_df.head()])
                ])),
        ])),
        dbc.Row(html.Div(className="footer", children=["footer"])),
    ])

@app.callback(
    Output(component_id='table', component_property='columns'),
    [Input(component_id='selected_columns', component_property='value')]
    )
def update_tableColumns(input_value):
    if len(input_value) != 0:
        return [{"name": i, "id": i} for i in full_df.columns[input_value]]
    else:
        return [{"name": i, "id": i} for i in [full_df.columns[0]]]
    # if len(input_value) == 1:
    #     return input_value
    # else:
    #     return ", ".join(input_value)



# @app.callback(
#     Output('datatable-interactivity', 'style_data_conditional'),
#     [Input('datatable-interactivity', 'selected_columns')]
# )
# def update_styles(selected_columns):
#     return [{
#         'if': { 'column_id': i },
#         'background_color': '#D2F3FF'
#     } for i in selected_columns]


if __name__ == "__main__":
    app.run_server(port=8050, host="127.0.0.1", debug=True)
