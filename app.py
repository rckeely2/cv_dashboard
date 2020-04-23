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

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

country_list = ['Spain', 'Italy', 'France', 'Germany', 'United States', 'China',
                'United Kingdom', 'Korea, Rep.']
plot_vars = ['Confirmed (Total)', 'Deaths (Total)', 'Recovered (Total)',
       'Active (Total)', 'Confirmed (Total, normalised)',
       'Deaths (Total, normalised)']

app.layout = html.Div(className="container",children=
    [
        dbc.Row(html.Div(className="masthead", children=
        [
            "masthead here"
        ])),
        dbc.Row(html.Div(
        [
            html.H1('Progression over time'),
            html.H3('Countries:'),
            dcc.Dropdown(
                id='country_names',
                options=[{'label': country_name, 'value': idx} for idx, country_name in enumerate(full_df['Name'].unique())],
                #value=[{'label':'deaths' 'value':1} for idx, colname],
                value=[137,76,59,55],
                placeholder='Select Countries',
                multi=True,
                className="side_controls"
            ),
            html.H3('Variables'),
            dcc.Dropdown(
                id='cv_variables',
                options=[{'label': country_name, 'value': idx} for idx, country_name in enumerate(plot_vars)],
                #value=[{'label':'deaths' 'value':1} for idx, colname],
                value=0,
                placeholder='Select Variables',
                multi=False,
                className="side_controls"
            ),
            dcc.Graph(id='progressionOverTime',
                    figure={'data': [
                                dict(
                                    x = full_df[full_df['Name']==country]['Date'],
                                    y = full_df[full_df['Name']==country]['Confirmed (Total)'],
                                    #'text': ['a', 'b', 'c', 'd'],
                                    #'customdata': ['c.a', 'c.b', 'c.c', 'c.d'],
                                    name =  country,
                                    mode = 'line',
                                    marker =  {'size': 10}
                                ) for country in country_list
                                # {
                                #     'x': [1, 2, 3, 4],
                                #     'y': [9, 4, 1, 4],
                                #     'text': ['w', 'x', 'y', 'z'],
                                #     'customdata': ['c.w', 'c.x', 'c.y', 'c.z'],
                                #     'name': 'Trace 2',
                                #     'mode': 'line',
                                #     'marker': {'size': 12}
                                # }
                            ],
                            'layout': {
                                'clickmode': 'event+select'
                            }
                        }
                        ),
        ])),
        dbc.Row(html.Div(className="bodyDiv", children=
        [
            html.Div(className="tableCont",children=
            [
                html.H1('Data Table'),
                html.H3('Select Columns:'),
                #html.H3('Selected Columns:', className="subcomponent"),
                dcc.Dropdown(
                    id='table_dropdown_select',
                    options=[{'label': colname, 'value': idx} for idx, colname in enumerate(full_df.columns)],
                    #value=[{'label':'deaths' 'value':1} for idx, colname],
                    value=[5,3,7,8,11,12,16,22,13,17,23,14,24,15,25],
                    placeholder='Select Columns',
                    multi=True,
                    className="side_controls"
                ),
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
                page_size=20,
                style_table={'overflowX': 'scroll', 'padding': '10px'},
                ),
                html.Div(id="datatable-interactivity-container")
            ]),
        ])),
        dbc.Row(html.Div(className="footer", children=["footer"])),
    ])

@app.callback(
    Output(component_id='table', component_property='columns'),
    [Input(component_id='table_dropdown_select', component_property='value')]
    )
def update_tableColumns(input_value):
    if len(input_value) != 0:
        return [{"name": i, "id": i} for i in full_df.columns[input_value]]
    else:
        return [{"name": i, "id": i} for i in [full_df.columns[0]]]

@app.callback(
    Output(component_id='progressionOverTime', component_property='figure'),
    [Input(component_id='country_names', component_property='value'),
     Input(component_id='cv_variables', component_property='value')]
    )
def update_graphCountries(countries, cv_variable):
    country_list_l = full_df['Name'].unique()[countries]
    plot_var = plot_vars[cv_variable]
    figure={'data': [
                dict(
                    x = full_df[full_df['Name']==country]['Date'],
                    y = full_df[full_df['Name']==country][plot_var],
                    #'text': ['a', 'b', 'c', 'd'],
                    #'customdata': ['c.a', 'c.b', 'c.c', 'c.d'],
                    name =  country,
                    mode = 'line',
                    marker =  {'size': 10}
                ) for country in country_list_l
                # {
                #     'x': [1, 2, 3, 4],
                #     'y': [9, 4, 1, 4],
                #     'text': ['w', 'x', 'y', 'z'],
                #     'customdata': ['c.w', 'c.x', 'c.y', 'c.z'],
                #     'name': 'Trace 2',
                #     'mode': 'line',
                #     'marker': {'size': 12}
                # }
            ],
            'layout': {
                'clickmode': 'event+select'
            }
        }
    return figure
    # if len(input_value) != 0:
    #     return [{"name": i, "id": i} for i in full_df.columns[input_value]]
    # else:
    #     return [{"name": i, "id": i} for i in [full_df.columns[0]]]
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
