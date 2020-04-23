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

default_countries = ['Spain', 'Italy', 'France', 'Germany', 'United States', 'China',
                'United Kingdom', 'Korea, Rep.']
plot_vars = ['Confirmed', 'Deaths', 'Recovered', 'Active']
rmean_options = ['None', '3', '5', '7']

def generate_plot_var(cv_variable, normalise, cumulative):
    norm_str = ""
    if normalise == "normalise":
        norm_str = ", normalised"
    elif normalise == "percent":
        norm_str = ", percent"

    if cumulative:
        typeStr = 'Total'
    else:
        typeStr = 'Daily'
    var_str = f"{plot_vars[cv_variable]} ({typeStr}{norm_str})"
    return var_str

def reverse_lookup_col_idx(search_col, search_list):
    s = pd.Series(full_df[search_col].unique()).isin(search_list)
    return list(s[s].index)

def apply_rmean(series, rmean):
    if rmean == 0:
        return series
    else:
        rmean = int(rmean_options[rmean])
        series = pd.Series(series).rolling(window=rmean).mean()
        return series

app.layout = html.Div(className="container",children=
    [
        # dbc.Row(html.Div(className="masthead", children=
        # [
        #     "masthead here"
        # ])),
        dbc.Row(html.Div(
        [
            html.H1(id="testbox", children=["testbox"]),
            html.H3('Cross country comparisons'),

            dbc.Col(className="graph_controls", children=[
                html.P(className="graph_controls", children='Y Scale'),
                dcc.RadioItems(options=[{'label':'Linear', 'value':'linear'},
                                        {'label':'Log', 'value':'log'},],
                               value='linear',
                               id='yscale_rb',
                               labelStyle={'display': 'inline-block'}),
                ]),
            dbc.Col(className="graph_controls", children=[
                html.P(className="graph_controls", children='Normalisation'),
                dcc.RadioItems(options=[{'label':'Simple', 'value':'simple'},
                                        {'label':'Normalise', 'value':'normalise'},
                                        {'label':'Percent', 'value':'percent'}],
                               value='simple',
                               id='normalise',
                               labelStyle={'display': 'inline-block'}),
                ]),
            dbc.Col(className="graph_controls", children=[
                html.P(className="graph_controls", children='Rolling mean'),
                dcc.Dropdown(
                    id='rollingMean',
                    options=[{'label': item, 'value': idx } for idx, item in enumerate(rmean_options)],
                    #value=[{'label':'deaths' 'value':1} for idx, colname],
                    value=0,
                    placeholder='Select Countries',
                    multi=False,
                    className="dropdown"
                )
                ]),
            dbc.Col(className="graph_controls", children=[
                html.P('Countries:'),
                dcc.Dropdown(
                    id='country_names',
                    options=[{'label': country_name, 'value': idx} for idx, country_name in enumerate(full_df['Name'].unique())],
                    #value=[{'label':'deaths' 'value':1} for idx, colname],
                    value=reverse_lookup_col_idx('Name', default_countries),
                    placeholder='Select Countries',
                    multi=True,
                    className="dropdown"
                ),]),
            dbc.Col(className="graph_controls", children=[
                html.P(className="graph_controls", children='Variables'),
                dcc.Dropdown(
                    id='cv_variables',
                    options=[{'label': country_name, 'value': idx} for idx, country_name in enumerate(plot_vars)],
                    #value=[{'label':'deaths' 'value':1} for idx, colname],
                    value=0,
                    placeholder='Select Variables',
                    multi=False,
                    className="dropdown"
                ),]
            ),
            dcc.Graph(id='topGraph',
                    figure={'data': [ dict(
                                        x = full_df[full_df['Name']==country]['Date'],
                                        y = full_df[full_df['Name']==country][generate_plot_var(0, 'disable', True)],
                                        #'text': ['a', 'b', 'c', 'd'],
                                        #'customdata': ['c.a', 'c.b', 'c.c', 'c.d'],
                                        name =  country,
                                        mode = 'line',
                                        marker =  {'size': 10}
                                    ) for country in default_countries
                                    ],
                            'layout': dict(
                                clickmode='event+select',
                                xaxis={'type': 'date', 'title': 'time'},
                                yaxis={'type': 'linear', 'title':generate_plot_var(0, 'disable', True)},
                                title="Cumulative"
                            )
                            }
                    ),
            #dcc.H3("BottomGraph"),
            dcc.Graph(id='bottomGraph',
                    figure={'data': [
                                dict(
                                    x = full_df[full_df['Name']==country]['Date'],
                                    y = full_df[full_df['Name']==country][generate_plot_var(0, 'disable', False)],
                                    #'text': ['a', 'b', 'c', 'd'],
                                    #'customdata': ['c.a', 'c.b', 'c.c', 'c.d'],
                                    name =  country,
                                    mode = 'line',
                                    marker =  {'size': 10}
                                ) for country in default_countries
                            ],
                            'layout': dict(
                                clickmode='event+select',
                                xaxis={'type': 'date', 'title': 'time'},
                                yaxis={'type': 'linear', 'title':generate_plot_var(0, 'disable', False)}
                            )
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
    Output(component_id='topGraph', component_property='figure'),
    [Input(component_id='country_names', component_property='value'),
     Input(component_id='cv_variables', component_property='value'),
     Input(component_id='yscale_rb', component_property='value'),
     Input(component_id='normalise', component_property='value'),
     Input(component_id='rollingMean', component_property='value')]
    )
def update_topGraph(countries, cv_variable, yscale, normalise, rmean):
    country_list_l = full_df['Name'].unique()[countries]
    plot_var = generate_plot_var(cv_variable, normalise, True)
    #plot_var = plot_vars[cv_variable]
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
            ],
            'layout': dict(
                clickmode='event+select',
                xaxis={'title': 'time'},
                yaxis={'type': yscale, 'title':plot_var}
            )
        }
    return figure

@app.callback(
    Output(component_id='testbox', component_property='children'),
    [Input(component_id='country_names', component_property='value'),
     Input(component_id='cv_variables', component_property='value'),
     Input(component_id='yscale_rb', component_property='value'),
     Input(component_id='normalise', component_property='value'),
     Input(component_id='rollingMean', component_property='value')]
    )
def update_testbox(countries, cv_variable, yscale, normalise, rmean):
    return generate_plot_var(cv_variable, normalise, True)

@app.callback(
    Output(component_id='bottomGraph', component_property='figure'),
    [Input(component_id='country_names', component_property='value'),
     Input(component_id='cv_variables', component_property='value'),
     Input(component_id='yscale_rb', component_property='value'),
     Input(component_id='normalise', component_property='value'),
     Input(component_id='rollingMean', component_property='value')]
    )
def update_bottomGraph(countries, cv_variable, yscale, normalise, rmean):
    country_list_l = full_df['Name'].unique()[countries]
    plot_var = generate_plot_var(cv_variable, normalise, False)
    #plot_vars[cv_variable]

    figure={'data': [
                dict(
                    x = full_df[full_df['Name']==country]['Date'],
                    y = apply_rmean(full_df[full_df['Name']==country][plot_var], rmean),
                    #'text': ['a', 'b', 'c', 'd'],
                    #'customdata': ['c.a', 'c.b', 'c.c', 'c.d'],
                    name =  country,
                    mode = 'line',
                    marker =  {'size': 10}
                ) for country in country_list_l
            ],
            'layout': dict(
                clickmode='event+select',
                xaxis={'title': 'time'},
                yaxis={'type': yscale, 'title':f"{plot_var} rmean:{rmean_options[rmean]}"}
            )
        }
    return figure

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
