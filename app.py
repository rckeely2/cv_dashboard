import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_table
from dash.dependencies import Input, Output

import fetch_data

import ast
from urllib.parse import urlparse, parse_qsl, urlencode
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

full_df, cv_merged_df, iso_codes_df, indicator_df = fetch_data.fetch_all(purge=False)

app = dash.Dash("CV Dashboard")
app.config.suppress_callback_exceptions = True

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

def apply_rmean(series, rmean, cumulative):
    if (cumulative or (rmean == 0)):
        return series
    else:
        rmean = int(rmean_options[rmean])
        series = pd.Series(series).rolling(window=rmean).mean()
        return series

def rebase_series(series, threshold=0, ret_idx = False, trim_idx = -1):
    series.reset_index(drop=True,inplace=True)
    if threshold == 0:
        return series
    if (trim_idx == -1):
        trim_idx = bisect.bisect_left(series, threshold)
    trim_series = np.array(list(series[trim_idx:].values) + [np.nan]*trim_idx)
    if ret_idx:
        return trim_series, trim_idx
    else:
        return trim_series

def generate_x(series, threshold):
    if threshold == 0:
        return series
    else:
        return list(range(1,len(series)+1))

def generate_single_data(country, threshold, rmean,
                        cv_variable, normalise, cumulative):
    plot_vars_l = [ generate_plot_var(i, normalise, cumulative) for i in range(len(plot_vars))]
    plot_dict = [dict(
        x = generate_x(full_df[full_df['Name']==country]['Date'],threshold),
        y = rebase_series(apply_rmean(full_df[full_df['Name']==country][plot_var], rmean, cumulative), threshold),
        #y = rebase_series(apply_rmean(full_df[full_df['Name']==country][plot_var], rmean, cumulative),threshold),
        #'text': ['a', 'b', 'c', 'd'],
        #'customdata': ['c.a', 'c.b', 'c.c', 'c.d'],
        name =  plot_var,
        mode = 'line',
        marker =  {'size': 10}
    ) for plot_var in plot_vars_l ]
    return plot_dict

def generate_multi_data(country_list_l, threshold, rmean, cv_variable, normalise, cumulative):
    plot_var = generate_plot_var(cv_variable, normalise, cumulative)
    plot_dict = [dict(
        x = generate_x(full_df[full_df['Name']==country]['Date'],threshold),
        y = rebase_series(apply_rmean(full_df[full_df['Name']==country][plot_var], rmean, cumulative), threshold),
        #y = rebase_series(apply_rmean(full_df[full_df['Name']==country][plot_var], rmean, cumulative),threshold),
        #'text': ['a', 'b', 'c', 'd'],
        #'customdata': ['c.a', 'c.b', 'c.c', 'c.d'],
        name =  country,
        mode = 'line',
        marker =  {'size': 10}
    ) for country in country_list_l]
    return plot_dict

def generate_layout(threshold, rmean, yscale, cv_variable, normalise, cumulative):
    plot_var = generate_plot_var(cv_variable, normalise, cumulative)
    if threshold == 0:
        xaxis_val = {'type': 'date', 'title': 'Date'}
    else:
        #xaxis_val = {'type': 'date', 'title': 'Date'}
        xaxis_val = {'type': 'linear', 'title': f'Days since {threshold}th {plot_var}'}
    layout_dict = dict(
        clickmode='event+select',
        xaxis=xaxis_val,
        yaxis={'type': yscale, 'title':f"{plot_var} rmean:{rmean_options[rmean]} threshold:{threshold}"})
    return layout_dict

def generate_data(country_list_l, threshold, rmean, cv_variable, normalise, cumulative):
    #if len(country_l)
    if len(country_list_l) == 1:
        plot_dict = generate_single_data(country_list_l[0], threshold, rmean, cv_variable, normalise, cumulative)
        #layout_dict = generate_single_layout()
    else:
        plot_dict = generate_multi_data(country_list_l, threshold, rmean, cv_variable, normalise, cumulative)
        #layout_dict = generate_multi_layout()
    return plot_dict

def plot_figure(countries, cv_variable, yscale, normalise, rmean, threshold, cumulative):
    country_list_l = full_df['Name'].unique()[countries]
    figure={'data': generate_data(country_list_l, threshold, rmean, cv_variable, normalise, cumulative),
            'layout': generate_layout(threshold, rmean, yscale, cv_variable, normalise, cumulative)}
    return figure

app.layout = html.Div(className="container",children=
    [
        # dbc.Row(html.Div(className="masthead", children=
        # [
        #     "masthead here"
        # ])),
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-layout')
    ])

def apply_default_value(params):
    def wrapper(func):
        def apply_value(*args, **kwargs):
            if 'id' in kwargs and kwargs['id'] in params:
                if ((component_ids[kwargs['id']]['value_type'] == 'numeric') or \
                    (component_ids[kwargs['id']]['component'] == 'multi_dd')):
                    kwargs['value'] = ast.literal_eval((params[kwargs['id']]))
                # elif :
                #     kwargs['value'] = ast.literal_eval((params[kwargs['id']]))
                else:
                    kwargs['value'] = params[kwargs['id']]
            return func(*args, **kwargs)
        return apply_value
    return wrapper

component_ids = {
    'yscale_rb' : {'component' : 'radioButton', 'value_type': 'text'},
    'normalise' : {'component' : 'radioButton', 'value_type': 'text'},
    'threshold_cumulative' : {'component' : 'single_dd', 'value_type': 'numeric'},
    'threshold_daily' : {'component' : 'single_dd', 'value_type': 'numeric'},
    'rollingMean' : {'component' : 'single_dd', 'value_type': 'numeric'},
    'country_names' : {'component' : 'multi_dd', 'value_type': 'text'},
    'cv_variables' : {'component' : 'single_dd', 'value_type': 'numeric'},
    # 'single_dd' : {'component' : 'single_dd', 'value_type': 'text'},
    # 'multi_dd' : {'component' : 'multi_dd', 'value_type': 'text'},
    # 'input' : {'component' : 'input', 'value_type': 'text'},
    # #'topMap' : {'component' : 'slider', 'value_type': 'numeric'},
    # 'range' : {'component' : 'rangeSlider', 'value_type': 'numeric'},
    # 'radioButton' : {'component' : 'radioButton', 'value_type': 'numeric'}
}

def build_layout(params):
    layout = [

    # dbc.
    # Row 1
    dbc.Row(html.Div(
    [
        #html.H1(id="testbox", children=["testbox"]),
        #html.H3('Cross country comparisons'),
        # html.H2('URL State demo', id='state'),
        # apply_default_value(params)(dcc.Dropdown)(
        #     id='single_dd',
        #     options=[{'label': i, 'value': i} for i in ['LA', 'NYC', 'MTL']],
        #     value='LA',
        #     multi=False
        # ),
        dbc.Col(className="graph_controls", width=2, children=[
            html.Div(className="inline_div", children=[
                html.P(className="graph_controls", children='Y Scale'),
                apply_default_value(params)(dcc.RadioItems)(
                    options=[{'label':'Linear', 'value':'linear'},
                            {'label':'Log', 'value':'log'},],
                   value='linear',
                   id='yscale_rb',
                   labelStyle={'display': 'inline-block'}
                   ),]),
            ]),
        dbc.Col(className="graph_controls", width=4, children=[
            html.P(className="graph_controls", children='Normalisation'),
            apply_default_value(params)(dcc.RadioItems)(
                options=[{'label':'Simple', 'value':'simple'},
                        {'label':'Normalise', 'value':'normalise'},
                        {'label':'Percent', 'value':'percent'}],
                value='simple',
                id='normalise',
                labelStyle={'display': 'inline-block'}
                ),
            ]),
        dbc.Col(className="graph_controls", width=4, children=[
            html.P(className="graph_controls", children='Cumulative Threshold'),
            apply_default_value(params)(dcc.Dropdown)(
                id="threshold_cumulative",
                 options=[{'label':'None', 'value':0},
                          {'label':'100', 'value':100},
                          {'label':'500', 'value':500},
                          {'label':'1000', 'value':1000}],
                 placeholder='Select threshold',
                 value=0,
                 )
            ]),
        ])),
    # Row 2
        dbc.Row(html.Div(
        [
        dbc.Col(className="graph_controls", width=4, children=[
            html.P(className="graph_controls", children='Daily Threshold'),
            apply_default_value(params)(dcc.Dropdown)(
                id="threshold_daily",
                 options=[{'label':'None', 'value':0},
                          {'label':'25', 'value':25},
                          {'label':'100', 'value':100},
                          {'label':'250', 'value':250}],
                 placeholder='Select threshold',
                 value=0,
                 )
            ]),
        dbc.Col(className="graph_controls", width=4, children=[
            html.P(className="graph_controls", children='Rolling mean'),
            apply_default_value(params)(dcc.Dropdown)(
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
            apply_default_value(params)(dcc.Dropdown)(
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
            apply_default_value(params)(dcc.Dropdown)(
                id='cv_variables',
                options=[{'label': country_name, 'value': idx} for idx, country_name in enumerate(plot_vars)],
                #value=[{'label':'deaths' 'value':1} for idx, colname],
                value=0,
                placeholder='Select Variables',
                multi=False,
                className="dropdown"
            ),]
        ),
    ])),
    dbc.Row(html.Div(
    [
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
    # dbc.Row(html.Div(className="bodyDiv", children=
    # [
    #     html.Div(className="tableCont",children=
    #     [
    #         html.H1('Data Table'),
    #         html.H3('Select Columns:'),
    #         #html.H3('Selected Columns:', className="subcomponent"),
    #         dcc.Dropdown(
    #             id='table_dropdown_select',
    #             options=[{'label': colname, 'value': idx} for idx, colname in enumerate(full_df.columns)],
    #             #value=[{'label':'deaths' 'value':1} for idx, colname],
    #             value=[5,3,7,8,11,12,16,22,13,17,23,14,24,15,25],
    #             placeholder='Select Columns',
    #             multi=True,
    #             className="side_controls"
    #         ),
    #         dash_table.DataTable(
    #         id='table',
    #         columns=[{"name": i, "id": i} for i in full_df.columns],
    #         #columns=[],
    #         data=full_df.to_dict('records'),
    #         editable=True,
    #         filter_action="native",
    #         sort_action="native",
    #         sort_mode="multi",
    #         column_selectable="single",
    #         row_selectable="multi",
    #         selected_columns=[],
    #         selected_rows=[],
    #         page_action="native",
    #         page_size=20,
    #         style_table={'overflowX': 'scroll', 'padding': '10px'},
    #         ),
    #         html.Div(id="datatable-interactivity-container")
    #     ]),
    # ])),
    # dbc.Row(html.Div(className="footer", children=["footer"])),
    ]
    return layout


def parse_state(url):
    parse_result = urlparse(url)
    params = parse_qsl(parse_result.query)
    state = dict(params)
    return state

@app.callback(Output('page-layout', 'children'),
              inputs=[Input('url', 'href')])
def page_load(href):
    if not href:
        return []
    state = parse_state(href)
    return build_layout(state)

@app.callback(Output('url', 'search'),
              inputs=[Input(i, 'value') for i in component_ids])
def update_url_state(*values):
    state = urlencode(dict(zip(component_ids.keys(), values)))
    return f'?{state}'


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
     Input(component_id='rollingMean', component_property='value'),
     Input(component_id='threshold_cumulative', component_property='value')]
    )
def update_topGraph(countries, cv_variable, yscale, normalise, rmean, threshold):
    cumulative = True
    figure = plot_figure(countries, cv_variable, yscale, normalise, rmean, threshold, cumulative)
    return figure

@app.callback(
    Output(component_id='bottomGraph', component_property='figure'),
    [Input(component_id='country_names', component_property='value'),
     Input(component_id='cv_variables', component_property='value'),
     Input(component_id='yscale_rb', component_property='value'),
     Input(component_id='normalise', component_property='value'),
     Input(component_id='rollingMean', component_property='value'),
     Input(component_id='threshold_daily', component_property='value')]
    )
def update_bottomGraph(countries, cv_variable, yscale, normalise, rmean, threshold):
    cumulative = False
    figure = plot_figure(countries, cv_variable, yscale, normalise, rmean, threshold, cumulative)
    return figure

if __name__ == "__main__":
    app.run_server(port=8050, host="127.0.0.1", debug=True)
