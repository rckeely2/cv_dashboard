import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_table
from dash.dependencies import Input, Output, State

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

app = dash.Dash("CV Dashboard", external_stylesheets=[dbc.themes.SUPERHERO])
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
    #var_str = f"{plot_vars[cv_variable]}"
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
        #yaxis={'type': yscale, 'title':f"{plot_var} rmean:{rmean_options[rmean]} threshold:{threshold}"})
        yaxis={'type': yscale, 'title':f"{plot_var}"})
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

app.layout = dbc.Container(
                html.Div([
                    dcc.Location(id='url', refresh=False),
                    html.Div(id='page-layout'),
                ])
            )

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
    # 'single_dd' : {'component' : 'single_dd', 'value_type': 'text'}, # Template
    # 'multi_dd' : {'component' : 'multi_dd', 'value_type': 'text'}, # Template
    # 'input' : {'component' : 'input', 'value_type': 'text'}, # Template
    # #'topMap' : {'component' : 'slider', 'value_type': 'numeric'}, # Template
    # 'range' : {'component' : 'rangeSlider', 'value_type': 'numeric'}, # Template
    # 'radioButton' : {'component' : 'radioButton', 'value_type': 'numeric'} # Template
}

def build_layout(params):
    layout = [
        dbc.ButtonGroup([
            # dbc.Button("Cumulative", id="cumPlot_button", className="mb-3", color="primary", ),
            dbc.Button("Daily", id="dailyPlot_button", className="mb-3", color="secondary", ),
            dbc.Button("Data", id="data_button", className="mb-3", color="primary",),
            dbc.Button("Filters", id="filter_button",  className="mb-3", color="primary"),
            dbc.Button("Transforms", id="transform_button", className="mb-3", color="primary",),
            dbc.Button("Plot", id="plotVar_button", className="mb-3", color="primary", ),
            #dbc.Button("Link", id="link_button", className="mb-3", color="info", ),
        ]),
        dbc.Collapse([
            dbc.CardHeader("Generate Short Link:"),
            dbc.Card(dbc.CardBody([
                    dbc.Button("GenerateLink", id="genLink_button", className="mb-3", color="info",),
                    dbc.Input(id='link_label'),
                    ])
                ),
        ],
        id="link_collapse"),
        dbc.Collapse([
            dbc.CardHeader("Select data:"),
            dbc.Card(dbc.CardBody(
                dbc.Form([
                        dbc.FormGroup(
                        [
                            dbc.Label("Countries", html_for="example-email-row", width=2),
                            dbc.Col(
                                apply_default_value(params)(dcc.Dropdown)(
                                                id='country_names',
                                                options=[{'label': country_name, 'value': idx} for idx, country_name in enumerate(full_df['Name'].unique())],
                                                #value=[{'label':'deaths' 'value':1} for idx, colname],
                                                value=reverse_lookup_col_idx('Name', default_countries),
                                                placeholder='Select Countries',
                                                multi=True,
                                                className="dropdown"
                                            ),
                                width=10,
                            ),
                        ],
                        row=True,
                        ),
                        dbc.FormGroup(
                        [
                            dbc.Label("Variables", html_for="example-password-row", width=2),
                            dbc.Col(
                                apply_default_value(params)(dcc.Dropdown)(
                                                id='cv_variables',
                                                options=[{'label': country_name, 'value': idx} for idx, country_name in enumerate(plot_vars)],
                                                #value=[{'label':'deaths' 'value':1} for idx, colname],
                                                value=0,
                                                placeholder='Select Variables',
                                                multi=False,
                                                className="dropdown"
                                            ),
                                width=10,
                            ),
                        ],
                        row=True,
                        )])
            )),
            ],
            id="data_collapse",
        ),
        dbc.Collapse([
            dbc.CardHeader("Apply filters to selected data:"),
            dbc.Card(dbc.CardBody(
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.FormGroup([
                                dbc.Label("Cumulative Threshold", className="mr-2"),
                                apply_default_value(params)(dcc.Dropdown)(
                                    id="threshold_cumulative",
                                     options=[{'label':'None', 'value':0},
                                              {'label':'100', 'value':100},
                                              {'label':'500', 'value':500},
                                              {'label':'1000', 'value':1000}],
                                     placeholder='Select threshold',
                                     value=0,
                                     )
                            ],
                            className="mr-3",
                            ),
                            width=4
                        ),
                        dbc.Col(
                            dbc.FormGroup([
                                dbc.Label("Daily Threshold", className="mr-2"),
                                apply_default_value(params)(dcc.Dropdown)(
                                    id="threshold_daily",
                                     options=[{'label':'None', 'value':0},
                                              {'label':'25', 'value':25},
                                              {'label':'100', 'value':100},
                                              {'label':'250', 'value':250}],
                                     placeholder='Select threshold',
                                     value=0,
                                     )
                            ],
                            className="mr-3",
                            ),
                            width=4
                        ),
                        dbc.Col(
                            dbc.FormGroup([
                                dbc.Label("Rolling mean", className="mr-2"),
                                apply_default_value(params)(dcc.Dropdown)(
                                        id='rollingMean',
                                        options=[{'label': item, 'value': idx } for idx, item in enumerate(rmean_options)],
                                        value=0,
                                        placeholder='Select Countries',
                                        multi=False,
                                        className="dropdown"
                                    )
                            ],
                            className="mr-3",
                            ),
                            width=4
                        ),
                        #dbc.Button("Submit", color="primary"),
                    ],

                )
            )),
            ],
            id="filter_collapse",
        ),
        dbc.Collapse([
            dbc.CardHeader("Transform selected data:"),
            dbc.Card(
                dbc.CardBody([
                    dbc.FormGroup([
                        dbc.Label("Transformations", width=3),
                        dbc.Col(
                            apply_default_value(params)(dbc.RadioItems)(
                                id='normalise',
                                options=[
                                        {'label':'None', 'value':'simple'},
                                        {'label':'Normalise by population', 'value':'normalise'},
                                        {'label':'Percentage of Total (by day)', 'value':'percent'},
                                        {"label": "Indexed to Date", "value": 'index',"disabled": True,},
                                        ],
                                value='simple',
                                ),
                                width=9,
                        ),
                    ],
                    row=True,
                    ),
                    #
                    ]),
                ),
            ],
            id="transform_collapse",
        ),

        dbc.Collapse([
            dbc.CardHeader("Configure the plot:"),
            dbc.Card(
                dbc.CardBody([
                    dbc.Label("Y-Axis:", width=3),
                    apply_default_value(params)(dbc.RadioItems)(
                        options=[{'label':'Linear scale', 'value':'linear'},
                                {'label':'Log scale', 'value':'log'},],
                        value='linear',
                        id='yscale_rb',
                        # labelStyle={'display': 'inline-block'}
                   )#]),
                ])
            ),
            ],
            id="plotVar_collapse",
        ),

        # dbc.Collapse([
        dbc.CardHeader("Cumulative Plot:"),
        dbc.Card(
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
                                title="Cumulative",
                                margin={'t': 0, 'pad':0},
                                #margin=dict(l=20, r=20, t=20, b=20),
                                #height= 800,
                            )
                            }
                    ),
        ),
        #],
        #id="cumPlot_collapse",
        # ),

        # dbc.Collapse([
        dbc.CardHeader("Daily Plot:"),
        dbc.Card(
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
                                #height= 2000,
                                clickmode='event+select',
                                xaxis={'type': 'date', 'title': 'time'},
                                yaxis={'type': 'linear', 'title':generate_plot_var(0, 'disable', False)},
                                margin={'t': 0, 'pad':0},
                                #margin=dict(l=20, r=20, t=20, b=20),
                            )
                            }
                    ),
        ),
        #    ],
        #    id="dailyPlot_collapse",
        #),
    ]
    return layout


def parse_state(url):
    parse_result = urlparse(url)
    params = parse_qsl(parse_result.query)
    state = dict(params)
    return state

@app.callback(
    Output("data_collapse", "is_open"),
    [Input("data_button", "n_clicks")],
    [State("data_collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("filter_collapse", "is_open"),
    [Input("filter_button", "n_clicks")],
    [State("filter_collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("transform_collapse", "is_open"),
    [Input("transform_button", "n_clicks")],
    [State("transform_collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("plotVar_collapse", "is_open"),
    [Input("plotVar_button", "n_clicks")],
    [State("plotVar_collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("link_collapse", "is_open"),
    [Input("link_button", "n_clicks")],
    [State("link_collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# @app.callback(
#     Output("cumPlot_collapse", "is_open"),
#     [Input("cumPlot_button", "n_clicks")],
#     [State("cumPlot_collapse", "is_open")],
# )
# def toggle_collapse(n, is_open):
#     if n:
#         return not is_open
#     return is_open
#
@app.callback(
    Output("dailyPlot_collapse", "is_open"),
    [Input("dailyPlot_button", "n_clicks")],
    [State("dailyPlot_collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("link_label", "value"),
    [Input("genLink_button", "n_clicks")],
)
def toggle_collapse(n):
    return "testLabel"

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
