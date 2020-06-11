import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_table
from dash.dependencies import Input, Output, State

import fetch_data
import db_util

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

import short_url
import country_converter as coco

# Global calls
full_df, constant_df = fetch_data.fetch_all(purge=False)

def map_iso3_to_name(country_list):
	country_list = list(set(full_df['ISO3166:3'].unique()).intersection(set(country_list)))
	print(country_list)
	full_subset = full_df.loc[full_df['ISO3166:3'].isin(country_list)]
	subset_df = full_subset[['Name','ISO3166:3']]
	subset_df = subset_df.drop_duplicates()
	subset_dict = dict(zip(subset_df['ISO3166:3'], subset_df['Name']))
	return [subset_dict[x] for x in country_list]

app = dash.Dash("CV Dashboard", external_stylesheets=[dbc.themes.SUPERHERO])
app.title= "CV Dashboard"
app.config.suppress_callback_exceptions = True

styles = {
	'pre': {
		'border': 'thin lightgrey solid',
		'overflowX': 'scroll'
	}
}

cc = coco.CountryConverter()
EU_states = list(cc.data.loc[cc.data.EU <= 2017]['ISO3'].values)
default_countries = [] #map_iso3_to_name(EU_states)
#print(default_countries)
# default_countries = ['Spain', 'Italy', 'France', 'Germany', 'United States',
# 				'China', 'United Kingdom', 'Korea, Rep.']
unRegion = sorted(list(cc.data.UNregion.unique()))
continents = sorted(list(cc.data.continent.unique()))
intOrg = ['EU', 'OECD', 'EU (Big 6)', 'East Asian Tigers', 'Former Dominions', 'BRICS', 'World']
# superNats = ['EU', 'OECD', 'EU (Big 6)', 'East Asian Tigers', 'World'] + \
# 			list(cc.data.UNregion.unique()) + list(cc.data.continent.unique())
plot_vars = ['Confirmed', 'Deaths', 'Recovered', 'Active', 'Tested']
rmean_options = ['None', '3', '5', '7', '14', '21', '28']

def generate_plot_var(cv_variable, normalise, cumulative):
	if (cv_variable == None) or (cv_variable == 'None'):
		cv_variable = 0
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
	if (threshold == 0):
	#if (threshold == 0) or (threshold is None):
		return series
	if (trim_idx == -1):
		trim_idx = bisect.bisect_left(series, threshold)
	trim_series = np.array(list(series[trim_idx:].values) + [np.nan]*trim_idx)
	if ret_idx:
		return trim_series, trim_idx
	else:
		return trim_series

def generate_x(series, threshold, start_date, end_date):
	if threshold == 0:
		#mask =
		series = pd.to_datetime(series)
		mask = (series >= start_date) & (series <= end_date)
		series = series.loc[mask]
		#df = pd.to_datetime(series)
		#df[]
		#print(series[])
		return series
	else:
		return list(range(1,len(series)+1))

def generate_y(df, plot_var, country, threshold, start_date, end_date):
	if threshold == 0:
		df['Date'] = pd.to_datetime(df['Date'])
		mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
		#print(mask)

		df = df.loc[mask]
		series = df[df['Name']==country][plot_var]

		#print(len(df.columns))
		return series
	else:
		series = df[df['Name']==country][plot_var]
		return series

def generate_single_data(country, threshold, rmean,
						cv_variable, normalise, cumulative,
						start_date, end_date):
	plot_vars_l = [ generate_plot_var(i, normalise, cumulative) for i in range(len(plot_vars))]
	plot_dict = [dict(
		x = generate_x(full_df[full_df['Name']==country]['Date'],
						threshold, start_date, end_date),
		# y = rebase_series(apply_rmean(full_df[full_df['Name']==country][plot_var], rmean, cumulative), threshold),
		y = rebase_series(apply_rmean(generate_y(full_df, plot_var, country,
												threshold, start_date, end_date),
												rmean, cumulative), threshold),
		#y = rebase_series(apply_rmean(full_df[full_df['Name']==country][plot_var], rmean, cumulative),threshold),
		#'text': ['a', 'b', 'c', 'd'],
		#'customdata': ['c.a', 'c.b', 'c.c', 'c.d'],
		name =  plot_vars[idx],
		mode = 'line',
		marker =  {'size': 10}
	) for idx, plot_var in enumerate(plot_vars_l) ]
	return plot_dict

def generate_multi_data(country_list_l, threshold, rmean,
						cv_variable, normalise, cumulative,
						start_date, end_date):
	plot_var = generate_plot_var(cv_variable, normalise, cumulative)
	#print(full_df.columns)
	plot_dict = [dict(
		x = generate_x(full_df[full_df['Name']==country]['Date'],threshold,
							start_date, end_date), # Adjust this to handle start & end date
		y = rebase_series(apply_rmean(generate_y(full_df, plot_var, country,
												threshold, start_date, end_date),
												rmean, cumulative), threshold),
		#y = rebase_series(apply_rmean(full_df[full_df['Name']==country][plot_var], rmean, cumulative), threshold),
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
		if (yscale == 'log'):
			threshold = int(round(math.exp(threshold),1))
		xaxis_val = {'type': 'linear',
					'title': f'Days since {threshold}th {plot_var}'}
	layout_dict = dict(
		clickmode='event+select',
		xaxis=xaxis_val,
		#yaxis={'type': yscale, 'title':f"{plot_var} rmean:{rmean_options[rmean]} threshold:{threshold}"})
		yaxis={'type': yscale, 'title':f"{plot_var}"},
		legend_orientation="h")
	return layout_dict

def generate_data(country_list_l, threshold, rmean,
					cv_variable, normalise, cumulative,
					start_date, end_date):
	#if len(country_l)
	if len(country_list_l) == 1:
		plot_dict = generate_single_data(country_list_l[0], threshold, rmean,
										cv_variable, normalise, cumulative,
										start_date, end_date)
		#layout_dict = generate_single_layout()
	else:
		plot_dict = generate_multi_data(country_list_l, threshold, rmean,
										cv_variable, normalise, cumulative,
										start_date, end_date)
		#layout_dict = generate_multi_layout()
	return plot_dict

def plot_figure(countries, cv_variable, yscale, normalise, rmean, threshold,
					cumulative, start_date, end_date, superNats_sel):
	country_list_l = generate_country_name_list(countries, superNats_sel)
	if (threshold is None):
		threshold = 0
	if (rmean is None):
		rmean = 0
	if (yscale == 'log') and (threshold > 0):
		threshold = math.log(threshold)
	figure={'data': generate_data(country_list_l, threshold, rmean, cv_variable,
								normalise, cumulative, start_date, end_date),
			'layout': generate_layout(threshold, rmean, yscale, cv_variable,
										normalise, cumulative)}
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
				#	 kwargs['value'] = ast.literal_eval((params[kwargs['id']]))
				elif (component_ids[kwargs['id']]['component'] == 'datePicker_range'):
					date_list = ast.literal_eval((params[kwargs['id']]))
					kwargs['start_date'] = dt.datetime.strptime(date_list[0], '%Y-%m-%d')
					kwargs['end_date'] = dt.datetime.strptime(date_list[1], '%Y-%m-%d')
				else:
					kwargs['value'] = params[kwargs['id']]
			return func(*args, **kwargs)
		return apply_value
	return wrapper



def generate_country_name_list(countries=None, superNats_sel=None):
	if (countries == None):
		country_list = full_df['Name'].unique()
	#country_list += ['EU', 'OECD']
		#country_list = ['EU', 'OECD'] + list(country_list)
		return country_list
	elif (superNats_sel == None):
		return full_df['Name'].unique()[countries]
	else:
		states = []
		if superNats_sel == 'EU':
			states = list(cc.data.loc[cc.data.EU <= 2017]['ISO3'].values)
		elif superNats_sel == 'OECD':
			states = list(cc.data.loc[cc.data.OECD <= 2017]['ISO3'].values)
		elif superNats_sel == 'EU (Big 6)':
			states = ['GBR', 'DEU', 'FRA', 'POL', 'ITA', 'ESP']
		elif superNats_sel == 'East Asian Tigers':
			states = ['SGP', 'KOR', 'HKG', 'TWN']
		elif superNats_sel == 'Former Dominions':
			states = ['ZAF', 'AUS', 'NZL', 'CAN']
		elif superNats_sel == 'BRICS':
			states = ['ZAF', 'RUS', 'IND', 'CHN', 'BRA']
		elif superNats_sel == 'World':
			states = list(full_df['ISO3166:3'].unique())
		elif superNats_sel in list(cc.data.UNregion.unique()):
			states = list(cc.data[cc.data.UNregion == superNats_sel].ISO3)
		elif superNats_sel in list(cc.data.continent.unique()):
			states = list(cc.data[cc.data.continent == superNats_sel].ISO3)
		states = map_iso3_to_name(states)
		country_list = list(set(list(full_df['Name'].unique()[countries]) + states))
		# if 'EU' in countries:
		#
		# 	#countries +=
		# if 'OECD' in countries:
		# 	oecd_states = list(cc.data.loc[cc.data.OECD <= 2017]['ISO3'].values)
		# 	countries +=
		return country_list
	# country_list =
	#
	# country_list = [{'label': 'EU', 'value': idx},
	# 				{'label': 'OECD', 'value': idx}'EU', 'OECD'] + country_list


def generate_country_name_options():
	options = [{'label': country_name, 'value': idx} \
		for idx, country_name in enumerate(generate_country_name_list())]
	return options

component_ids = {
	'yscale_rb' : {'component' : 'radioButton', 'value_type': 'text'},
	'normalise' : {'component' : 'radioButton', 'value_type': 'text'},
	'threshold_cumulative' : {'component' : 'single_dd', 'value_type': 'numeric'},
	'threshold_daily' : {'component' : 'single_dd', 'value_type': 'numeric'},
	'rollingMean' : {'component' : 'single_dd', 'value_type': 'numeric'},
	'country_names' : {'component' : 'multi_dd', 'value_type': 'text'},
	'cv_variables' : {'component' : 'single_dd', 'value_type': 'numeric'},
	'date_picker' : {'component' : 'datePicker_range', 'value_type': 'datetime'},
	'intOrg_selector' : {'component' : 'single_dd', 'value_type': 'text'},
	'unRegion_selector' : {'component' : 'single_dd', 'value_type': 'text'},
	'continent_selector' : {'component' : 'single_dd', 'value_type': 'text'},
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
			#dbc.Button("Daily", id="dailyPlot_button", className="mb-3", color="secondary", ),
			dbc.Button("Data", id="data_button", className="mb-3", color="primary",),
			dbc.Button("Filters", id="filter_button",  className="mb-3", color="primary"),
			dbc.Button("Transforms", id="transform_button", className="mb-3", color="primary",),
			dbc.Button("Plot", id="plotVar_button", className="mb-3", color="primary", ),
			#dbc.Button("Link", id="link_button", className="mb-3", color="info", ),
		]), # Collapse control buttons
		dbc.Collapse([
			dbc.CardHeader("Generate Short Link:"),
			dbc.Card(dbc.CardBody([
					dbc.Button("GenerateLink", id="genLink_button",
								className="mb-3", color="info",),
					dbc.Input(id='link_label'),
					])
				),
		], # Short link generator :: ToDo : Replace with encode/decode URL
		id="link_collapse"),
		dbc.Collapse([
			dbc.CardHeader("Select data:"),
			dbc.Card(dbc.CardBody(
				dbc.Form([
					dbc.FormGroup([
						dbc.Label("Int. Org.",
									html_for="example-email-row",
									width=2),
						dbc.Col(
							apply_default_value(params)(dcc.Dropdown)(
								id='intOrg_selector',
								options=[{'label': plot_var, 'value': idx} \
									for idx, plot_var in enumerate(intOrg)],
								#value=[{'label':'deaths' 'value':1} for idx, colname],
								value=0,
								placeholder='Select Entity',
								multi=False,
								className="dropdown"
							),
							width=10,
						),
					],row=True,), # International Organisations Picker
					dbc.FormGroup([
						dbc.Label("UN Region",
									html_for="example-email-row",
									width=2),
						dbc.Col(
							apply_default_value(params)(dcc.Dropdown)(
								id='unRegion_selector',
								options=[{'label': plot_var, 'value': idx} \
									for idx, plot_var in enumerate(unRegion)],
								#value=[{'label':'deaths' 'value':1} for idx, colname],
								value=None,
								placeholder='Select Entity',
								multi=False,
								className="dropdown"
							),
							width=10,
						),
					],row=True,), # UN Region Picker
					dbc.FormGroup([
						dbc.Label("Continent",
									html_for="example-email-row",
									width=2),
						dbc.Col(
							apply_default_value(params)(dcc.Dropdown)(
								id='continent_selector',
								options=[{'label': plot_var, 'value': idx} \
									for idx, plot_var in enumerate(continents)],
								#value=[{'label':'deaths' 'value':1} for idx, colname],
								value=None,
								placeholder='Select Entity',
								multi=False,
								className="dropdown"
							),
							width=10,
						),
					],row=True,), # Continent picker
					dbc.FormGroup([
						dbc.Label("Countries",
									html_for="example-email-row",
									width=2),
						dbc.Col(
							apply_default_value(params)(dcc.Dropdown)(
								id='country_names',
								options=generate_country_name_options(),
								#value=[{'label':'deaths' 'value':1} for idx, colname],
								value=reverse_lookup_col_idx('Name',
														default_countries),
								placeholder='Select Countries',
								multi=True,
								className="dropdown"
							),
							width=10,
						),
					],row=True,), # Countries picker
					dbc.FormGroup([
						dbc.Label("Variables", html_for="example-password-row", width=2),
						dbc.Col(
							apply_default_value(params)(dcc.Dropdown)(
											id='cv_variables',
											options=[{'label': plot_var,
														'value': idx} \
												for idx, plot_var in enumerate(plot_vars)],
											#value=[{'label':'deaths' 'value':1} for idx, colname],
											value=0,
											placeholder='Select Variables',
											multi=False,
											className="dropdown"
										),
							width=10,
						),
					],row=True,) # Variable picker
				]) # Data Form
			)),
		], id="data_collapse",), # Data Collapse
		dbc.Collapse([
			dbc.CardHeader("Apply filters to selected data:"),
			dbc.Card(
				dbc.CardBody([
					dbc.Row([
						dbc.Col(
							dbc.FormGroup([
								dbc.Label("Cumulative Threshold", className="mr-2"),
								apply_default_value(params)(dcc.Dropdown)(
									id="threshold_cumulative",
									 options=[{'label':'None', 'value':0},
											  {'label':'50', 'value':50},
											  {'label':'100', 'value':100},
											  {'label':'500', 'value':500},
											  {'label':'1000', 'value':1000},
											  {'label':'5000', 'value':5000}],
									 placeholder='Select threshold',
									 value=0,
									 )
							], className="mr-3",),
							width=4), # Cumulative threshold
						dbc.Col(
							dbc.FormGroup([
								dbc.Label("Daily Threshold", className="mr-2"),
								apply_default_value(params)(dcc.Dropdown)(
									id="threshold_daily",
									 options=[{'label':'None', 'value':0},
											  {'label':'10', 'value':10},
											  {'label':'25', 'value':25},
											  {'label':'50', 'value':50},
											  {'label':'100', 'value':100},
											  {'label':'250', 'value':250}],
									 placeholder='Select threshold',
									 value=0,
									 )
							],
							className="mr-3",
							),
							width=4
						), # Daily threshold
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
						), # Rolling mean
						#dbc.Button("Submit", color="primary"),
					]),
					dbc.Row([
							dbc.Col([
								dbc.FormGroup([
									dbc.Label("Date Range"),
									#dcc.DatePickerRange(
									apply_default_value(params)(dcc.DatePickerRange)(
										id='date_picker',
										min_date_allowed=dt.datetime(2020, 1, 22),
										max_date_allowed=(dt.datetime.today() - dt.timedelta(1)),
										#initial_visible_month=dt.datetime(2020, 1, 20),
										start_date=dt.datetime(2020, 1, 31),
										end_date=(dt.datetime.today() - dt.timedelta(1)),
										className="dash-bootstrap"
									), # Date Picker
								], className="mr-3")
							]),
					]),
				]),
			)
		], id="filter_collapse",), # Filters Collapse
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
					],row=True,), # Transformation form group
					#
					]),
				),
			],
			id="transform_collapse",
		), # Transform Collapse
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
		], id="plotVar_collapse",), # Plot Var Collapse

		# dbc.Collapse([
		dbc.CardHeader("Cumulative Plot:"),
		dbc.Card(
			dcc.Graph(
				id='topGraph',
				figure={
					'data': [dict(
							x = full_df[full_df['Name']==country]['Date'],
							y = full_df[full_df['Name']==country][generate_plot_var(0, 'disable', True)],
							#'text': ['a', 'b', 'c', 'd'],
							#'customdata': ['c.a', 'c.b', 'c.c', 'c.d'],
							name =  country,
							mode = 'line',
							marker =  {'size': 10}
						) for country in default_countries ],
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
		), # Cumulative plot Card
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
		), # Daily Plot Card
		#	],
		#	id="dailyPlot_collapse",
		#),
	] # End of layout
	return layout

def handle_country_group(group, idx):
	if (idx == 'None') or (idx == None):
		return []
	else:
		return [group[int(idx)]]

def handle_groups(intOrg_sel, unRegion_sel, continent_sel):
	superNats_sel = handle_country_group(intOrg,intOrg_sel)
	superNats_sel += handle_country_group(unRegion,unRegion_sel)
	superNats_sel += handle_country_group(continents,continent_sel)
	if len(superNats_sel) > 0:
		superNats_sel = superNats_sel[0]
	elif len(superNats_sel) == 0:
		superNats_sel = None
	return superNats_sel

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
#	 Output("cumPlot_collapse", "is_open"),
#	 [Input("cumPlot_button", "n_clicks")],
#	 [State("cumPlot_collapse", "is_open")],
# )
# def toggle_collapse(n, is_open):
#	 if n:
#		 return not is_open
#	 return is_open
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

	# Partially complete implementation of URL encode
	# print(f"On page load href : [ {href} ]")
	# conn = db_util.open_connection()
	# state_info = href.split('/')[-]
	# decoded_state_info = db_util.decode_url(conn, state_info)
	# conn.close()
	# state = parse_state(decoded_href)

	state = parse_state(href)
	return build_layout(state)

@app.callback(Output('url', 'search'),
			  inputs=[Input(i, 'value') for i in component_ids] +
			  [Input(component_id='date_picker', component_property='start_date'),
			   Input(component_id='date_picker', component_property='end_date')])
def update_url_state(*values):
	# Need to append the start & end date for the date picker here, bit of a hack but...
	values = list(values[0:-2]) + [[values[-2][0:10],values[-1][0:10]]]
	state = urlencode(dict(zip(list(component_ids.keys()) + ['date_picker'], values)))
	href = f'?{state}'

	# Partially complete implementation of short url
	# print(f"In update_url_state href : [ {href} ]")
	# conn = db_util.open_connection()
	# encoded_href = f"?{db_util.encode_url(conn, href)}"
	# conn.close()
	# # return encoded_href
	# print(f"In update_url_state encoded_href : [ {encoded_href} ]")
	# return encoded_href

	return href

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
	 Input(component_id='threshold_cumulative', component_property='value'),
	 Input(component_id='date_picker', component_property='start_date'),
	 Input(component_id='date_picker', component_property='end_date'),
	 Input(component_id='intOrg_selector', component_property='value'),
	 Input(component_id='unRegion_selector', component_property='value'),
	 Input(component_id='continent_selector', component_property='value'),]
	)
def update_topGraph(countries, cv_variable, yscale, normalise,
					rmean, threshold, start_date, end_date, intOrg_sel,
					unRegion_sel, continent_sel):
	cumulative = True
	superNats_sel = handle_groups(intOrg_sel, unRegion_sel, continent_sel)
	figure = plot_figure(countries, cv_variable, yscale, normalise, rmean,
							threshold, cumulative, start_date, end_date,
							superNats_sel)
	return figure

@app.callback(
	Output(component_id='bottomGraph', component_property='figure'),
	[Input(component_id='country_names', component_property='value'),
	 Input(component_id='cv_variables', component_property='value'),
	 Input(component_id='yscale_rb', component_property='value'),
	 Input(component_id='normalise', component_property='value'),
	 Input(component_id='rollingMean', component_property='value'),
	 Input(component_id='threshold_daily', component_property='value'),
	 Input(component_id='date_picker', component_property='start_date'),
	 Input(component_id='date_picker', component_property='end_date'),
	 Input(component_id='intOrg_selector', component_property='value'),
	 Input(component_id='unRegion_selector', component_property='value'),
	 Input(component_id='continent_selector', component_property='value'),]
	)
def update_bottomGraph(countries, cv_variable, yscale, normalise, rmean,
						threshold, start_date, end_date, intOrg_sel,
						unRegion_sel, continent_sel):
	cumulative = False
	# print(f"Countries : {countries}")
	# print(f"CV Variable : {cv_variable}")
	# print(f"Y-scale : {yscale}")
	# print(f"Normalise : {normalise}")
	# print(f"Rolling mean : {rmean}")
	# print(f"Threshold : {threshold}")
	# print(f"Start date : {start_date}")
	# print(f"End date : {end_date}")
	superNats_sel = handle_groups(intOrg_sel, unRegion_sel, continent_sel)
	figure = plot_figure(countries, cv_variable, yscale, normalise, rmean,
						threshold, cumulative, start_date, end_date, superNats_sel)
	return figure

if __name__ == "__main__":
	app.run_server(port=8050, host="127.0.0.1", debug=True)
