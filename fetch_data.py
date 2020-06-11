import pandas as pd
import numpy as np
import urllib
import urllib.request
import ssl
import bs4
import json
import os
import time
import datetime as dt
import eurostat
import fake_useragent
import requests
import io

import country_converter as coco

ssl._create_default_https_context = ssl._create_unverified_context

def fetch_jh_raw(save_file=True,
					data_path='assets/',
					filename='coronaVirus_global.csv',
					threshold=1):
	if check_file_age(filename, threshold=threshold):
		return pd.read_csv(f'{data_path}{filename}')
	df = pd.DataFrame()
	region = "global"
	for c in [ "deaths", "confirmed", "recovered"]:
		target_url = f"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_{c}_{region}.csv"
		new_df = pd.read_csv(target_url)
		first_date = new_df.columns.get_loc("1/22/20")
		last_date = new_df.shape[1]
		new_df["Counter"] = f"{c}_total"
		df = pd.concat([df, new_df])
	empty_columns = ['UID', 'FIPS', 'CountyName', 'FullName']
	for col in empty_columns:
		df[col] = np.nan
	new_order = list(df.columns[0:first_date]) + list(df.columns[last_date:]) + list(df.columns[first_date:last_date])
	df = df[new_order]
	df.rename(columns={'Lat': 'Latitude',
						'Long':'Longitude',},
			 inplace=True)
	if save_file:
		df.to_csv(f"{data_path}{filename}", index=False)
	return df

def fetch_US(save_file=True,
			 data_path='assets/',
			 filename='coronaVirus_US.csv',
			 threshold=1):
	if check_file_age(filename, threshold):
		return pd.read_csv(f'{data_path}{filename}')
	df = pd.DataFrame()
	region = "US"
	for c in [ "deaths", "confirmed"]:
		target_url = f"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_{c}_{region}.csv"
		new_df = pd.read_csv(target_url)
		new_df["Counter"] = f"{c}_total"
		df = pd.concat([df, new_df])
	df.rename(columns={'Province_State':'Province/State',
					'Country_Region':'Country/Region',
					'Lat': 'Latitude',
					'Long_':'Longitude',
					'iso2' : 'ISO3166_alpha2',
					'iso3' : 'ISO3166_alpha3',
					'code3': 'ISO3166_numeric',
					'Admin2':'CountyName',
					'Combined_Key': 'FullName'},
			  inplace=True)
	if save_file:
		df.to_csv(f"{data_path}{filename}", index=False)
	return df

def build_frame(df, country, province=None):
	first_date = df.columns.get_loc("1/22/20")
	if province is None:
		sf = df.loc[(df['Country/Region'] == country) &
					(df['Province/State'].isna())]
		sf = sf.iloc[:,first_date:].T
		sf['Province/State'] = 'All'
	else:
		sf = df.loc[(df['Country/Region'] == country) &
					(df['Province/State'] == province)]
		sf = sf.iloc[:,first_date:].T
		sf['Province/State'] = province
	sf.columns = ['confirmed_total', 'deaths_total',
				  'recovered_total', 'Province/State']
	sf['Country/Region'] = country
	codes = map_name_to_ISOcode(country)
	sf['ISO3166_alpha2'] = codes[0]
	sf['ISO3166_alpha3'] = codes[1]
	sf['ISO3166_numeric'] = codes[2]
	return sf

def plot_df(df, days=30):
	plt.title("Per day")
	plt.plot(df.index[-days:], df['confirmed'][-days:], label="Confirmed")
	plt.plot(df.index[-days:], df['deaths'][-days:], label="Deaths")
	plt.plot(df.index[-days:], df['recovered'][-days:], label="Recovered")
	plt.legend()
	plt.show()

#	 plt.title("2nd diff")
#	 plt.plot(df.index[-days:], df['confirmed_diff_2'][-days:], label="Confirmed")
#	 plt.plot(df.index[-days:], df['deaths_diff_2'][-days:], label="Deaths")
#	 plt.plot(df.index[-days:], df['recovered_diff_2'][-days:], label="Recovered")
#	 plt.legend()
#	 plt.show()

	plt.title("Cumulative, Linear")
	plt.plot(df.index[-days:], df['confirmed_total'][-days:], label="Confirmed")
	plt.plot(df.index[-days:], df['deaths_total'][-days:], label="Deaths")
	plt.plot(df.index[-days:], df['recovered_total'][-days:], label="Recovered")
	plt.legend()
	plt.show()

	plt.title("Cumulative, Log")
	plt.plot(df.index[-days:], df['confirmed_log_total'][-days:], label="Confirmed")
	plt.plot(df.index[-days:], df['deaths_log_total'][-days:], label="Deaths")
	plt.plot(df.index[-days:], df['recovered_log_total'][-days:], label="Recovered")
	plt.legend()
	plt.show()

	plt.title("Per day, Log")
	plt.plot(df.index[-days:], df['confirmed_log'][-days:], label="Confirmed")
	plt.plot(df.index[-days:], df['deaths_log'][-days:], label="Deaths")
	plt.plot(df.index[-days:], df['recovered_log'][-days:], label="Recovered")
	plt.legend()
	plt.show()

def calc_maxes(vec_length, row_length, panels = 3):
	# Calculate columns from desired row length
	col_max = row_length
	if vec_length < panels:
		col_max = vec_length
	# Calculate rows from vector length and row length
	div, rem = divmod(vec_length, row_length)
	if rem == 0:
		row_max = div
		del_cells = 0
	else:
		row_max = div + 1
		del_cells = col_max - rem
	return row_max, col_max, del_cells

def plot_set(series,
			 panels = 3,
			 col_width  = 7,
			 row_height = 5,
			 fig_title="Title"):
	# Calculate the number of rows
	row_max, col_max, del_cells = calc_maxes(len(series), panels)
	fig, axs = plt.subplots(row_max, col_max)
	fig.set_size_inches(col_max * col_width, row_max * row_height)
	fig.suptitle(fig_title)

	row_idx = 0
	col_idx = 0

	# Handle single row case
	if (row_max == 1):
		for s in series:
			axs[col_idx].plot(s)
			col_idx += 1
	# Multirow case
	else:
		for s in series:
			axs[row_idx, col_idx].plot(s)
			col_idx += 1
			if (col_idx == col_max):
				row_idx += 1
				col_idx = 0

	# Hide extra panels
	offset = col_max - del_cells
	for i in range(0, del_cells):
		axs[row_max-1, offset+i].axis('off')

def summarise_multiregion(df):
	capital_provinces = { 'Australia' : 'Australian Capital Territory',
						  'China' : 'Beijing',
						  'Canada' : 'Ontario'}

	for country in list(capital_provinces.keys()):
		if country == 'Canada':
			counter_vars = ['deaths_total', 'confirmed_total']
		else:
			counter_vars = list(df['Counter'].unique())
		temp_df = df.head(0)
		for counter_var in counter_vars:
			sf = df.loc[(df['Country/Region'] == country) &
						(df['Counter'] == counter_var)]
			row = sf.sum(numeric_only=True)

			row['Country/Region'] = country
			row['Counter'] = counter_var
			for update_var in ['Latitude', 'Longitude']:
				cp = capital_provinces[country]
				row[update_var] = sf.loc[(sf['Province/State'] == cp)][update_var].values[0]
			temp_df = temp_df.append(row, ignore_index = True)
		df = df.append(temp_df, ignore_index=True)
	return df

def threshold_plot(df,
					countries,
					plot_var,
					threshold=25,
					x_lim=45,
					y_lim=None,
					log_var = False):
	if log_var:
		threshold = math.log(threshold)
	plt.figure(figsize=(10,7))
	plt.title(f"Total coronavirus deaths for places with at least {threshold} deaths")

#	 max_region_st = 0
	for c in countries:
		country_df = build_frame(df, c)
		region_s = country_df[plot_var]
		trim_idx = bisect.bisect_left(region_s,threshold)
		region_st = np.array(list(region_s[trim_idx:].values) + [np.nan]*trim_idx)
		plt.plot(region_st-threshold, label=c)
#		 if region_st > max_region_st:
#			 max_region_st = region_st
	#print(f"region_st {len(region_st)}")
	x = np.linspace(0,len(region_st),len(region_st))

	plt.plot(np.power(2,x), color="#666666")
	plt.plot(np.power(2,x/2), color="#666666")
	plt.plot(np.power(2,x/3), color="#666666")
	plt.plot(np.power(2,x/7), color="#666666")
	#weekly = np.power(2,x/30)
	#print(f"region_st {len(region_st)}")
	plt.plot(np.power(2,x/30), color="#666666")

	if y_lim is not None:
		plt.ylim((1,y_lim))
	plt.ylabel("Cumulative deaths")
	plt.xlabel(f"Days since {threshold}th death")
	plt.xlim((0,x_lim))
	plt.yscale("log", basey=2)

	plt.legend()
	plt.show()

def test_countries(df):
	all_countries = df['Country/Region'].unique()
	fail_count = 0
	for c in all_countries:
		try:
			country_df = build_frame(df, c)
		except ValueError:
			print(f"\t+ [{c}] not found...")
			fail_count += 1
		# Getting more detail : [{sys.exc_info()[0]}]
		# from https://docs.python.org/3/tutorial/errors.html
	if fail_count == 0:
		print("\t+ No failures found.")

iso_codes_df = None

def map_name_to_ISOcode(country):
	global iso_codes_df
	if iso_codes_df is None:
		iso_codes_df = fetch_ISO_codes()
	if country == 'World':
		codes = ['WL', 'WLD', '000']
	else:
		codes = iso_codes_df.loc[(iso_codes_df['name'] == country),
				['ISO3166_alpha2', 'ISO3166_alpha3', 'ISO3166_numeric']].values[0]
	return list(codes)

def fetch_ISO_codes(save_file = True,
					data_path='assets/',
					filename='ISO_codes.csv',
					threshold=30):

	if check_file_age(filename, threshold=threshold):
		return pd.read_csv(f'{data_path}{filename}')

	source_url = "https://www.cia.gov/library/publications/the-world-factbook/appendix/appendix-d.html"
	source_page = urllib.request.urlopen(source_url)
	soup = bs4.BeautifulSoup(source_page, 'html.parser')
	table = soup.find('div', attrs={'class':'nav-list'}).find('table')
	rows = table.findChildren('tr')
	country = {}
	countries_df = pd.DataFrame()
	for idx, row in enumerate(rows[1:]):
		columns = row.findChildren('td')
		country['name'] = columns[0].text.replace("\n","")
		country['GEC'] = columns[1].text
		country['ISO3166_alpha2'] = columns[2].text
		country['ISO3166_alpha3'] = columns[3].text
		country['ISO3166_numeric'] = columns[4].text
		country['STANAG'] = columns[5].text
		country['ccTLDs'] = columns[6].text
		country['notes'] = columns[7].text
		countries_df = countries_df.append(country,
											ignore_index=True)
	if save_file:
		countries_df.to_csv(f"{data_path}{filename}", index=False)
	return countries_df

def fix_names(df):
	manual_overwrites = {
		'Bahamas' : 'Bahamas, The',
		'Congo (Brazzaville)' : 'Congo, Republic of the',
		'Congo (Kinshasa)' : 'Congo, Democratic Republic of the',
		'Gambia' : 'Gambia, The',
		'Holy See' : 'Holy See (Vatican City)',
		'Taiwan*' : 'Taiwan',
		'US' : 'United States',
		'West Bank and Gaza': 'West Bank'}
	for key in manual_overwrites.keys():
		df.loc[df['Country/Region']==key,
					'Country/Region'] = manual_overwrites[key]
	return df

def check_iso(df, iso_codes_df):
	not_found_count = 0
	for country in df['Country/Region'].unique():
		if iso_codes_df.loc[iso_codes_df.name==country].shape[0] == 0:
			not_found_count += 1
			print(f"\t+ [ {country} ] not found...")
	if not_found_count == 0:
		print("\t+ All found...")

def fix_ships(df):
	for ship in ['MS Zaandam', 'Diamond Princess']:
		df.loc[(df["Country/Region"] == ship), "Province/State"] = ship
		df.loc[(df["Country/Region"] == ship), "Country/Region"] = 'Other'

	df.loc[(df["Country/Region"] == 'Canada') &
			(df["Province/State"] == 'Grand Princess'),
			'Country/Region'] = 'Other'

	df.loc[(df["Country/Region"] == 'Canada') &
			(df["Province/State"] == 'Diamond Princess'),
			"Province/State"] = 'Diamond Princess (US)'
	df.loc[(df["Country/Region"] == 'Canada') &
			(df["Province/State"] == 'Diamond Princess (US)'),
			'Country/Region'] = 'Other'
	return df

def fetch_indicators(indicators,
					save_file=True,
					data_path='assets/',
					filename='wb_indicators.csv',
					threshold=30):
	if check_file_age(filename, threshold=threshold):
		return pd.read_csv(f'{data_path}{filename}')

	mrnev = 1
	indicator_df = pd.DataFrame()
	for indicator in indicators:
		request_df = wb_request_indicator(indicator, mrnev=1)
		indicator_df = pd.concat([indicator_df, request_df])
		#print(indicator_df.shape[0])

	if save_file:
		indicator_df.to_csv(f'{data_path}{filename}', index=False)
	return indicator_df

def wb_request_indicator(indicator, country=None, mrnev=None,
						 date=None):
	# Adapted from here: [ https://datahelpdesk.worldbank.org/knowledgebase/articles/898581-api-basic-call-structures ]
	if country is None:
		country = 'all'
	elif type(country) is list:
		country = ";".join(country).lower()
	wb_request_url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json"

	if mrnev is not None:
		wb_request_url += f'&mrnev={str(mrnev)}'
	elif date is not None:
		if type(date) is list:
			wb_request_url += f'&date={date_range[0]}:{date_range[1]}'
		else:
			wb_request_url += f'&date={date}'
	#
	print(f"Request URL [ {wb_request_url} ]")
	with urllib.request.urlopen(wb_request_url) as response:
		response_json = json.loads(response.read())

	if 'page' in response_json[0].keys():
		page = response_json[0]['page']
		total_pages = response_json[0]['pages']
		indicator_list = response_json[1]

		while page != total_pages:
			with urllib.request.urlopen(wb_request_url + f'&page={page+1}') as response:
				response_json = json.loads(response.read())
			indicator_list += response_json[1]
			page = response_json[0]['page']

		indicator_list = list(map(flatten_item, indicator_list))
		return pd.DataFrame(indicator_list)
	else:
		print("Error. Invalid request:")
		print(f"Request [{wb_request_url}]")
		print(f"Response:")
		print(response_json)
		return None

def flatten_item(item):
	item['indicator_id'] = item['indicator']['id']
	item['indicator_name'] = item['indicator']['value']
	item['country_id'] = item['country']['id']
	item['country_name'] = item['country']['value']
	item.pop('indicator')
	item.pop('country')
	return item

def reorder_jh_columns(df, first_date_str="1/22/20", final_offset=8):#, iso_codes_df, indicator_df,
					#indicator_names):
	first_date = df.columns.get_loc(first_date_str)
	last_date = len(df.columns) - final_offset
	new_order = list(df.columns[0:first_date]) + \
					list(df.columns[last_date:]) + \
					list(df.columns[first_date:last_date])
	df = df[new_order]
	return df

def remove_local_copy(filename, data_path='assets/'):
	filepath = f"{data_path}{filename}"
	if os.path.exists(filepath):
		print(f"\t+ Removing file [ {filepath} ]...")
		os.remove(filepath)
	else:
		print(f"\t+ File [ {filepath} ] not found. Cannot remove...")

def merge_world_bank_indicators():
	# first = True
	# print("Processing indicators...")
	# for indicator in indicator_names.keys():
	# 	indicator_name = indicator_names[indicator]
	# 	sub_df = indicator_df.loc[indicator_df['indicator_id'] == indicator].copy()
	# 	sub_df.rename(columns={'value':indicator_name,
	# 							'countryiso3code':'ISO3166_alpha3',
	# 							'country_name':'DisplayName'}
	# 				  ,inplace=True)
	# 	sub_df[f'{indicator_name}_notes'] = str(sub_df['indicator_name']) + \
	# 					" in " + str(sub_df['date']) + " (" + \
	# 					str(sub_df['indicator_id']) + ")"
	#
	# 	column_list = ['ISO3166_alpha3', indicator_name,
	# 					f'{indicator_name}_notes']
	# 	if first:
	# 		column_list += ['DisplayName']
	# 		first = False
	# 	sub_df = sub_df[column_list]
	# 	cv_merged_df = pd.merge(cv_merged_df, sub_df, how='left',
	# 						on='ISO3166_alpha3')
	pass

def fetch_jh_clean(#indicator_names,
					save_file=True,
					data_path='assets/',
					filename='jh_clean.csv',
					threshold=1,):
	if check_file_age(filename, threshold=threshold):
		jh_clean_df = pd.read_csv(f'{data_path}{filename}')
		#testing_df = None
		#iso_codes_df = fetch_ISO_codes()
		#indicator_df = fetch_indicators(list(indicator_names.keys()))
		return jh_clean_df#, iso_codes_df#, indicator_df

	# Fetch the raw John Hopkins data
	jh_raw_df = fetch_jh_raw()
	# Fetch the ISO codes from the CIA World Factbook page
	iso_codes_df = fetch_ISO_codes()
	# Fix the names for the John Hopkins data
	print(">> Before name fixes:")
	check_iso(jh_raw_df, iso_codes_df)
	jh_fixed_df = fix_names(jh_raw_df)
	jh_fixed_df = fix_ships(jh_fixed_df)
	print(">> After name fixes:")
	check_iso(jh_fixed_df, iso_codes_df)
	# Corrections to the John Hopkins data
	print(">> Before summarise countries:")
	test_countries(jh_fixed_df)
	jh_fixed_df = summarise_multiregion(jh_fixed_df)
	jh_fixed_df = jh_fixed_df.loc[jh_fixed_df['Country/Region'] != 'Other']
	print(">> After summarise countries:")
	test_countries(jh_fixed_df)

	# print(">> Fetching World Bank Indicators...")
	# indicator_df = fetch_indicators(list(indicator_names.keys()))
	#indicator_df.loc[indicator_df['countryiso3code'] == 'USA']

	print(">> Merging ISO codes...")
	jh_fixed_df = pd.merge(jh_fixed_df, iso_codes_df, how='left',
							left_on='Country/Region', right_on='name')

	print(">> Reordering columns...")
	jh_fixed_df = reorder_jh_columns(jh_fixed_df)



	# print(">> Calculating world totals...")
	# cv_merged_df = calculate_world_totals(cv_merged_df)

	print(">> Converting to desired format...")
	jh_clean_df = rotate_jh_df(jh_fixed_df)

	if save_file:
		full_path = f"{data_path}{filename}"
		print(f"\t+ Saving to [ {full_path} ]")
		jh_clean_df.to_csv(full_path, index=False)

	return jh_clean_df#, iso_codes_df, indicator_df

def generate_covid_features(df):
	print(">> Generating Covid features...")
	df = df.copy()

	roots = ['confirmed', 'deaths', 'recovered', 'tested']

	for root in roots:
		df[f'{root}_daily'] = 0
	df[f'active_daily'] = 0
	df[f'active_total'] = 0

	for entity in df['ISO3166_alpha3'].unique():
		sdf = df.loc[df['ISO3166_alpha3'] == entity].copy()
		for root in roots:
			sdf[f'{root}_daily'] = sdf[f'{root}_total'].diff()

		sdf[f'active_daily'] = sdf['confirmed_daily'] - sdf['deaths_daily'] - sdf['recovered_daily']
		sdf[f'active_total'] = sdf[f'active_daily'].cumsum()
		df.update(sdf)

	roots += ['active']
	suffixes = ['_total', '_daily']
	for suffix in suffixes:
		for root in roots:
			df[f'{root}{suffix}_norm'] = (df[f'{root}{suffix}'] / df['owid_population']) * 1e6

	for date in df['Date'].unique():
		for suffix in suffixes:
			for root in roots:
				try:
					world_val = df.loc[(df['ISO3166_alpha3']=='WLD') &
										(df['Date']==date),
										f'{root}{suffix}'].values[0]
					assigned_value = (df[f'{root}{suffix}'] / world_val) * 100
					df.loc[df['Date']==date,f'{root}{suffix}_pct'] = assigned_value
				except:
					df.loc[df['Date']==date,f'{root}{suffix}_pct'] = 0
	return df

def fetch_owid(save_file=True,
				covid_filename = 'owid_covid_raw.csv',
				constant_filename = 'owid_constants.csv',
				threshold=1,
				data_path='assets/'):
	if check_file_age(covid_filename, threshold) and check_file_age(constant_filename, threshold):
		owid_covid_df = pd.read_csv(f'{data_path}{covid_filename}')
		owid_constant_df = pd.read_csv(f'{data_path}{constant_filename}')
		return owid_covid_df, owid_constant_df

	# Load raw data and rename columns
	print(">> Loading covid data...")
	owid_raw_df = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')
	owid_raw_df.columns = [ f"owid_{x}" for x in owid_raw_df.columns ]

	# Select the covid data
	owid_covid_cols = ['owid_iso_code', 'owid_location', 'owid_date',
						'owid_total_cases', 'owid_total_deaths', 'owid_total_tests',
						'owid_tests_units', 'owid_stringency_index', 'owid_population']
	owid_covid_df = owid_raw_df[owid_covid_cols].copy()
	owid_covid_df['owid_date'] = pd.to_datetime(owid_covid_df['owid_date'])

	# Select the constant data
	constant_columns = list(owid_raw_df.columns[0:2]) + list(owid_raw_df.columns[20:])
	owid_constant_df = owid_raw_df[constant_columns].drop_duplicates()

	if save_file:
		owid_covid_df.to_csv(f"{data_path}{covid_filename}", index=False)
		owid_constant_df.to_csv(f"{data_path}{constant_filename}", index=False)
	return owid_covid_df, owid_constant_df

def merge_covid_sources(jh_df, owid_covid_df, 	save_file=True,
						data_path='assets/',
						filename='covid_merged.csv',
						threshold=1):
	if check_file_age(filename, threshold=threshold):
		return pd.read_csv(f'{data_path}{filename}')

	print(">> Merging Covid data sources...")
	non_covid_cols = ['Country/Region', 'Province/State', 'ISO3166_alpha2',
						'ISO3166_alpha3', 'ISO3166_numeric',
						#'population', 'gdp_total', 'lifeExp_male', 'lifeExp_female',
						'Date']

	last_date = jh_df['Date'].unique()[-1]
	noncovid_sdf = jh_df.loc[jh_df['Date'] == last_date][non_covid_cols]
	owid_dates = set(owid_covid_df['owid_date'].unique())
	jh_dates = set(jh_df['Date'].unique())
	owid_only = sorted(list(owid_dates - jh_dates))
	common_dates = sorted(list(owid_dates.intersection(jh_dates)))
	merged_df_columns = list(jh_df.columns) + list(owid_covid_df.columns)

	owid_covid_df.loc[owid_covid_df['owid_iso_code'] == 'OWID_WRL', 'owid_iso_code'] = 'WLD'

	merged_df = pd.DataFrame(columns=merged_df_columns)

	for date in owid_only:
		owid_sdf = owid_covid_df.loc[owid_covid_df['owid_date'] == date]
		temp_sdf = noncovid_sdf.merge(owid_sdf, how='left', left_on='ISO3166_alpha3',
								 right_on='owid_iso_code')
		temp_sdf['Date'] = temp_sdf['owid_date']
		merged_df = pd.concat([merged_df, temp_sdf])

	for date in common_dates:
		owid_sdf = owid_covid_df.loc[owid_covid_df['owid_date'] == date]
		jh_sdf = jh_df.loc[jh_df['Date'] == date]
		jh_sdf = jh_sdf.merge(owid_sdf, how='left', left_on='ISO3166_alpha3',
								 right_on='owid_iso_code')
		merged_df = pd.concat([merged_df, jh_sdf])

	merged_df['tested_total'] = merged_df['owid_total_tests']
	merged_df = merged_df.drop_duplicates()

	# Add summary rows for the World, the EU and the OECD
	#merged_df = generate_summary_rows(merged_df)

	if save_file:
		merged_df.to_csv(f"{data_path}{filename}", index=False)
	return merged_df

# population = np.nan
# if 'population' in df.columns:
#	 population = df.loc[(df['Country/Region'] == country),
#			 'population'].values[0]
# # else:
# #	 print(f"{country} : population missing")
# #	 #print(sf.columns)
#
# col_vars = ['confirmed', 'deaths', 'recovered']
#
# for cv in col_vars:
#	 sf[f'{cv}'] = sf[f'{cv}_total'].diff()
#	 try:
#		 sf[f'{cv}_total_norm'] = (sf[f'{cv}_total'] / population) * 1e6
#	 except:
#		 sf[f'{cv}_total_norm'] = np.nan
#	 sf[f'{cv}_log_total'] = np.log2(sf[f'{cv}_total'])
#	 sf[f'{cv}_log'] = sf[f'{cv}_log_total'].diff()
#	 sf[f'{cv}_log_growth'] = np.power(2, sf[f'{cv}_log']) - 1
#
# sf['net_cases'] = sf['confirmed'] - sf['deaths'] - sf['recovered']
# sf['active_cases'] = sf['net_cases'].cumsum()
	# np.exp(np.diff(np.log(population))) - 1
	#sf[f'{cv}_diff_2'] = sf[f'{cv}_cum'].diff(2)
	#sf[f'{cv}_log_cum'] = np.log(sf[f'{cv}_cum'])
	#sf[f'{cv}_log_diff'] = np.log(sf[f'{cv}_diff']

def rotate_jh_df(cv_df):
	countries = cv_df['Country/Region'].unique()
	first = True

 	# Sometimes the counters are in a different order, this fixes the cat problem
	cv_df = cv_df.sort_values('Counter')

	for country in countries:
		if first:
			full_df = build_frame(cv_df, country)
			first = False
		else:
			temp_df = build_frame(cv_df, country)
			full_df = pd.concat([full_df, temp_df])

	full_df['Date'] = pd.to_datetime(full_df.index)
	full_df.reset_index(inplace=True, drop=True)
	return full_df

def merge_non_covid(covid_df, non_covid_df):

	# Merge in non-covid data <- Should be moved to own function
	merge_columns = [ 'ISO3166_alpha3', 'population', 'gdp_total',
				  'lifeExp_male', 'lifeExp_female', 'DisplayName']
	non_covid_df = non_covid_df[merge_columns]
	full_df = pd.merge(covid_df, non_covid_df, how='inner', on='ISO3166_alpha3')

	# Fix ISO3166:3 codes for states with... issues...
	full_df.loc[full_df["ISO3166_alpha3"] == 'ESH', "DisplayName"] = 'Western Sahara'
	full_df.loc[full_df["ISO3166_alpha3"] == 'TWN', "DisplayName"] = 'Taiwan'
	full_df.loc[full_df["ISO3166_alpha3"] == 'VAT', "DisplayName"] = 'Holy See (Vatican City)'
	full_df.loc[full_df["ISO3166_alpha3"] == 'XKS', "DisplayName"] = 'Kosovo'
	# Fix the world name
	full_df.loc[full_df["ISO3166_alpha3"] == 'WLD', "DisplayName"] = 'World'

	full_df['Date'] = pd.to_datetime(full_df['Date'])
	#full_df = merge_owid(full_df)
	return full_df

def generate_summary_rows(df):
	cc = coco.CountryConverter()
	# First generate the world summary, taking everything into account
	# ToDo: fix this to use a list of the 186 codes...
	df = generate_summary(df)

	# Then do the OECD and EU
	oecd_states = list(cc.data.loc[cc.data.OECD <= 2017]['ISO3'].values)
	oecd_dict = {'Name':'OECD','ISO2':'OE','ISO3':'OEC','ISO#':'997'}
	df = generate_summary(df, oecd_states, oecd_dict)
	eu_states = list(cc.data.loc[cc.data.EU <= 2017]['ISO3'].values)
	eu_dict = {'Name':'EU','ISO2':'EU','ISO3':'EUX','ISO#':'996'}
	df = generate_summary(df, eu_states, eu_dict)
	# Might be an idea to add continent level summaries?
	#df = df.drop_duplicates()
	return df

def generate_summary(df, subset=None, name_dict=None):
	if subset is not None:
		sdf = df.loc[df['ISO3166_alpha3'].isin(subset)].copy()
	else:
		sdf = df.copy()
		name_dict = {'Name': 'World',
			 'ISO2': 'WD',
			 'ISO3': 'WLD',
			 'ISO#': '999'}

	row_list = []

	last_date = sdf['Date'].unique()[-1]
	population = sdf.loc[sdf['Date'] == last_date]['owid_population'].sum()

	for date in sdf['Date'].unique():
		row_dict = sdf.loc[sdf['Date'] == date].sum().to_dict()
		row_dict['Province/State'] = np.nan
		row_dict['owid_stringency_index'] = np.nan
		row_dict['Date'] = date
		row_dict['owid_population'] = population
		row_dict['Country/Region'] = name_dict['Name']
		row_dict['owid_location'] = name_dict['Name']
		row_dict['ISO3166_alpha2'] = name_dict['ISO2']
		row_dict['ISO3166_alpha3'] = name_dict['ISO3']
		row_dict['ISO3166_numeric'] = name_dict['ISO#']
		row_dict['owid_iso_code'] = name_dict['ISO3']

		row_list.append(row_dict)
		#sdf = pd.concat([sdf, temp_df], axis=1)

	df = df.append(row_list, ignore_index=True)
	return df

def rename_select_columns(df, data_path='assets/', filename="final_columns.csv"):

	df['DisplayName'] = df['Country/Region']
	# # ToDo : put this in a file to allow it to be pulled out easily
	new_names = {"Country/Region" : "Country/Region",
			"Province/State" : "Province/State",
			"ISO3166_alpha2" : "ISO3166:2",
			"ISO3166_alpha3" : "ISO3166:3",
			"ISO3166_numeric" : "ISO3166:#",
			"DisplayName" : "Name",
			#"owid_population" : "Population",
			# "population" : "Population",
			# "gdp_total" : "GDP (Total)",
			# "lifeExp_male" : "Life Expectancy (Male)",
			# "lifeExp_female" : "Life Expectancy (Female)",
			"Date" : "Date",
			"confirmed_total" : "Confirmed (Total)",
			"deaths_total" : "Deaths (Total)",
			"recovered_total" : "Recovered (Total)",
			"active_total" : "Active (Total)",
			"tested_total" : "Tested (Total)",
			"confirmed_total_norm" : "Confirmed (Total, normalised)",
			"deaths_total_norm" : "Deaths (Total, normalised)",
			"recovered_total_norm" : "Recovered (Total, normalised)",
			"active_total_norm" : "Active (Total, normalised)",
			"tested_total_norm" : "Tested (Total, normalised)",
			"confirmed_total_pct" : "Confirmed (Total, percent)",
			"deaths_total_pct" : "Deaths (Total, percent)",
			"recovered_total_pct" : "Recovered (Total, percent)",
			"active_total_pct" : "Active (Total, percent)",
			"tested_total_pct" : "Tested (Total, percent)",
			# "confirmed_log_total" : "Confirmed (Total, log)",
			# "deaths_log_total" : "Deaths (Total, log)",
			# "recovered_log_total" : "Recovered (Total, log)",
			"confirmed_daily" : "Confirmed (Daily)",
			"deaths_daily" : "Deaths (Daily)",
			"recovered_daily" : "Recovered (Daily)",
			"active_daily" : "Active (Daily)",
			"tested_daily" : "Tested (Daily)",
			"confirmed_daily_norm" : "Confirmed (Daily, normalised)",
			"deaths_daily_norm" : "Deaths (Daily, normalised)",
			"recovered_daily_norm" : "Recovered (Daily, normalised)",
			"active_daily_norm" : "Active (Daily, normalised)",
			"tested_daily_norm" : "Tested (Daily, normalised)",
			"confirmed_daily_pct" : "Confirmed (Daily, percent)",
			"deaths_daily_pct" : "Deaths (Daily, percent)",
			"recovered_daily_pct" : "Recovered (Daily, percent)",
			"active_daily_pct" : "Active (Daily, percent)",
			"tested_daily_pct" : "Tested (Daily, percent)",
			# "confirmed_log" : "Confirmed (Daily, log)",
			# "deaths_log" : "Deaths (Daily, log)",
			# "recovered_log" : "Recovered (Daily, log)",
			# "confirmed_log_growth" : "Confirmed (Daily, log growth)",
			# "deaths_log_growth" : "Deaths (Daily, log growth)",
			# "recovered_log_growth" : "Recovered (Daily, log growth)"
			}
	#print(new_names)
	df = df.rename(columns=new_names)
	df = df[new_names.values()]
	return df

# def add_non_covid_data(df):
# 	# ToDo : Refactor later to move all the merges after the rotate,
# 	# but for now this works...
#
# 	return df

def fetch_all(save_file=True,
				data_path='assets/',
				filename='full_df.csv',
				#constant_filename='constant_df.csv',
				threshold=1,
				purge=False):
	# indicator_names = {"SP.POP.TOTL" : "population",
	# 				"NY.GDP.MKTP.CD" : "gdp_total",
	# 				"SP.DYN.LE00.MA.IN" : "lifeExp_male",
	# 				"SP.DYN.LE00.FE.IN" : "lifeExp_female" }

	if purge:
		print(">> Purging cached files...")
		for file in [filename, 'CountryFixedEffects', 'covid_merged.csv',
					'owid_constants.csv', 'owid_covid_raw.csv', 'jh_clean.csv',
					'coronaVirus_global.csv','coronaVirus_US.csv', 'ISO_codes.csv']:
			remove_local_copy(file)
	fixedEffects_df = fetch_fixedEffects()

	if check_file_age(filename, threshold=threshold):
		full_df = pd.read_csv(f'{data_path}{filename}')
		return full_df, fixedEffects_df

	# Fetch John Hopkins data
	jh_df = fetch_jh_clean()
	# Fetch Our World in Data data, splitting out constant data
	owid_covid_df, owid_constant_df = fetch_owid()
	# Now, merge these two data sources together
	full_covid_df = merge_covid_sources(jh_df, owid_covid_df)
	full_df = generate_covid_features(full_covid_df)
	full_df.reset_index(inplace=True, drop=True)
	full_df = rename_select_columns(full_df)

	if save_file:
		full_df.to_csv(f"{data_path}{filename}", index=False)
	print("\t+ Complete.")
	return full_df, fixedEffects_df
#
# def calculate_world_totals(cv_merged_df):
# 	temp_df = cv_merged_df.head(0)
# 	for counter in ['confirmed_total', 'deaths_total', 'recovered_total', 'tested_total']:
# 		row = cv_merged_df.loc[(cv_merged_df['Counter'] == counter) &
# 						 (cv_merged_df['Province/State'].isna())].sum(numeric_only=True)
# 		row['Country/Region'] = 'World'
# 		row['Latitude'] = 0
# 		row['Longitude'] = 0
# 		row['ISO3166_alpha3'] = 'WLD'
# 		row['Counter'] = counter
# 		temp_df = temp_df.append(row, ignore_index=True)
# 	cv_merged_df = cv_merged_df.append(temp_df, ignore_index=True)
# 	return cv_merged_df

def check_file_age(filename, threshold=30, data_path='assets/'):
	threshold *= 86_400 # Convert days to seconds
	file_path = f"{data_path}{filename}"
	if os.path.exists(file_path):
		print(f">> File  [ {filename} ] found. Checking age... ")
		st=os.stat(file_path)
		mtime=time.time() - st.st_mtime
		if mtime > threshold:
			print(f"\t+ File [ {filename} ] exceeds age threshold. Refresh...")
			return False
		else:
			print(f"\t+ File  [ {filename} ] not expired. Load from disk...")
			return True
	else:
		print(f">> File [ {filename} ] not found. Fetching...")
		return False

def test_refresh():
	print(f"Refreshed at {dt.datetime.now()}")

def get_latest(full_df):
	latest = sorted(full_df['Date'].unique())[-1]
	snapshot = full_df.loc[full_df['Date'] == latest]
	snapshot = snapshot[['Country/Region',
						 'confirmed_total', 'confirmed',
						 'deaths_total', 'deaths',
						 'recovered_total', 'recovered',
						 'active_cases',
						 'confirmed_total_norm', 'deaths_total_norm']]
	snapshot = snapshot.sort_values('confirmed_total',ascending=False)
	return snapshot

def fetch_eu_states(add_efta=True):
	cc = coco.CountryConverter()
	efta_states = [{'ISO3':'NOR','name_short':'Norway','ISO2':'NO'},
			   {'ISO3':'ISL','name_short':'Iceland','ISO2':'IS'},
			   {'ISO3':'CHE','name_short':'Switzerland','ISO2':'CH'},
			  ]
	eu_states = cc.data.loc[cc.data.EU <= 2017][['ISO3', 'ISO2', 'name_short']]
	if add_efta:
		eu_states = eu_states.append(efta_states, ignore_index=True)
	eu_states.columns = ['ISO3166_a3','ISO3166_a2','CountryName']
	return eu_states


def fetch_fixedEffects(save_file=True,
						threshold=30,
						data_path='assets/',
						filename='CountryFixedEffects.csv'):
	if check_file_age(filename, threshold=threshold):
		return pd.read_csv(f'{data_path}{filename}')
	indicators = {'PAGGTOPY':{'unit':'PERSMYNB','name':'Phys_total'},
			  'MINUINFI':{'unit':'PERSMYNB','name':'Nurse_total'},
			  'HOPITBED':{'unit':'NOMBRENB','name':'HospBed_total'},}

	final_df = fetch_eu_states()[['ISO3166_a3', 'CountryName']]
	eurostat_df = fetch_eurostat_indicators().drop('CountryName', axis=1)
	oecd_indicators = get_oecd_indicators(indicators).drop('CountryName', axis=1)
	hofstede_df = load_cultural_dimensions_data().drop('CountryName', axis=1)
	di_df = fetch_democracy_index().drop('CountryName', axis=1)

	final_df = pd.merge(final_df, eurostat_df, how='left', on='ISO3166_a3')
	final_df = pd.merge(final_df, hofstede_df, how='left', on='ISO3166_a3')
	final_df = pd.merge(final_df, oecd_indicators, how='left', on='ISO3166_a3')
	final_df = pd.merge(final_df, di_df, how='left', on='ISO3166_a3')

	if save_file:
		final_df.to_csv(f"{data_path}{filename}", index=False)

	return final_df

def fetch_eurostat_indicators(save_file=True,
								threshold=30,
								data_path='assets/',
								filename='Eurostat_indicators_EU.csv'):
	if check_file_age(filename, threshold=threshold):
		return pd.read_csv(f'{data_path}{filename}')

	indicators = {'hlth_sha11_hp':{'name':'EUS_healthExp',
							   'year':2016,
							   'units':['PC_GDP','EUR_HAB'],
							   'providers':['TOTAL'],
							   'indicator':'icha11_hp'}}

	health_exp = fetch_health_exp(indicators) # This one has is a little more complicated...

	gdp_per_capita = fetch_economic_indicator('sdg_08_10', 'EUS_population', 2019, 'CLV10_EUR_HAB')
	govt_budget_surplus = fetch_economic_indicator('tec00127', 'EUS_govt_budget_surplus_pct_gdp', 2019, 'PC_GDP')
	govt_debt = fetch_economic_indicator('sdg_17_40', 'EUS_govt_debt_pct_gdp', 2019, 'PC_GDP')
	population_density = fetch_economic_indicator('tps00003', 'EUS_population_density', 2018, 'PER_KM2')
	propOver65 = fetch_economic_indicator('tps00028', 'EUS_popOver65_pct', 2019)
	singleHouseholds_pct = fetch_economic_indicator('tesov190', 'EUS_singleHouseholds_pct', 2018, 'A1', 'hhtyp')
	population = fetch_economic_indicator('tps00001', 'EUS_pop_mil', 2019, unit=None)

	final_df = fetch_eu_states()

	final_df = pd.merge(final_df, health_exp, how='left', on='ISO3166_a2')
	final_df = pd.merge(final_df, gdp_per_capita, how='left', on='ISO3166_a2')
	final_df = pd.merge(final_df, govt_budget_surplus, how='left', on='ISO3166_a2')
	final_df = pd.merge(final_df, govt_debt, how='left', on='ISO3166_a2')
	final_df = pd.merge(final_df, population_density, how='left', on='ISO3166_a2')
	final_df = pd.merge(final_df, propOver65, how='left', on='ISO3166_a2')
	final_df = pd.merge(final_df, singleHouseholds_pct, how='left', on='ISO3166_a2')
	final_df = pd.merge(final_df, population, how='left', on='ISO3166_a2')

	final_df.drop('ISO3166_a2', axis=1, inplace=True)

	if save_file:
		final_df.to_csv(f"{data_path}{filename}", index=False)

	return final_df

def fetch_health_exp(indicators):
	#Working rewrite of health care stats

	for code in indicators.keys():
		full_df = eurostat.get_data_df(code, flags=False)
		geo_entities = full_df['geo\\time'].unique()[:-7]
		units = indicators[code]['units']
		providers = indicators[code]['providers']
		indicator = indicators[code]['indicator']

		indicator_df = full_df.loc[(full_df['unit'].isin(units)) &
								  (full_df[indicator].isin(providers)) &
								  (full_df['geo\\time'].isin(geo_entities))].copy()

		year = indicators[code]['year']

		final_df = fetch_eu_states()['ISO3166_a2']
		for unit in units:
			indicator_name = f"EUS_{indicators[code]['name']}_{str.lower(unit)}_{year}"
			subset_df = indicator_df.loc[indicator_df['unit']==unit, ['geo\\time', year]]
			subset_df.columns = ['ISO3166_a2', indicator_name]
			# Fix Greece and the UK
			subset_df.loc[subset_df['ISO3166_a2'] == 'EL', 'ISO3166_a2'] = 'GR'
			subset_df.loc[subset_df['ISO3166_a2'] == 'UK', 'ISO3166_a2'] = 'GB'
			final_df = pd.merge(final_df, subset_df, how='left', on='ISO3166_a2')

	#final_df.drop('ISO3166_a2', axis=1, inplace=True)
	return final_df

def fetch_economic_indicator(code, name, year, unit=None, unit_name='unit'):
	full_df = eurostat.get_data_df(code, flags=False)
	geo_entities = full_df['geo\\time'].unique()[:-7]
	if unit is not None:
		full_df = full_df.loc[(full_df[unit_name] == unit) &
				(full_df['geo\\time'].isin(geo_entities))]
	else:
		full_df = full_df.loc[(full_df['geo\\time'].isin(geo_entities))]
	indicator_name = f"EUS_{name}_{year}"

	full_df = full_df[['geo\\time', year]]
	full_df.columns = ['ISO3166_a2', indicator_name]
	full_df.loc[full_df['ISO3166_a2'] == 'EL', 'ISO3166_a2'] = 'GR'
	full_df.loc[full_df['ISO3166_a2'] == 'UK', 'ISO3166_a2'] = 'GB'
	return full_df

def get_oecd_indicators(indicators,
						save_file=True,
						threshold=30,
						data_path='assets/',
						filename='OECD_healthIndicators_EU.csv'):
	if check_file_age(filename, threshold=threshold):
		return pd.read_csv(f'{data_path}{filename}')

	# Create list of countries that are in the EU and OECD
	cc = coco.CountryConverter()
	eu_states = fetch_eu_states()['ISO3166_a3']
	oecd_states = set(cc.data.loc[cc.data.OECD <= 2017]['ISO3'])
	eu_oecd_overlap = oecd_states.intersection(eu_states)
	country_str = "+".join(eu_oecd_overlap)

	# Request the indicators
	request_df = pd.DataFrame()
	for key in indicators.keys():
		json_req = f"HEALTH_REAC/{key}.{indicators[key]['unit']}.{country_str}/all"
		additional_parameters = "&startTime=2017-Q1&endTime=2017-Q4&dimensionAtObservation=allDimensions"
		temp_df = get_from_oecd(json_req,additional_parameters)
		request_df = pd.concat([request_df, temp_df])

	# Funge the resulting dataframe into the desired shape
	indicator_df = fetch_eu_states()[['ISO3166_a3', 'CountryName']]
	#indicator_df.columns = ['ISO3166_a3', 'CountryName']
	for idx, indicator in enumerate(request_df.VAR.unique()):
		subset_df = request_df.loc[request_df.VAR == indicator]
		name = indicators[indicator]['name']
		year = subset_df['Year'].unique()[0]
		name = f'OECD_{name}_{year}'
		subset_df = subset_df[['COU', 'Value']]
		subset_df.columns = ['ISO3166_a3', name]
		indicator_df = pd.merge(indicator_df, subset_df, how='left', on='ISO3166_a3')

	if save_file:
		indicator_df.to_csv(f"{data_path}{filename}", index=False)

	return indicator_df

def get_from_oecd(sdmx_query,
				  additional_parameters=""):
	df = pd.read_csv(f"https://stats.oecd.org/SDMX-JSON/data/{sdmx_query}?contentType=csv{additional_parameters}")
	return df

def load_cultural_dimensions_data(save_file=True,
								  threshold=30,
								  data_path='assets/',
								  filename='HCD_culturalDimensions_EU.csv'):
	if check_file_age(filename, threshold=threshold):
		return pd.read_csv(f'{data_path}{filename}')

	hofstede_url = "https://geerthofstede.com/wp-content/uploads/2016/08/6-dimensions-for-website-2015-08-16.csv"
	ua_str = fake_useragent.UserAgent().chrome
	r = requests.get(hofstede_url, headers={"User-Agent": ua_str})
	hofstede_df = pd.read_csv(io.StringIO(r.content.decode('utf-8')),
								sep=";",na_values="#NULL!")

	#ISO3_df = cc.data.loc[cc.data.EU <= 2017][['ISO3','name_short']]
	eu_states = fetch_eu_states()

	# Some countries are wrong, so fix manually...
	# It would be an idea to create a Hofstede to ISO3 code map
	country_pairs = {'Slovak Rep': 'Slovakia',
				 'Czech Rep' : 'Czech Republic',
				 'Great Britain' : 'United Kingdom'}
	for key in country_pairs.keys():
		hofstede_df.loc[hofstede_df['country'] == key,
							'country'] = country_pairs[key]

	hofstede_df = pd.merge(eu_states, hofstede_df, how='left',
						   left_on="CountryName", right_on="country")
	hofstede_df = hofstede_df[['ISO3166_a3', 'CountryName', 'pdi',
							   'idv', 'mas', 'uai', 'ltowvs', 'ivr']]

	hofstede_df.columns = ['ISO3166_a3',
						   'CountryName',
						   'HCD_PowerDistance',
						   'HCD_Individualism',
						   'HCD_Masculinity',
						   'HCD_UncertaintyAvoidance',
						   'HCD_LongTermOrientation',
						   'HCD_Indulgence']
	if save_file:
		hofstede_df.to_csv(f"{data_path}{filename}", index=False)

	return hofstede_df

def fetch_democracy_index(save_file=True,
							threshold=30,
							data_path='assets/',
							filename='EUI_democracyIndex_EU.csv'):
	if check_file_age(filename, threshold=threshold):
			return pd.read_csv(f'{data_path}{filename}')

	di_url = 'https://en.wikipedia.org/wiki/Democracy_Index'
	r = requests.get(di_url)
	soup = bs4.BeautifulSoup(r.content, 'html.parser')
	tables = soup.find_all("table")
	dict_list = []
	for idx, table in enumerate(tables):
		if ((table.find('caption') is not None) and
			(table.find('caption').get_text() == 'Democracy Index 2019\n')):
				rows = table.find_all('tr')
				for rdx, row in enumerate(rows[1:168]):
					row_dict = {}
					columns = row.find_all('td')
					try:
						row_dict['DI_Rank'] = int(columns[0].get_text())
						offset = 1
					except:
						row_dict['DI_Rank'] = rdx - 1
						offset = 0
					row_dict['DI_Country'] = columns[0+offset].get_text().strip('\xa0')
					row_dict['DI_Overall'] = float(columns[1+offset].get_text())
					row_dict['DI_EPP'] = float(columns[2+offset].get_text())
					row_dict['DI_FoG'] = float(columns[3+offset].get_text())
					row_dict['DI_PP'] = float(columns[4+offset].get_text())
					row_dict['DI_PC'] = float(columns[5+offset].get_text())
					row_dict['DI_CL'] = float(columns[6+offset].get_text())
					row_dict['DI_Regime'] = columns[7+offset].get_text()
					#row_dict['DI_Region'] = columns[8+offset].get_text().strip('\n')
					dict_list.append(row_dict)
	di_df = pd.DataFrame(dict_list)

	eu_states = fetch_eu_states()
	di_df = pd.merge(eu_states, di_df, how='left',
					 left_on='CountryName', right_on='DI_Country')

	sel_cols = ['ISO3166_a3', 'CountryName', 'DI_Rank', 'DI_Overall', 'DI_EPP',
				'DI_FoG', 'DI_PP', 'DI_PC', 'DI_CL', 'DI_Regime']
	di_df = di_df[sel_cols]

	di_df.columns = ['ISO3166_a3', 'CountryName', 'DI_Rank', 'DI_Overall', 'DI_EPP',
				'DI_FoG', 'DI_PP', 'DI_PC', 'DI_CL', 'DI_Regime']
	if save_file:
		di_df.to_csv(f"{data_path}{filename}", index=False)

	return di_df


if __name__ == '__main__':
	fixedEffects_df = fetch_fixedEffects()

	# indicator_names = {"SP.POP.TOTL" : "population",
	#				"NY.GDP.MKTP.CD" : "gdp_total",
	#				"SP.DYN.LE00.MA.IN" : "lifeExp_male",
	#				"SP.DYN.LE00.FE.IN" : "lifeExp_female" }
	#
	# indicator_df = fetch_indicators(list(indicator_names.keys()))

	#full_df, cv_merged_df, iso_codes_df, indicator_df = fetch_all()
	# snapshot = get_latest(full_df)
	# print(full_df.shape[0])
	# print(snapshot.shape[0])
