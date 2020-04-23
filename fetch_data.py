import pandas as pd
import numpy as np
import urllib
import urllib.request
import ssl
from bs4 import BeautifulSoup
import json
import os
import time
import datetime as dt

ssl._create_default_https_context = ssl._create_unverified_context

def fetch_cv_global(save_file=True,
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
             filename='coronaVirus_US.csv'):
    if check_file_age(filename, threshold=1):
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

    population = np.nan
    if 'population' in df.columns:
        population = df.loc[(df['Country/Region'] == country),
                'population'].values[0]
    # else:
    #     print(f"{country} : population missing")
    #     #print(sf.columns)

    col_vars = ['confirmed', 'deaths', 'recovered']

    for cv in col_vars:
        sf[f'{cv}'] = sf[f'{cv}_total'].diff()
        try:
            sf[f'{cv}_total_norm'] = (sf[f'{cv}_total'] / population) * 1e6
        except:
            sf[f'{cv}_total_norm'] = np.nan
        sf[f'{cv}_log_total'] = np.log2(sf[f'{cv}_total'])
        sf[f'{cv}_log'] = sf[f'{cv}_log_total'].diff()
        sf[f'{cv}_log_growth'] = np.power(2, sf[f'{cv}_log']) - 1

    sf['net_cases'] = sf['confirmed'] - sf['deaths'] - sf['recovered']
    sf['active_cases'] = sf['net_cases'].cumsum()
        # np.exp(np.diff(np.log(population))) - 1
        #sf[f'{cv}_diff_2'] = sf[f'{cv}_cum'].diff(2)
        #sf[f'{cv}_log_cum'] = np.log(sf[f'{cv}_cum'])
        #sf[f'{cv}_log_diff'] = np.log(sf[f'{cv}_diff'])


    return sf

def plot_df(df, days=30):
    plt.title("Per day")
    plt.plot(df.index[-days:], df['confirmed'][-days:], label="Confirmed")
    plt.plot(df.index[-days:], df['deaths'][-days:], label="Deaths")
    plt.plot(df.index[-days:], df['recovered'][-days:], label="Recovered")
    plt.legend()
    plt.show()

#     plt.title("2nd diff")
#     plt.plot(df.index[-days:], df['confirmed_diff_2'][-days:], label="Confirmed")
#     plt.plot(df.index[-days:], df['deaths_diff_2'][-days:], label="Deaths")
#     plt.plot(df.index[-days:], df['recovered_diff_2'][-days:], label="Recovered")
#     plt.legend()
#     plt.show()

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

#     max_region_st = 0
    for c in countries:
        country_df = build_frame(df, c)
        region_s = country_df[plot_var]
        trim_idx = bisect.bisect_left(region_s,threshold)
        region_st = np.array(list(region_s[trim_idx:].values) + [np.nan]*trim_idx)
        plt.plot(region_st-threshold, label=c)
#         if region_st > max_region_st:
#             max_region_st = region_st
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
    soup = BeautifulSoup(source_page, 'html.parser')
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

def merge_dataframes(cv_fixed_df, iso_codes_df, indicator_df,
                    indicator_names):
    first_date = cv_fixed_df.columns.get_loc("1/22/20")
    last_date = len(cv_fixed_df.columns)

    cv_merged_df = pd.merge(cv_fixed_df, iso_codes_df, how='left',
                            left_on='Country/Region', right_on='name')

    first = True
    for indicator in indicator_names.keys():
        indicator_name = indicator_names[indicator]
        sub_df = indicator_df.loc[indicator_df['indicator_id'] == indicator].copy()
        sub_df.rename(columns={'value':indicator_name,
                               'countryiso3code':'ISO3166_alpha3',
                               'country_name':'DisplayName'}
                      ,inplace=True)
        sub_df[f'{indicator_name}_notes'] = sub_df['indicator_name'] + \
                        " in " + sub_df['date'] + " (" + \
                        sub_df['indicator_id'] + ")"

        column_list = ['ISO3166_alpha3', indicator_name,
                       f'{indicator_name}_notes']
        if first:
            column_list += ['DisplayName']
            first = False
        sub_df = sub_df[column_list]
        cv_merged_df = pd.merge(cv_merged_df, sub_df, how='left',
                            on='ISO3166_alpha3')


    new_order = list(cv_merged_df.columns[0:first_date]) + \
                    list(cv_merged_df.columns[last_date:]) + \
                    list(cv_merged_df.columns[first_date:last_date])
    cv_merged_df = cv_merged_df[new_order]

    return cv_merged_df

def remove_local_copy(filename, data_path='assets/'):
    filepath = f"{data_path}{filename}"
    if os.path.exists(filepath):
        print(f"\t+ Removing file [ {filepath} ]...")
        os.remove(filepath)
    else:
        print(f"\t+ File [ {filepath} ] not found. Cannot remove...")

def fetch_components(indicator_names,
                    save_file=True,
                    data_path='assets/',
                    filename='merged_df_global.csv',
                    threshold=1,):
    if check_file_age(filename, threshold=threshold):
        cv_merged_df = pd.read_csv(f'{data_path}{filename}')
        iso_codes_df = fetch_ISO_codes()
        indicator_df = fetch_indicators(list(indicator_names.keys()))
        return cv_merged_df, iso_codes_df, indicator_df


    cv_df = fetch_cv_global()
    iso_codes_df = fetch_ISO_codes() # Using CIA World Factbook page
    print(">> Before name fixes:")
    check_iso(cv_df, iso_codes_df)
    cv_fixed_df = fix_names(cv_df)
    cv_fixed_df = fix_ships(cv_fixed_df)
    print(">> After name fixes:")
    check_iso(cv_fixed_df, iso_codes_df)
    print(">> Before summarise countries:")
    test_countries(cv_fixed_df)
    cv_fixed_df = summarise_multiregion(cv_fixed_df)
    cv_fixed_df = cv_fixed_df.loc[cv_fixed_df['Country/Region'] != 'Other']
    print(">> After summarise countries:")
    test_countries(cv_fixed_df)

    print(">> Fetching World Bank Indicators...")
    indicator_df = fetch_indicators(list(indicator_names.keys()))
    indicator_df.loc[indicator_df['countryiso3code'] == 'USA']
    print(">> Merging data...")
    cv_merged_df = merge_dataframes(cv_fixed_df, iso_codes_df,
                                indicator_df, indicator_names)
    cv_merged_df = calculate_world_totals(cv_merged_df)

    if save_file:
        full_path = f"{data_path}{filename}"
        print(f"\t+ Saving to [ {full_path} ]")
        cv_merged_df.to_csv(full_path, index=False)

    return cv_merged_df, iso_codes_df, indicator_df

def build_full_df(cv_df):
    countries = cv_df['Country/Region'].unique()
    first = True

    cv_df = cv_df.sort_values('Counter') # Sometimes the counters are in a different order, this fixes the cat problem

    for country in countries:
        if first:
            full_df = build_frame(cv_df, country)
            first = False
        else:
            temp_df = build_frame(cv_df, country)
            full_df = pd.concat([full_df, temp_df])

    full_df['Date'] = pd.to_datetime(full_df.index)
    full_df.reset_index(inplace=True, drop=True)

    merge_columns = [ 'ISO3166_alpha3', 'population', 'gdp_total',
                  'lifeExp_male', 'lifeExp_female', 'DisplayName']
    cv_df = cv_df[merge_columns]
    full_df = pd.merge(full_df, cv_df, how='inner', on='ISO3166_alpha3')
    # Fix the world name
    full_df.loc[full_df["ISO3166_alpha3"] == 'WLD', "Name"] = 'World'

    # ToDo : Refactor later to move all the merges after the rotate,
    # but for now this works...
    full_df = full_df.drop_duplicates()

    # # ToDo : put this in a file to allow it to be pulled out easily
    new_names = {"Country/Region" : "Country/Region",
            "Province/State" : "Province/State",
            "ISO3166_alpha2" : "ISO3166:2",
            "ISO3166_alpha3" : "ISO3166:3",
            "ISO3166_numeric" : "ISO3166:#",
            "DisplayName" : "Name",
            "population" : "Population",
            "gdp_total" : "GDP (Total)",
            "lifeExp_male" : "Life Expectancy (Male)",
            "lifeExp_female" : "Life Expectancy (Female)",
            "Date" : "Date",
            "confirmed_total" : "Confirmed (Total)",
            "deaths_total" : "Deaths (Total)",
            "recovered_total" : "Recovered (Total)",
            "active_cases" : "Active (Total)",
            "confirmed_total_norm" : "Confirmed (Total, normalised)",
            "deaths_total_norm" : "Deaths (Total, normalised)",
            "recovered_total_norm" : "Recovered (Total, normalised)",
            "confirmed_log_total" : "Confirmed (Total, log)",
            "deaths_log_total" : "Deaths (Total, log)",
            "recovered_log_total" : "Recovered (Total, log)",
            "confirmed" : "Confirmed (Daily)",
            "deaths" : "Deaths (Daily)",
            "recovered" : "Recovered (Daily)",
            "net_cases" : "Net Cases (Daily)",
            "confirmed_log" : "Confirmed (Daily, log)",
            "deaths_log" : "Deaths (Daily, log)",
            "recovered_log" : "Recovered (Daily, log)",
            "confirmed_log_growth" : "Confirmed (Daily, log growth)",
            "deaths_log_growth" : "Deaths (Daily, log growth)",
            "recovered_log_growth" : "Recovered (Daily, log growth)"
            }
    full_df = full_df.rename(columns=new_names)
    full_df = full_df[new_names.values()]

    return full_df

def fetch_all(save_file=True,
                data_path='assets/',
                filename='full_df.csv',
                threshold=1,
                purge=False):
    indicator_names = {"SP.POP.TOTL" : "population",
                   "NY.GDP.MKTP.CD" : "gdp_total",
                   "SP.DYN.LE00.MA.IN" : "lifeExp_male",
                   "SP.DYN.LE00.FE.IN" : "lifeExp_female" }

    if purge:
        print(">> Purging cached files...")
        for file in ['full_df.csv', 'merged_df_global.csv',
                     'coronaVirus_global.csv', 'coronaVirus_US.csv',
                     'ISO_codes.csv', 'wb_indicators.csv']:
            remove_local_copy(file)

    cv_merged_df, iso_codes_df, indicator_df = fetch_components(indicator_names)

    if check_file_age(filename, threshold=threshold):
        full_df = pd.read_csv(f'{data_path}{filename}')
        return full_df, cv_merged_df, iso_codes_df, indicator_df

    print(">> Building full df...")
    full_df = build_full_df(cv_merged_df)

    if save_file:
        full_df.to_csv(f"{data_path}{filename}", index=False)
    print("\t+ Complete.")
    return full_df, cv_merged_df, iso_codes_df, indicator_df

def calculate_world_totals(cv_merged_df):
    temp_df = cv_merged_df.head(0)
    for counter in ['confirmed_total', 'deaths_total', 'recovered_total']:
        row = cv_merged_df.loc[(cv_merged_df['Counter'] == counter) &
                         (cv_merged_df['Province/State'].isna())].sum(numeric_only=True)
        row['Country/Region'] = 'World'
        row['Latitude'] = 0
        row['Longitude'] = 0
        row['ISO3166_alpha3'] = 'WLD'
        row['Counter'] = counter
        temp_df = temp_df.append(row, ignore_index=True)
    cv_merged_df = cv_merged_df.append(temp_df, ignore_index=True)
    return cv_merged_df

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

if __name__ == '__main__':

    # indicator_names = {"SP.POP.TOTL" : "population",
    #                "NY.GDP.MKTP.CD" : "gdp_total",
    #                "SP.DYN.LE00.MA.IN" : "lifeExp_male",
    #                "SP.DYN.LE00.FE.IN" : "lifeExp_female" }
    #
    # indicator_df = fetch_indicators(list(indicator_names.keys()))

    full_df, cv_merged_df, iso_codes_df, indicator_df = fetch_all()
    # snapshot = get_latest(full_df)
    # print(full_df.shape[0])
    # print(snapshot.shape[0])
