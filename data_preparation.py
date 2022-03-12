#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 17:28:03 2022

@author: DanBickelhaupt
"""

# %% Packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor

# %% Data input & initial organization functions
def importPriceData(main,filename):
    # Set columns from WRDS price dataset to import
    col_list = ['COMNAM','TSYMBOL','date','PRC','VOL']
    # Import price dataset
    price_data = pd.read_csv(main+filename,usecols=col_list)
    # Convert date to useable format
    price_data['date'] = pd.to_datetime(price_data['date'],format='%Y%m%d')
    # Rename columns
    price_data = price_data.rename(columns={'COMNAM':'CompanyName','TSYMBOL':'Ticker','date':'Date','PRC':'Price','VOL':'Volume'})
    # Load SimFin dataset for 2021 prices
    finsim_price_dataloc = '/Users/DanBickelhaupt/Desktop/Coding/AI_Investor_Book/SimFin_Free_Data/us-shareprices-daily.csv'
    recent_prices = pd.read_csv(finsim_price_dataloc,delimiter=';',usecols=['Ticker','Date','Close','Volume'])
    recent_prices['Date'] = pd.to_datetime(recent_prices['Date'])
    recent_prices = recent_prices.rename(columns = {'Close':'Price'})
    recent_prices = recent_prices[recent_prices['Date'].dt.year == 2021]
    # Merge datasets
    price_data = pd.concat([price_data,recent_prices],axis=0)
    # Remove bad rows
    bool_list = ~(price_data['Price'] < 0 | price_data['Ticker'].isnull() | price_data['Date'].isnull() | price_data['Price'].isnull() | price_data['Volume'].isnull())
    price_data = price_data[bool_list]
    bool_list = ~(price_data['Ticker'].isna() | price_data['Date'].isna() | price_data['Price'].isna() | price_data['Volume'].isna())
    price_data = price_data[bool_list]
    # Return dataframe
    price_data = price_data.reset_index(drop=True)
    return price_data
def importFundData(main,filename):
    # Set columns from Compustat price dataset to import
    col_list = ['conm','tic','fyear','datadate','ni','xint','txt','csho','dltt','dlc','che','oiadp','act','lct','ppent','ceq','sale','at','lt','cogs','seq','oancf','capx','re']
    # Import fundamental dataset
    funda_data = pd.read_csv(main+filename,usecols=col_list)
    # Convert date to useable format
    funda_data['datadate'] = pd.to_datetime(funda_data['datadate'],format='%Y%m%d')
    # Rename columns
    funda_data = funda_data.rename(columns={'conm':'CompanyName','tic':'Ticker','fyear':'Year','datadate':'StatementDate',\
                                       'ni':'NetIncome','xint':'NetInterestExpense','txt':'NetIncomeTax','csho':'CommonSharesOutstanding',\
                                       'dltt':'LongTermDebt','dlc':'ShortTermDebt','che':'Cash','oiadp':'OpIncome','act':'TotalCurrentAssets',\
                                       'lct':'TotalCurrentLiabilities','ppent':'NetPPE','ceq':'TotalEquity','sale':'Revenue',\
                                       'at':'TotalAssets','lt':'TotalLiabilities','cogs':'COGS','seq':'ShareholderEquity',\
                                       'oancf':'OpCashFlow','capx':'CapEx','re':'RetainedEarnings'})
    # Add EBIT
    funda_data['EBIT'] = funda_data['NetIncome'] - funda_data['NetInterestExpense'] - funda_data['NetIncomeTax']
    # Fix NaNs
    for key in funda_data.keys():
        funda_data.loc[funda_data[key].isnull() | funda_data[key].isna(), key] = 0
    # Return dataframe
    return funda_data
def getPriceNearDate(ticker,date,modifier,p_data):
    buffer = 30 # number of days after (date+modifier) to search for a valid price datapoint
    search = p_data[(p_data['Date'].between(pd.to_datetime(date)+pd.Timedelta(days=modifier),pd.to_datetime(date)+pd.Timedelta(days=buffer+modifier))) & (p_data['Ticker']==ticker)]
    if search.empty:
        return [ticker , np.float('NaN') , np.datetime64('NaT') , np.float('NaN')]
    else:
        return [ticker , search.iloc[0]['Price'] , search.iloc[0]['Date'] , search.iloc[0]['Volume']]
def getStatementPriceFast(fin_data,p_data,delay=0):
    # Pre-allocate output matrix
    y = [[None]*4 for i in range(len(fin_data))]
    # Set date column heading to interrogate
    dateColHeading = 'StatementDate'
    # Setup to search by year (which is hopefully faster)
    fin_data['StatementDate'] = pd.to_datetime(fin_data['StatementDate'])
    fyears = fin_data.Year.unique()
    # Loop thru all years in 'fin_data'
    i = 0
    for year in fyears:
        # Separate out a subset of financial statement data that's just for the current year.
        bool_list = (fin_data['StatementDate'].dt.year == year)
        sub_fin_data = fin_data[bool_list] 
        # Separate out a subset of price data that's for the current year thru 2 years from current year.
        bool_list = ((p_data['Date'].dt.year == year) | (p_data['Date'].dt.year == year+1) | (p_data['Date'].dt.year == year+2))
        sub_p_data = p_data[bool_list]
        # Index thru each finanical statment datapoint in the current year 
        for index in range(len(sub_fin_data)):
            y[i] = (getPriceNearDate(sub_fin_data['Ticker'].iloc[index],sub_fin_data[dateColHeading].iloc[index],delay,sub_p_data))
            i += 1            
            if i % round(len(fin_data)/100) == 0:
                print('Statement price pull {:0.0f}% complete.'.format(i/len(fin_data)*100))
        # Append index to maintain orginal order of fin_data dataframe
        if year == fyears[0]:
            # If first iteration thru, make new index variable
            save_idx_order = sub_fin_data.index
        else:
            save_idx_order = save_idx_order.append(sub_fin_data.index)
    y = pd.DataFrame(y,index=save_idx_order,columns=['Ticker','Price','Date','Volume'])
    y = y.sort_index()
    return y
def cullData(f_data,p_data):
    # Create copies of input data
    culled_f_data = f_data.copy()
    culled_p_data = p_data.copy()
    # Cull data
    bool_list = ~(culled_p_data['Price'].isna())
    culled_f_data = culled_f_data[bool_list]
    culled_p_data = culled_p_data[bool_list]
    bool_list = ~(culled_f_data['CommonSharesOutstanding'].isna())
    culled_f_data = culled_f_data[bool_list]
    culled_p_data = culled_p_data[bool_list]
    bool_list = ~(culled_f_data['CommonSharesOutstanding']== 0)
    culled_f_data = culled_f_data[bool_list]
    culled_p_data = culled_p_data[bool_list]
    bool_list = ~(culled_p_data['Date'].isna())
    culled_f_data = culled_f_data[bool_list]
    culled_p_data = culled_p_data[bool_list]
    # Reset indices
    culled_f_data = culled_f_data.reset_index(drop=True)
    culled_p_data = culled_p_data.reset_index(drop=True)
    # Return datasets
    return culled_f_data , culled_p_data
def addMarketCapEV(data,price):
    data['MarketCap'] = price['Price'] * data['CommonSharesOutstanding']
    data['MarketCapDate'] = price['Date']
    data['EV'] = data['MarketCap'] + data['LongTermDebt'] + data['ShortTermDebt'] - data['Cash']
    return data

# %% Data Preparation Functions
def print_feature_stats(data, feature_list):
    for feat in feature_list:
    #Print statistics for each feature of the data
        print('Feature = "{}"'.format(feat))
        print('\tMax = {}'.format(data[feat].max()))
        print('\tMin = {}'.format(data[feat].min()))
        print('\tMean = {}'.format(data[feat].mean()))
        print('\tMedian = {}'.format(data[feat].median()))
        print('\tStd Dev = {}'.format(data[feat].std()))
        print()
def min_max_transform(data, feature_list, a=0, b=1):
    data_trans = data.copy()    
    for feat in feature_list:
        data_trans[feat] = (data_trans[feat] - data_trans[feat].min()) / (data_trans[feat].max() - data_trans[feat].min()) * (b - a) - a
    return data_trans
def z_score_transform(data, feature_list):
    transform_parameters = np.zeros((len(feature_list), 2))
    data_trans = data.copy()
    for i, feat in enumerate(feature_list):
         transform_parameters[i, 0] = data[feat].mean()
         transform_parameters[i, 1] =  data[feat].std()
         data_trans[feat] = (data_trans[feat] - transform_parameters[i, 0]) / transform_parameters[i, 1]
    transform_parameters = pd.DataFrame(transform_parameters, index=feature_list, columns=['Mean','Stdev'])
    return data_trans, transform_parameters
def compile_nplus_results(data, N):
    in_idx = []
    ou_idx = []
    companies = data['Ticker'].unique()
    for i, company in enumerate(companies):
        cur_data = data.loc[data['Ticker'] == company]
        in_idx.extend(cur_data.index[0:-N])
        ou_idx.extend(cur_data.index[N:])
        if i % round(len(companies)/20) == 0:
            print('N-plus results {:0.0f}% complete.'.format(i/len(companies)*100))
    in_data = data.loc[in_idx].reset_index(drop=True)
    ou_data = data.loc[ou_idx].reset_index(drop=True)
    return in_data, ou_data

# %% Export prepared dataset to .csv files

# Import datasets
data_repo = '/Users/DanBickelhaupt/Desktop/Coding/WRDS_Datasets/'
price_file = 'crsp_monthly_price_dataset.csv'
raw_price_data = importPriceData(data_repo,price_file)
funda_file = 'compustat_annual_fundamental_dataset.csv' 
funda_data = importFundData(data_repo,funda_file)
# Get price data for statement dataset
est_statement_release_delay = 55 # estimated time between statement date & publish date
price_data = getStatementPriceFast(funda_data, raw_price_data, est_statement_release_delay)
# Cull bad rows from dataset, add MarketCap & EV to dataset
funda_data , price_data = cullData(funda_data, price_data)
funda_data = addMarketCapEV(funda_data, price_data)
# Export datasets to .csv
price_data.to_csv('Annual_Statement_Price_Data.csv', index=False)    
funda_data.to_csv('Annual_FinStatement_Data.csv', index=False)
