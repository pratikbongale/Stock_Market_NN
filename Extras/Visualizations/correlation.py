from __future__ import division
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pandas_datareader import DataReader

# creating dataframe for apple, google, microsoft and amazon stocks
AAPL = []
GOOG = []
MSFT = []
AMZN = []

sns.set_style('whitegrid')

#generating list for stocks
listOfStocks = ['GOOG', 'AAPL', 'AMZN', 'MSFT']

#initializing end time
endTime = datetime.now()
#initializing start time
startTime = datetime(endTime.year - 5, endTime.month, endTime.day)

#generating global list containing global variable for each stock
for i in listOfStocks:
    globals()[i] = DataReader(i, 'yahoo', startTime, endTime)

#plotting graph for the volumn column of apple stock
AAPL['Volume'].plot(legend=True, figsize=(10, 4))
#plotting graph for the adjusted close for apple stock
AAPL['Adj Close'].plot(legend=True, figsize=(10, 4))

#moving average for 10 days, 20 days, 30 days
moving_average_day = [10, 20, 30]

for moving_average in moving_average_day:
    columnName = "MA for " + str(moving_average) + "days"
    #pandas provide the functionality to find the moving averge using rolling
    AAPL[columnName] = AAPL['Adj Close'].rolling(window=5, center=False).mean()

#we calculate daily return by finding the percentage change
AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()

AAPL['Daily Return'].plot(figsize=(12, 4), legend=True, linestyle='--', marker='o')

closing_dataframe = DataReader(['AAPL', 'GOOG', 'MSFT', 'AMZN'], 'yahoo', startTime, endTime)['Adj Close']

closing_dataframe.head()

stock_returns = closing_dataframe.pct_change()

sns.jointplot('AMZN', 'GOOG', stock_returns, kind='scatter')

sns.pairplot(stock_returns.dropna())

fig = sns.PairGrid(closing_dataframe)
fig.map_upper(plt.scatter, color='purple')
fig.map_lower(sns.kdeplot, cmap='cool_d')
fig.map_diag(plt.hist, bins=30)

fig = sns.PairGrid(stock_returns.dropna())
fig.map_upper(plt.scatter, color='blue')
fig.map_lower(sns.kdeplot, cmap='cool_d')
fig.map_diag(plt.hist, bins=50)

#plotting the correlation table using heatmap for analysis
corr = stock_returns.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
