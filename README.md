## Description
The US stock market is best represented by Standards & Poor’s(S&P 500) indices based on the market capitalizations of 500 companies having common stock listed on the NYSE or NASDAQ.

This project uses Neural Networks(built using Tensorflow) to predict whether S&P index returns will rise or fall based on changes in stock prices of top 5 companies (Microsoft, Amazon, Facebook, Google, Apple). 

Further details about the project can be found in [this](Extras/Report_Stock_Market_Analysis.pdf) report and [this](Extras/Stock_Market_Prediction.pptx) presentation.

## Dataset
Currently the dataset is provided as part of the code. You can look for clean filtered dataset with S&P returns data for top 5 companies in the dataset file [data_stocks.csv](data_stocks.csv). 

I am working on building a online learning model which will pull most recent stock price changes(per minute) using API calls to quandl.com and finance.yahoo.com. 

## How to run the code
Syntax: python <script_name> <dataset_file> <num_epochs>
```commandline
$ python predictor_main.py data_stocks.csv 256
```

## Output
Output plots representing how well the model fits given data is stored in a folder `img`. \
`img/ideal_pred` : Ideal prediction expected.\
`img/final_pred` : Final prediction by the neural network model.