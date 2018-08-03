## Description
The US stock market is best represented by Standards & Poor’s(S&P 500) indices based on the market capitalization of 500 companies having common stock listed on the NYSE or NASDAQ.

This project uses Neural Networks(built using Tensorflow) to predict whether S&P index returns will rise or fall based on changes in stock prices of top 5 companies (Microsoft, Amazon, Facebook, Google, Apple). 

Further details about the project can be found in [this](Extras/Report_Stock_Market_Analysis.pdf) report and [this](Extras/Stock_Market_Prediction.pptx) presentation.

## Dataset
Currently the dataset is provided as part of the code. You can look for clean filtered dataset with S&P returns data for top 5 companies in the dataset file [data_stocks.csv](data_stocks.csv). 

Coming soon: I am working on building a online learning model which will pull most recent stock price changes(per minute) using API calls to quandl.com and finance.yahoo.com. 

## How to run the code
Syntax: python <script_name> <dataset_file> <num_epochs> \[debug_at_epoch]
```commandline
$ python predictor_main.py data_stocks.csv 100
```

You can see the progress of model training using the debug option. You can specify a number (example. 25) to see how well the model has fit the test dataset after every 25 epochs. 
Note: If debug is on, plots will be shown upfront and you will have to close them manually.
```commandline
$ python predictor_main.py data_stocks.csv 100 25
```

## Output
1. Commandline outputs SSE(Sum of Squared Errors).
2. Output plots representing how well the model fits given data is stored in a folder `img`. \
`img/ideal_pred.png` : Ideal prediction expected.\
`img/final_pred.png` : Final prediction by the neural network model.\
`img/pred_error.png` : Difference between ground truth and predicted output.