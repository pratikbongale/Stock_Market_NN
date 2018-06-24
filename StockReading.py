import datetime as dt
import pandas as p
import pandas_datareader.data as webData
import numpy as npy
import matplotlib.pyplot as plot
from matplotlib import style
from datetime import datetime
import seaborn as sea
sea.set_style('whitegrid')

style.use("ggplot")
excel = []
starttime = dt.datetime(1996, 1, 1)
endtime = dt.datetime(2017, 11, 23)

def roundingUp(x):
    return round(x, 1)

def riskInvestment(company, datasheet):
    print("Risk Investment")
    starttime = dt.datetime(1996, 1, 1)
    endtime = dt.datetime(2017, 11, 23)
    '''
    df = []
    frames = []
    for i in datasheet:
        df = p.read_excel(i, sheet_name='Sheet1')
        frames.append(df)
    result = p.concat(frames)
    print(result)
    '''
    closingDataFrame = webData.DataReader(['MSFT', 'AAPL', 'IBM', 'GOOG', 'AMZN'], 'yahoo', starttime, endtime)['Adj Close']

    pctChange = closingDataFrame.pct_change()
    #print(pctChange)
    cleanPctChange = pctChange.dropna()
    print(cleanPctChange)
    plot.xlabel("Expected Returns")
    plot.ylabel("Risk")
    plot.scatter(cleanPctChange.mean(), cleanPctChange.std(), alpha=0.5)
    for label, x, y in zip(cleanPctChange.columns, cleanPctChange.mean(), cleanPctChange.std()):
        plot.annotate(label, xy=(x, y), ha='right', va='bottom')
    plot.title("Investment Risk Analysis")
    plot.show()
    quantile = 0.1
    appleQuantile = cleanPctChange['AAPL'].quantile(quantile)
    print("The empirical quantile of", quantile,"for Apple stocks is", appleQuantile)
    msftQuantile = cleanPctChange['MSFT'].quantile(quantile)
    print("The empirical quantile of", quantile,"for Microsoft stocks is", msftQuantile)
    ibmQuantile = cleanPctChange['IBM'].quantile(quantile)
    print("The empirical quantile of", quantile,"for IBM stocks is", ibmQuantile)
    googQuantile = cleanPctChange['GOOG'].quantile(quantile)
    print("The empirical quantile of", quantile,"for Google stocks is", googQuantile)
    amznQuantile = cleanPctChange['AMZN'].quantile(quantile)
    print("The empirical quantile of", quantile,"for Amazon stocks is", amznQuantile)
    investmentAmount = int(input("Enter the amount in digits to be invested in stock:"))
    appl = investmentAmount * abs(appleQuantile)
    msft = investmentAmount * abs(msftQuantile)
    ibm = investmentAmount * abs(ibmQuantile)
    goog = investmentAmount * abs(googQuantile)
    amzn = investmentAmount * abs(amznQuantile)
    print("Results for Risk Investment in IT stocks:")
    print("If you invest $", investmentAmount, "in Apple stocks than you hold a risk of $", appl,".")
    print("If you invest $", investmentAmount, "in Microsoft stocks than you hold a risk of $", msft, ".")
    print("If you invest $", investmentAmount, "in IBM stocks than you hold a risk of $", ibm, ".")
    print("If you invest $", investmentAmount, "in Google stocks than you hold a risk of $", goog, ".")
    print("If you invest $", investmentAmount, "in Amazon stocks than you hold a risk of $", amzn, ".")

def getCompanyData(companyName):
    dataSheet = ""
    print("Fetching Data for ", companyName)
    starttime = dt.datetime(1996, 1, 1)
    endtime = dt.datetime(2017, 11, 23)
    dataSheet = webData.DataReader(companyName, 'quandl', starttime, endtime)
    '''
    df = p.DataFrame(data=dataSheet)
    columns = []
    columns.append("Date")
    for i in list(df):
        columns.append(i)
    print(df['Date'])
    '''
    excelName = companyName + '.xlsx'
    excel.append(excelName)
    dataSheet.to_excel(excelName)

def transformData(companyName, excelfile):
    print("Data Transformation for", companyName)
    dataSheet = p.read_excel(excelfile, parse_dates=True, index_col=0)
    dataSheet = dataSheet.drop_duplicates()
    previousClose = dataSheet["Close"].shift(1)
    dataSheet["%Gain/ Loss"] = ((dataSheet["Close"] - dataSheet["Open"])/previousClose)*100
    dataSheet["90DayMA"] = dataSheet["AdjClose"].rolling(window=90).mean()
    excelFile = "Transform_Unclean_" + excelfile
    dataSheet.to_excel(excelFile)

def dataPreProcessing(companyName, excelfile):
    print("Data Pre-Processing for", companyName)
    dataSheet = p.read_excel("Transform_Unclean_" + excelfile, parse_dates=True, index_col=0)
    previousClose = dataSheet["Close"].shift(1)
    dataSheet["%Gain/ Loss"] = ((dataSheet["Close"] - dataSheet["Open"]) / previousClose) * 100
    dataSheet["90DayMA"] = dataSheet["AdjClose"].rolling(window=90, min_periods=0).mean()

    ######### DATA FORMATTING ##########

    decimal = 2
    dataSheet['Open'] = dataSheet['Open'].apply(lambda x: round(x, decimal))
    dataSheet['Close'] = dataSheet['Close'].apply(lambda x: round(x, decimal))
    dataSheet['High'] = dataSheet['High'].apply(lambda x: round(x, decimal))
    dataSheet['Low'] = dataSheet['Low'].apply(lambda x: round(x, decimal))
    dataSheet['%Gain/ Loss'] = dataSheet['%Gain/ Loss'].apply(lambda x: round(x, 3))
    dataSheet['90DayMA'] = dataSheet['90DayMA'].apply(lambda x: round(x, 3))
    dataSheet.to_excel("Transform_Clean_" + excelfile)
    dataSheet['Open'].plot()
    plot.show()
    return "Transform_Clean_" + excelfile

def monteCarloPrediction(startprice, name, days, mean, std, delta):
    print("Monte Carlo Stock Prediction for", name,"stocks.")
    priceArray = npy.zeros(days)
    sh = npy.zeros(days)
    change = npy.zeros(days)
    priceArray[0] = startprice
    for i in range(1, days):
        change[i] = mean * delta
        sqrdelta = npy.sqrt(delta)
        sh[i] = npy.random.normal(scale = std * sqrdelta, loc = mean * delta)
        sum = sh[i] + change[i]
        priceArray[i] = (priceArray[i-1] * sum) + priceArray[i-1]
    return priceArray

def doRiskPrediction():
    company = int(input(
        "Enter the company name number for Monte carlo method for Stock prediction:-   1: Google, 2: Microsoft, 3: Apple, 4: Amazon, 5: Facebook"))
    if company == 1:
        name = "GOOG"
    elif company == 2:
        name = "MSFT"
    elif company == 3:
        name = "AAPL"
    elif company == 4:
        name = "AMZN"
    elif company == 5:
        name = "FB"
    else:
        print("Invalid Input")
    delta = 1 / 180
    starttime = dt.datetime(2016, 1, 1)
    closingDataFrame = webData.DataReader(['MSFT', 'AAPL', 'GOOG', 'AMZN', 'FB'], 'yahoo', starttime, endtime)['Adj Close']
    pctChange = closingDataFrame.pct_change()
    cleanPctChange = pctChange.dropna()
    mean = cleanPctChange.mean()[name]
    std = cleanPctChange.std()[name]
    price = webData.DataReader([name], 'yahoo', starttime, endtime)
    startprice = price["Close"].head(1)
    startprice = startprice.values[0][0]
    for i in range(100):
        priceArray = monteCarloPrediction(startprice, name, 180, mean, std, delta)
        plot.plot(priceArray)
    plot.xlabel("Days")
    plot.ylabel("Price")
    plot.show()

if __name__ == '__main__':
    print("Enter company name for stock analysis:")
    print("Microsoft")
    print("Apple")
    companies = ['MSFT', 'AAPL']
    cleanFile = []
    for comp in companies:
        getCompanyData(comp)
        transformData(comp, comp + '.xlsx')
        cleanFile.append(dataPreProcessing(comp, comp + '.xlsx'))
