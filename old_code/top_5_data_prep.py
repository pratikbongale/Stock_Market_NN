import pandas as pd

def derive_features(df):
    # add new features as per 4 metrics(Daily Returns, Moving Avg, Lagged Daily Returns, 5 Minute Returns Change)

    n = 1

    for col_name in df.columns.values:
        df[col_name+'_M_RET'] = df[col_name].pct_change(1)
        df[col_name + '_5M_RET'] = df[col_name].pct_change(5)
        df[col_name + '_5M_AVG'] = df[col_name+'_M_RET'].rolling(window=5, center=False).mean()
        df[col_name + '_LAGGED'] = df[col_name+'_M_RET'].shift(1)

    drop_features = ['AAPL', 'AMZN', 'FB', 'GOOG', 'MSFT', 'SP500_LAGGED', 'SP500']
    df.drop(drop_features, axis=1, inplace=True)
    df = df[5:]
    df.to_csv(path_or_buf='data_stocks_prepared.csv', index=False)
    print(df.columns.values)

    return df

# Testing this module:

# data = pd.read_csv('data_stocks_clean.csv')
# # prepare data
# data = derive_features(data)



