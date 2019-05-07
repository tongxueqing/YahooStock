import pandas as pd
import numpy as np

def get_data(sector = 'total', period = 5):
    '''
    Read from a certain close.csv. If name not given, default is total_close.csv
    (all the files are in `/csvFiles`)
    Return a feature matrix X (749 * 995) and a label matrix Y (749 * 199)
    Every row of X is 5 days data of all the companies (5 * 199 = 995), there are 749 rows (749 days in total data)
    Every row of Y is 6th day data of all the companies, there are 749 rows
    '''
    data = pd.read_csv('csvFiles/sp500_%s_close.csv' % sector, index_col = 0) # get data
    cols = data.columns
    filt = ~np.isnan(data).any(axis = 0)
    cols = np.array(cols)[filt]
    data = np.array(data[cols])
    X = np.array([data[x:x + period,:].flatten() for x in range(0, data.shape[0] - 5)]) # 5 day as features, total 749
    Y = np.array([data[x,:].flatten() for x in range(period, data.shape[0])]) # for each 5 day, use the 6th day as the label (result)
    data = pd.read_csv('csvFilesTest/%s_test.csv' % sector, index_col = 0) # get data
    data = np.array(data[cols])
    X_test = np.array([data[x:x + period,:].flatten() for x in range(0, data.shape[0] - 5)])
    Y_test = np.array([data[x,:].flatten() for x in range(period, data.shape[0])])
    return X, Y, X.shape[0], X.shape[1], Y.shape[1], X_test, Y_test

if __name__ == '__main__':
    x, y, x0, x1, y1, xt, yt = get_data()
    print(x.shape)
    print(y.shape)
    print(xt.shape)
    print(yt.shape)
    print(x)
    print(xt)
