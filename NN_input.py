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
    data = np.array(pd.read_csv('csvFiles/sp500_%s_close.csv' % sector, index_col = 0))
    X = np.array([data[x:x + 5,:].flatten('F') for x in range(0, 749)])
    Y = np.array([data[x,:].flatten() for x in range(5, 754)])
    return X, Y

if __name__ == '__main__':
    x, y = get_data()
    print(x.shape)
    print(y.shape)
