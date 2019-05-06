import pandas as pd
import numpy as np

def get_data(sector = 'total', period = 5):
    data = np.array(pd.read_csv('csvFiles/sp500_%s_close.csv' % sector, index_col = 0))
    X = np.array([data[x:x + 5,:].flatten('F') for x in range(0, 749)])
    Y = np.array([data[x,:].flatten() for x in range(5, 754)])
    return X, Y

if __name__ == '__main__':
    x, y = get_data()
