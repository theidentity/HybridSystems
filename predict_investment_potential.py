import io_data
import numpy as np
import pandas as pd


def predict_cis(X, y):

    print(X.columns)
    print(y.columns)

    y_pred = y[['cust Investment Potential Score ']]
    print(y_pred)

    return y_pred

if __name__ == '__main__':
    X_test, y_test = io_data.load_orig_dataset('test')
    cis_pred = predict_cis(X_test, y_test)

    df = pd.DataFrame()
    df['index'] = np.arange(1001,5001)
    df['cis'] = cis_pred
    df.to_csv('data/prod_cis.csv',index=False)