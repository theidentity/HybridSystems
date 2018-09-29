import numpy as np
import pandas as pd

import io_data


def calc_profit(customer_data):
    prod = customer_data[:, 1]
    cis = customer_data[:, 2]
    profit = np.zeros(shape=(len(customer_data), 1))
    profit[prod == 'A'] = 0.6
    profit[prod == 'B'] = 1
    profit = profit * np.expand_dims(cis, 1)
    return profit.flatten()


def select_cust(customer_data,n=400):
    customer_data = customer_data.values
    profit = calc_profit(customer_data)
    print(np.unique(profit))

    idx = np.argsort(profit)[-n:]
    profit = profit[idx]
    print('total_profit : ',np.sum(profit))

    return customer_data[idx,:]

if __name__ == '__main__':
    X_test, y_test = io_data.load_orig_dataset('test')

    select_cust(y_test)
