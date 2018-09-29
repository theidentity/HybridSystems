import io_data
import pandas as pd

from sklearn.preprocessing import StandardScaler
from helpers import convt_to_one_hot


def transform_X_clf(train_X, test_X):

    cont_cols = [3, 7, 8, 9]
    cat_cols = [1, 2, 4, 5, 6]

    scaler = StandardScaler()
    scaler.fit(train_X.iloc[:, cont_cols])
    train_X.iloc[:, cont_cols] = scaler.transform(train_X.iloc[:, cont_cols])
    test_X.iloc[:, cont_cols] = scaler.transform(test_X.iloc[:, cont_cols])

    cat_train_X = pd.get_dummies(train_X.iloc[:, cat_cols], drop_first=True)
    cat_test_X = pd.get_dummies(test_X.iloc[:, cat_cols], drop_first=True)

    train_X = train_X.drop(train_X.columns[cat_cols],axis=1)
    test_X = test_X.drop(test_X.columns[cat_cols],axis=1)

    train_X = pd.concat([train_X, cat_train_X], axis=1)
    test_X = pd.concat([test_X, cat_test_X], axis=1)
    return train_X, test_X


def transform_Y_clf(train_y, test_y):
    clf_train_y = convt_to_one_hot(train_y['status'])
    clf_test_y = convt_to_one_hot(test_y['status'])
    clf_train_y = pd.DataFrame(clf_train_y)
    clf_test_y = pd.DataFrame(clf_test_y)
    return clf_train_y, clf_test_y


def prep_clf_data():

    train_X, train_y = io_data.load_orig_dataset('train')
    test_X, test_y = io_data.load_orig_dataset('test')

    train_X, test_X = transform_X_clf(train_X, test_X)
    train_X.to_csv('data/clf_train_X.csv', index=False)
    test_X.to_csv('data/clf_test_X.csv', index=False)

    clf_train_y, clf_test_y = transform_Y_clf(train_y, test_y)
    clf_train_y.to_csv('data/clf_train_y.csv', index=False)
    clf_test_y.to_csv('data/clf_test_y.csv', index=False)

if __name__ == '__main__':
    # train_X, train_y = io_data.load_orig_dataset('train')
    # test_X, test_y = io_data.load_orig_dataset('test')
    prep_clf_data()
