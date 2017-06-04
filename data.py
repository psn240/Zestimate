import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import Imputer

def prepareData():
    train_df = pd.read_csv("input/train_2016.csv", parse_dates=["transactiondate"])
    prop_df = pd.read_csv("input/properties_2016.csv")
    sample = pd.read_csv('input/sample_submission.csv')
    
    print('Binding to float32')

    for c, dtype in zip(prop_df.columns, prop_df.dtypes):
        if dtype == np.float64:
            prop_df[c] = prop_df[c].astype(np.float32)
    
    print('Creating training and test set ...')
    train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')
    sample['parcelid'] = sample['ParcelId']
    df_test = sample.merge(prop_df, on='parcelid', how='left')
    
    x_train = train_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
    y_train = train_df['logerror'].values
    for c in x_train.dtypes[x_train.dtypes == object].index.values:
        x_train[c] = (x_train[c] == True)


    split = 80000
    x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
    
    #Drop 60% missing columns
    x_train_temp = x_train.dropna(axis =1, how='any',thresh=48000)
    x_valid_temp = x_valid[x_train_temp.columns]
    
    imp = Imputer(missing_values=np.nan, strategy='most_frequent', axis=0)

    x_train_imputed = pd.DataFrame(imp.fit_transform(x_train_temp))
    x_train_imputed.columns = x_train_temp.columns
    x_train_imputed.index = x_train_temp.index

    x_valid_imputed = pd.DataFrame(imp.transform(x_valid_temp))
    x_valid_imputed.columns = x_valid_temp.columns
    x_valid_imputed.index = x_valid_temp.index

    x_test = df_test[x_train_imputed.columns]
    for c in x_test.dtypes[x_test.dtypes == object].index.values:
        x_test[c] = (x_test[c] == True)  

    x_test_imputed = pd.DataFrame(imp.transform(x_test))
    x_test_imputed.columns = x_test.columns
    x_test_imputed.index = x_test.index
    return x_train_imputed, y_train, x_valid_imputed, y_valid, x_test_imputed
