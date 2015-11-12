import pandas as pd
import numpy as np
import cPickle as pickle

from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer

from sklearn_pandas import DataFrameMapper

from param import config
# Gather some features
def build_features(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    features.extend(['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                     'CompetitionOpenSinceYear', 'Promo', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear'])

    # add some more with a bit of preprocessing
    features.append('SchoolHoliday')
    data['SchoolHoliday'] = data['SchoolHoliday'].astype(float)
    #
    #features.append('StateHoliday')
    #data.loc[data['StateHoliday'] == 'a', 'StateHoliday'] = '1'
    #data.loc[data['StateHoliday'] == 'b', 'StateHoliday'] = '2'
    #data.loc[data['StateHoliday'] == 'c', 'StateHoliday'] = '3'
    #data['StateHoliday'] = data['StateHoliday'].astype(float)

    features.append('DayOfWeek')
    features.append('month')
    features.append('day')
    features.append('year')
    data['year'] = data.Date.apply(lambda x: x.split('-')[0])
    data['year'] = data['year'].astype(float)
    data['month'] = data.Date.apply(lambda x: x.split('-')[1])
    data['month'] = data['month'].astype(float)
    data['day'] = data.Date.apply(lambda x: x.split('-')[2])
    data['day'] = data['day'].astype(float)

    features.append('StoreType')
    data.loc[data['StoreType'] == 'a', 'StoreType'] = '1'
    data.loc[data['StoreType'] == 'b', 'StoreType'] = '2'
    data.loc[data['StoreType'] == 'c', 'StoreType'] = '3'
    data.loc[data['StoreType'] == 'd', 'StoreType'] = '4'
    data['StoreType'] = data['StoreType'].astype(float)

    features.append('Assortment')
    data.loc[data['Assortment'] == 'a', 'Assortment'] = '1'
    data.loc[data['Assortment'] == 'b', 'Assortment'] = '2'
    data.loc[data['Assortment'] == 'c', 'Assortment'] = '3'
    data['Assortment'] = data['Assortment'].astype(float)

def extract_feature(path, train, test, type, feat_names):
    y_train = train[config.target].values
    train.drop(config.target, axis=1, inplace=True)
    y_test = [0] * len(test.index)
    if type == "valid":
        y_test = test[config.target].values
        test.drop(config.target, axis=1, inplace=True)

    if feat_names.count("stand") > 0:
        feat = "stand"
        train_s = train.copy()
        test_s = test.copy()

        with open("%s/train.%s.feat.pkl" %(path, feat), "wb") as f:
            pickle.dump([train_s, y_train], f, -1)
        with open("%s/%s.%s.feat.pkl" %(path, type, feat), "wb") as f:
            pickle.dump([test_s, y_test], f, -1)


if __name__ == "__main__":
    # load data
    train = pd.read_csv(config.origin_train_path, index_col=0)
    test = pd.read_csv(config.origin_test_path, index_col=0)

    features = []
    build_features(features, train)
    build_features([], test)
    features.append('Sales')
    train = train[features]
    test = test[features]

    ## load kfold object
    with open("%s/fold.pkl" % config.data_folder, 'rb') as i_f:
        skf = pickle.load(i_f)

    # extract features
    feat_names = config.feat_names

    # for cross validation
    print "Extract feature for cross validation"
    for iter in range(config.kiter):
        for fold, (trainInd, validInd) in enumerate(skf[iter]):
            path = "%s/iter%d/fold%d" %(config.data_folder, iter, fold)
            sub_train = train.iloc[trainInd].copy()
            sub_val = train.iloc[validInd].copy()
            # extract feature
            extract_feature(path, sub_train, sub_val, "valid", feat_names)
    print "Done"

    # for train/test
    print "Extract feature for train/test"
    path = "%s/all" % config.data_folder
    extract_feature(path, train, test, "test", feat_names)
    print "Done"
