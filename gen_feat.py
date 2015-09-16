import pandas as pd
import numpy as np
import cPickle as pickle

from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer

from sklearn_pandas import DataFrameMapper

from param import config

def extract_feature(path, train, test, type, feat_names):
    y_train = train[config.target].values
    train.drop(config.target, axis=1, inplace=True)
    y_test = [0] * len(test.index)
    if type == "valid":
        y_test = test[config.target].values
        test.drop(config.target, axis=1, inplace=True)

    if feat_names.count("label") > 0:
        feat = "label"
        train_s = train.copy()
        test_s = test.copy()
        train_s = np.array(train_s)
        test_s = np.array(test_s)

        for i in range(train_s.shape[1]):
            lbl = preprocessing.LabelEncoder()
            lbl.fit( list(train_s[:, i]) + list(test_s[:, i]) )
            train_s[:, i] = lbl.transform(list(train_s[:, i]))
            test_s[:, i] = lbl.transform(list(test_s[:, i]))

        train_s.astype(float)
        test_s.astype(float)


        with open("%s/train.%s.feat.pkl" %(path, feat), "wb") as f:
            pickle.dump([train_s, y_train], f, -1)
        with open("%s/%s.%s.feat.pkl" %(path, type, feat), "wb") as f:
            pickle.dump([test_s, y_test], f, -1)

    if feat_names.count("onehot") > 0:
        feat = "onehot"
        train_s = train.copy()
        test_s = test.copy()

        categorical_ids = []
        for i in range(len(train_s.columns)):
            key = train_s.columns[i]
            if train_s.dtypes[key] == 'object':
                categorical_ids.append(i)
        train_s = np.array(train_s)
        test_s = np.array(test_s)

        for i in range(train_s.shape[1]):
            if categorical_ids.count(i) > 0:
                lbl = preprocessing.LabelEncoder()
                lbl.fit( list(train_s[:, i]) + list(test_s[:, i]) )
                train_s[:, i] = lbl.transform(list(train_s[:, i]))
                test_s[:, i] = lbl.transform(list(test_s[:, i]))

        hot = preprocessing.OneHotEncoder(categorical_features=categorical_ids)
        all_data = np.concatenate((train_s, test_s), axis=0)
        hot.fit(all_data)
        train_s = hot.transform(train_s)
        test_s = hot.transform(test_s)

        train_s.astype(float)
        test_s.astype(float)
        with open("%s/train.%s.feat.pkl" %(path, feat), "wb") as f:
            pickle.dump([train_s, y_train], f, -1)
        with open("%s/%s.%s.feat.pkl" %(path, type, feat), "wb") as f:
            pickle.dump([test_s, y_test], f, -1)


if __name__ == "__main__":
    # load data
    train = pd.read_csv(config.origin_train_path, index_col=0).fillna(-1)
    test = pd.read_csv(config.origin_test_path, index_col=0).fillna(-1)

    # remove columns with only one unique value, or NaN
    remove_keys = []
    with open('remove_keys.pkl', 'rb') as f:
        remove_keys = pickle.load(f)
    for key in list(train.columns):
        vals = np.unique( train[key] )
        if len(vals) <= 1:
            remove_keys.append(key)
        if len(vals) == 2:
            for v in vals:
                if v == -1:
                    remove_keys.append(key)
                    break
        if remove_keys.count(key) == 0 and len( train[key].value_counts() ) < 1:
            remove_keys.append(key)
    print len(remove_keys)
    print remove_keys

    for key in remove_keys:
        train.drop(key, axis=1, inplace=True)
        test.drop(key, axis=1, inplace=True)

    # load kfold object
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
