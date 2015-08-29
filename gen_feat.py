import pandas as pd
import numpy as np
import cPickle as pickle

from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer

from sklearn_pandas import DataFrameMapper

from param import config

def extract_feature(path, train, test, type, feat_names):
    y_train = train['Hazard'].values
    train.drop("Hazard", axis=1, inplace=True)
    y_test = [1] * len(test.index)
    if type == "valid":
        y_test = test['Hazard'].values
        test.drop("Hazard", axis=1, inplace=True)

    if feat_names.count("onehot") > 0:
        feat = "onehot"
        train_s = train.copy()
        test_s = test.copy()
        train_s = np.array(train_s)
        test_s = np.array(test_s)

        for i in range(train_s.shape[1]):
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_s[:, i]) + list(test_s[:, i]))
            train_s[:, i] = lbl.transform(train_s[:, i])
            test_s[:, i] = lbl.transform(test_s[:, i])

        all_data = np.concatenate((train_s, test_s), axis=0)
        hot = preprocessing.OneHotEncoder()
        hot.fit(all_data)
        train_s = hot.transform(train_s)
        test_s = hot.transform(test_s)

        with open("%s/train.%s.feat.pkl" %(path, feat), "wb") as f:
            pickle.dump([train_s, y_train], f, -1)
        with open("%s/%s.%s.feat.pkl" %(path, type, feat), "wb") as f:
            pickle.dump([test_s, y_test], f, -1)


    if feat_names.count("label") > 0:
        feat = "label"
        train_s = train.copy()
        test_s = test.copy()
        train_s.drop("T2_V10", axis=1, inplace=True)
        train_s.drop("T2_V7", axis=1, inplace=True)
        train_s.drop("T1_V13", axis=1, inplace=True)
        train_s.drop("T1_V10", axis=1, inplace=True)

        test_s.drop("T2_V10", axis=1, inplace=True)
        test_s.drop("T2_V7", axis=1, inplace=True)
        test_s.drop("T1_V13", axis=1, inplace=True)
        test_s.drop("T1_V10", axis=1, inplace=True)

        train_s = np.array(train_s)
        test_s = np.array(test_s)

        #label encode the categorical variables
        for i in range(train_s.shape[1]):
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_s[:, i]) + list(test_s[:, i]))
            train_s[:, i] = lbl.transform(train_s[:, i])
            test_s[:, i] = lbl.transform(test_s[:, i])

        train_s = train_s.astype(float)
        test_s = test_s.astype(float)
        with open("%s/train.%s.feat.pkl" %(path, feat), "wb") as f:
            pickle.dump([train_s, y_train], f, -1)
        with open("%s/%s.%s.feat.pkl" %(path, type, feat), "wb") as f:
            pickle.dump([test_s, y_test], f, -1)

    if feat_names.count("dictvec") > 0:
        feat = "dictvec"
        train_s = train.copy()
        test_s = test.copy()
        train_s.drop("T2_V10", axis=1, inplace=True)
        train_s.drop("T2_V7", axis=1, inplace=True)
        train_s.drop("T1_V13", axis=1, inplace=True)
        train_s.drop("T1_V10", axis=1, inplace=True)

        test_s.drop("T2_V10", axis=1, inplace=True)
        test_s.drop("T2_V7", axis=1, inplace=True)
        test_s.drop("T1_V13", axis=1, inplace=True)
        test_s.drop("T1_V10", axis=1, inplace=True)

        train_s = train_s.T.to_dict().values()
        test_s = test_s.T.to_dict().values()

        vec = DictVectorizer()
        train_s = vec.fit_transform(train_s)
        test_s = vec.transform(test_s)
        with open("%s/train.%s.feat.pkl" %(path, feat), "wb") as f:
            pickle.dump([train_s, y_train], f, -1)
        with open("%s/%s.%s.feat.pkl" %(path, type, feat), "wb") as f:
            pickle.dump([test_s, y_test], f, -1)

    if feat_names.count('standard') > 0:
        feat = "standard"
        #concatanate the pandas dataframes
        temp = pd.concat([train,test])

        #inspired from http://stackoverflow.com/questions/24745879/
        binaries = ['T1_V6','T1_V17','T2_V3','T2_V11','T2_V12']
        encoders = ['T1_V4','T1_V5','T1_V7','T1_V8','T1_V9','T1_V11','T1_V12','T1_V15','T1_V16','T2_V5','T2_V13']
        scalars = ['T1_V1','T1_V2','T1_V3','T1_V10','T1_V13','T1_V14','T2_V1','T2_V2','T2_V4','T2_V6','T2_V7','T2_V8','T2_V9','T2_V10','T2_V14','T2_V15']

        mapper = DataFrameMapper(
            [(binary, preprocessing.LabelBinarizer()) for binary in binaries] +
            [(encoder, preprocessing.LabelEncoder()) for encoder in encoders] +
            [(scalar, preprocessing.StandardScaler()) for scalar in scalars]
        )

        tempMapped = mapper.fit_transform(temp)

        #split them apart  again
        train_s = tempMapped[:len(train)]
        test_s = tempMapped[len(train):]

        with open("%s/train.%s.feat.pkl" %(path, feat), "wb") as f:
            pickle.dump([train_s, y_train], f, -1)
        with open("%s/%s.%s.feat.pkl" %(path, type, feat), "wb") as f:
            pickle.dump([test_s, y_test], f, -1)


if __name__ == "__main__":
    # load data
    train = pd.read_csv(config.origin_train_path, index_col=0)
    test = pd.read_csv(config.origin_test_path, index_col=0)

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
