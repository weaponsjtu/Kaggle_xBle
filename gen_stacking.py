###
# gen_stacking.py
# author: Weipeng Zhang
#
#
# 1. prediction feature, LogisticRegression
# 2. prediction feature + origin feature, non-linear, GBF, KNN, ET, etc
###

from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn import cross_validation
from sklearn import manifold

import xgboost as xgb

import cPickle as pickle
import numpy as np
import pandas as pd
import sys,os,time
import multiprocessing

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, pyll
from hyperopt.mongoexp import MongoTrials

from param import config

from utils import *

from param import config
from gen_ensemble import gen_subm, check_model

class StackProcess(multiprocessing.Process):
    def __init__(self, x_train, y_train, x_test, y_test, gini_cv):
        multiprocessing.Process.__init__(self)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.gini_cv = gini_cv

    def run(self):
        y_pred = xgb_train(self.x_train, self.y_train, self.x_test)
        self.gini_cv.append(Gini(self.y_test, y_pred))
        print "Process %d done! Gini is %f" %(os.getpid(),  Gini(self.y_test, y_pred))


def gen_base_model():
    # load feat, labels and pred
    feat_names = config.feat_names
    model_list = config.model_list

    # combine them, and generate whold model_list
    model_library = []
    for feat in feat_names:
        for model in model_list:
            if check_model("%s_%s"%(feat, model)):
                model_library.append("%s_%s" %(feat, model))
            for num in range(config.hyper_max_evals - 10, config.hyper_max_evals+1):
                model_name = "%s_%s@%d" %(feat, model, num)
                if check_model(model_name):
                    model_library.append(model_name)
                    break

    #model_library = add_prior_models(model_library)
    return model_library

# stacking
def model_stacking():
    # load data
    train = pd.read_csv(config.origin_train_path, index_col=0)
    test = pd.read_csv(config.origin_test_path, index_col=0)

    x_label = np.array(train['Hazard'].values)
    y_len = len(list(test.index))


    model_library = gen_base_model()
    print len(model_library)
    print model_library
    blend_train = np.zeros((len(x_label), len(model_library), config.kiter))
    blend_test = np.zeros((y_len, len(model_library)))

    # load kfold object
    with open("%s/fold.pkl" % config.data_folder, 'rb') as i_f:
        skf = pickle.load(i_f)
    i_f.close()

    for iter in range(config.kiter):
        for i in range(len(model_library)):
            for j, (validInd, trainInd) in enumerate(skf[iter]):
                path = "%s/iter%d/fold%d/%s.pred.pkl" %(config.data_folder, iter, j, model_library[i])
                with open(path, 'rb') as f:
                    y_pred = pickle.load(f)
                f.close()
                #print "Gini score is %f" % Gini(x_label[validInd], y_pred)
                blend_train[validInd, i, iter] += y_pred
            blend_train[:, i, iter] /= config.kfold


    for i in range(len(model_library)):
        path = "%s/all/%s.pred.pkl" %(config.data_folder, model_library[i])
        with open(path, 'rb') as f:
            y_pred = pickle.load(f)
        f.close()
        blend_test[:, i] = y_pred

    return blend_train, blend_test, x_label, model_library

def xgb_train(x_train, x_label, x_test):
    model = 'xgb'
    #model = 'adaboost'
    #if model.count('xgb') >0:
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.005  # [0,1]
    params["min_child_weight"] = 6
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.7
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["max_depth"] = 9
    if config.nthread > 1:
        params["nthread"] = 1

    num_rounds = 10000

    xgtrain = xgb.DMatrix(x_train, label=x_label)
    xgval = xgb.DMatrix(x_test)

    #train using early stopping and predict
    watchlist = [(xgtrain, "train")]
    #model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=120, feval=gini_metric)
    model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
    pred1 = model.predict( xgval )

    #clf = RandomForestRegressor()
    #clf = LogisticRegression()
    #clf = GradientBoostingRegressor()
    clf = AdaBoostRegressor( ExtraTreesRegressor(max_depth=9), n_estimators=200 )
    clf.fit(x_train, x_label)
    pred2 = clf.predict(x_test)

    #pred = pred1 * pred2 / (pred1 + pred2)
    #pred = 0.7 * (pred1**0.01) + 0.3 * (pred2**0.01)
    #pred = (pred1.argsort() + pred2.argsort()) / 2
    pred = 0.6 * pred1 + 0.4 * pred2

    return pred

def stacking_base(blend_train, blend_test, x_label, model_library):
    print "Blending..., 4->9"

    ## kfold cv
    kf = cross_validation.KFold(n=blend_train.shape[0], n_folds=3, shuffle=False, random_state=None)

    if config.nthread > 1:
        mp_list = []
        manager = multiprocessing.Manager()
        gini_cv = manager.list()
        for iter in range(config.kiter):
            for train_index, test_index in kf:
                x_train, x_test = blend_train[train_index, :, iter], blend_train[test_index, :, iter]
                y_train, y_test = x_label[train_index], x_label[test_index]

                mp = StackProcess(x_train, y_train, x_test, y_test, gini_cv)
                mp_list.append(mp)

        for mp in mp_list:
            mp.start()

        for mp in mp_list:
            mp.join()

        print "Average gini is %f" %(np.mean(gini_cv))
    else:
        gini_cv = []
        #for iter in range(config.kiter):
        for iter in range(1):
            for train_index, test_index in kf:
                x_train, x_test = blend_train[train_index, :, iter], blend_train[test_index, :, iter]
                y_train, y_test = x_label[train_index], x_label[test_index]
                #clf.fit(x_train, y_train)
                #y_pred = clf.predict(x_test)
                y_pred = xgb_train(x_train, y_train, x_test)
                gini_cv.append(Gini(y_test, y_pred))
        print "Average gini is %f" %(np.mean(gini_cv))


    y_sub = np.zeros((blend_test.shape[0]))
    #for iter in range(config.kiter):
    for iter in range(1):
        #clf.fit(blend_train[:, :, iter], x_label)
        #y_pred = clf.predict(blend_test)
        y_pred = xgb_train(blend_train[:, :, iter], x_label, blend_test)
        y_sub += y_pred
    y_sub = y_sub / config.kiter
    gen_subm(y_sub, 'sub/model_stack.csv')

def stacking_nonlinear(blend_train_raw, blend_test_raw, x_label, model_library):
    print "stacking nonlinear"
    # add raw feature
    with open("%s/all/train.onehot.feat.pkl" %(config.data_folder), 'rb') as f:
        [x_train, y_train] = pickle.load(f)
    with open("%s/all/test.onehot.feat.pkl" %(config.data_folder), 'rb') as f:
        [x_test, y_test] = pickle.load(f)

    if type(x_train) != np.ndarray:
        x_train = x_train.toarray()
        x_test = x_test.toarray()

    meta_feature_dim = 4 + x_train.shape[1]
    blend_train = np.zeros((len(x_label), len(model_library) + meta_feature_dim, config.kiter))
    blend_train[:, :len(model_library), :] = blend_train_raw
    blend_test = np.zeros((blend_test_raw.shape[0], len(model_library) + meta_feature_dim))
    blend_test[:, :len(model_library)] = blend_test_raw

    # KNN feature, 4 dimension
    with open('../train_feature_engineered.pkl', 'rb') as f:
        df_train = pickle.load(f)
    with open('../test_feature_engineered.pkl', 'rb') as f:
        df_test = pickle.load(f)

    knn_feats = ['meanH_N_1', 'meanH_N_2', 'meanH_N_4', 'meanH_N_8']
    for i in range(4):
        for iter in range(config.kiter):
            blend_train[:, len(model_library) + i, iter] = df_train[ knn_feats[i] ].values
        blend_test[:, len(model_library) + i] = df_test[ knn_feats[i] ].values

    for iter in range(config.kiter):
        blend_train[:, (len(model_library) + 4):, iter] = x_train
    blend_test[:, (len(model_library) + 4):] = x_test



    ## t-SNE unsupervised feature,  memory error
    #feature = []
    #for feat in config.feat_names:
    #    with open("%s/all/train.%s.feat.pkl" %(config.data_folder, feat), 'rb') as f:
    #        [x_train, y_train] = pickle.load(f)
    #    if type(x_train) != np.ndarray:
    #        x_train = x_train.toarray()
    #    #model = manifold.TSNE(n_components=3, random_state=0)
    #    #x_train = model.fit_transform(x_train)
    #    feature.append(x_train)





    print "Feature Dimension is %d" %blend_train.shape[1]
    # stacking
    stacking_base(blend_train, blend_test, x_label, model_library)



if __name__ == '__main__':
    start_time = time.time()
    print "Code start at %s" %time.ctime()
    #model_stacking()
    blend_train, blend_test, x_label, model_library = model_stacking()
    print "Feature done!!!"
    #stacking_base(blend_train, blend_test, x_label, model_library)
    stacking_nonlinear(blend_train, blend_test, x_label, model_library)

    end_time = time.time()
    print "cost time %f" %( (end_time - start_time)/1000 )
