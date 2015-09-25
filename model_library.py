import pandas as pd
import numpy as np
import cPickle as pickle
import os

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn import neighbors

import xgboost as xgb

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL, pyll
from hyperopt.mongoexp import MongoTrials

from utils import *

from param import config

import time
import sys
import multiprocessing

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adadelta, Adagrad


def deep_model():
    model = Sequential()
    model.add(Dense(33, 20, init='uniform', activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(20, 10, init='uniform', activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(10, 1, init='uniform', activation='linear'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    #model.fit(X_train, y_train, nb_epoch=20, batch_size=16)
    #score = model.evaluate(X_test, y_test, batch_size=16)
    return model


def train_model(path, x_train, y_train, x_test, y_test, feat, param_best_dic):

    model_list = []
    for model in config.model_list:
        if param_best_dic.has_key("%s_%s"%(feat, model)):
            model_list.append(model)
    ######
    # approach different model
    # Deep Learning Model
    if model_list.count('dnn') > 0:
        model_type = 'dnn'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = param_best_dic["%s_%s" %(feat, model_type)]
            model = deep_model()
            model.fit(x_train, y_train, nb_epoch=2, batch_size=16)
            pred_val = model.predict( x_test, batch_size=16 )
            pred_val = pred_val.reshape( pred_val.shape[0] )
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"

    # Nearest Neighbors
    if model_list.count('knn') > 0:
        model_type = 'knn'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = param_best_dic["%s_%s" %(feat, model_type)]
            n_neighbors = model_param['n_neighbors']
            weights = model_param['weights']
            model = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
            model.fit( x_train, y_train )
            pred_val = model.predict( x_test )
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"

    # linear regression
    if model_list.count('linear') > 0:
        model_type = 'linear'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = param_best_dic["%s_%s" %(feat, model_type)]
            model = LinearRegression()
            model.fit( x_train, y_train )
            pred_val = model.predict( x_test )
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"

    # logistic regression
    if model_list.count('logistic') > 0:
        model_type = 'logistic'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = param_best_dic["%s_%s" %(feat, model_type)]
            model = LogisticRegression()
            model.fit( x_train, y_train )
            pred_val = model.predict( x_test )
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"

    # SVM regression
    if model_list.count('svr') > 0:
        model_type = 'svr'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = param_best_dic["%s_%s" %(feat, model_type)]
            model = SVR(C=model_param['C'], epsilon=model_param['epsilon'])
            model.fit( x_train, y_train )
            pred_val = model.predict( x_test )
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"


    # random forest regression
    if model_list.count('rf') > 0:
        model_type = 'rf'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = param_best_dic["%s_%s" %(feat, model_type)]
            model = RandomForestRegressor(n_estimators=model_param['n_estimators'])
            model.fit( x_train, y_train )
            pred_val = model.predict( x_test )
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print 'Done!'

    # extra tree regression
    if model_list.count('extratree') > 0:
        model_type = 'extratree'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = param_best_dic["%s_%s" %(feat, model_type)]
            model = ExtraTreesRegressor(n_estimators=model_param['n_estimators'])
            model.fit( x_train, y_train )
            pred_val = model.predict( x_test )
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"

    # GBRT regression
    if model_list.count('gbf') > 0:
        model_type = 'gbf'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = param_best_dic["%s_%s" %(feat, model_type)]
            model = GradientBoostingRegressor(n_estimators=model_param['n_estimators'])
            if type(x_train) != np.ndarray:
                model.fit( x_train.toarray(), y_train )
                pred_val = model.predict( x_test.toarray() )
            else:
                model.fit( x_train, y_train )
                pred_val = model.predict( x_test )

            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"

    # xgboost tree
    if model_list.count('xgb_tree') > 0:
        model_type = 'xgb_tree'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = param_best_dic["%s_%s" %(feat, model_type)]
            params = model_param
            num_rounds = model_param['num_rounds']
            #create a train and validation dmatrices
            xgtrain = xgb.DMatrix(x_train, label=y_train)
            xgval = xgb.DMatrix(x_test, label=y_test)

            #train using early stopping and predict
            watchlist = [(xgtrain, "train"),(xgval, "val")]
            #model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=100, feval=gini_metric)
            model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=model_param['early_stopping_rounds'])
            pred_val = model.predict( xgval, ntree_limit=model.best_iteration )
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"

    # xgboost rank pairwise
    if model_list.count('xgb_rank') > 0:
        model_type = 'xgb_rank'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = param_best_dic["%s_%s" %(feat, model_type)]
            params = model_param
            num_rounds = model_param['num_rounds']
            #create a train and validation dmatrices
            xgtrain = xgb.DMatrix(x_train, label=y_train)
            xgval = xgb.DMatrix(x_test, label=y_test)

            #train using early stopping and predict
            watchlist = [(xgtrain, "train"),(xgval, "val")]
            #model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=100, feval=gini_metric)
            model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=model_param['early_stopping_rounds'])
            pred_val = model.predict( xgval, ntree_limit=model.best_iteration )
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"

    # xgboost linear
    if model_list.count('xgb_linear') > 0:
        model_type = 'xgb_linear'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
            print "%s training..." % model_type
            model_param = param_best_dic["%s_%s" %(feat, model_type)]
            params = model_param
            num_rounds = model_param['num_rounds']
            #create a train and validation dmatrices
            xgtrain = xgb.DMatrix(x_train, label=y_train)
            xgval = xgb.DMatrix(x_test, label=y_test)

            #train using early stopping and predict
            watchlist = [(xgtrain, "train"),(xgval, "val")]
            #model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=100, feval=gini_metric)
            model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=model_param['early_stopping_rounds'])
            pred_val = model.predict( xgval )
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"

    if model_list.count('xgb_art') > 0:
        model_type = 'xgb_art'
        pred_file = "%s/%s_%s.pred.pkl" %(path, feat, model_type)
        if config.update_model.count(model_type) > 0 or os.path.exists(pred_file) is False:
            print "%s trainning..." % model_type
            model_param = param_best_dic["%s_%s" %(feat, model_type)]
            params = model_param
            num_rounds = model_param['num_rounds']
            offset = int(model_param['valid_size'] * y_train.shape[0]) + 1 # just for 4000
            #if type(x_train) != np.ndarray:
            #    x_train = x_train.toarray()
            #    x_test = x_test.toarray()
            xgtrain = xgb.DMatrix(x_train[offset:, :], label=y_train[offset:])
            xgval = xgb.DMatrix(x_train[:offset, :], label=y_train[:offset])
            watchlist = [(xgtrain, "train"), (xgval, "val")]
            model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds = model_param['early_stopping_rounds'])

            xgtest = xgb.DMatrix(x_test)
            pred_val1 = model.predict(xgtest, ntree_limit=model.best_iteration)

            # reverse train, and log label
            x_train_tmp = x_train[::-1, :]
            y_train_tmp = np.log(y_train[::-1])
            xgtrain = xgb.DMatrix(x_train_tmp[offset:, :], label=y_train_tmp[offset:])
            xgval = xgb.DMatrix(x_train_tmp[:offset, :], label=y_train_tmp[:offset])
            watchlist = [(xgtrain, "train"), (xgval, "val")]
            model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds = model_param['early_stopping_rounds'])
            pred_val2 = model.predict(xgtest, ntree_limit=model.best_iteration)

            pred_val = pred_val1*1.5 + pred_val2*8.5
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            print "Done!"
    ######

def one_model():
    # load feat names
    feat_names = config.feat_names

    # load best params for each model (feat, model)
    with open("%s/model_best_params" %config.data_folder) as f:
        param_best_dic = pickle.load(f)

    # supply the extra parameter from config.param_spaces
    for feat in config.feat_names:
        for model in config.model_list:
            if param_best_dic.has_key("%s_%s"%(feat, model)):
                param_space = config.param_spaces[model]
                for key in param_space.keys():
                    if param_best_dic["%s_%s"%(feat, model)].has_key(key) is False:
                        param_best_dic["%s_%s"%(feat, model)][key] = param_space[key]
    print param_best_dic

    # load feat, cross validation
    for iter in range(config.kiter):
        for fold in range(config.kfold):
            for feat in feat_names:
                print "Gen pred for (iter%d, fold%d, %s) cross validation" %(iter, fold, feat)
                with open("%s/iter%d/fold%d/train.%s.feat.pkl" %(config.data_folder, iter, fold, feat), 'rb') as f:
                    [x_train, y_train] = pickle.load(f)
                with open("%s/iter%d/fold%d/valid.%s.feat.pkl" %(config.data_folder, iter, fold, feat), 'rb') as f:
                    [x_val, y_val] = pickle.load(f)
                path = "%s/iter%d/fold%d" %(config.data_folder, iter, fold)
                train_model(path, x_train, y_train, x_val, y_val, feat, param_best_dic)

    # load feat, train/test
    for feat in feat_names:
        print "Gen pred for (%s) all test data" %(feat)
        with open("%s/all/train.%s.feat.pkl" %(config.data_folder, feat), 'rb') as f:
            [x_train, y_train] = pickle.load(f)
        with open("%s/all/test.%s.feat.pkl" %(config.data_folder, feat), 'rb') as f:
            [x_test, y_test] = pickle.load(f)
        path = "%s/all" %(config.data_folder)
        train_model(path, x_train, y_train, x_test, y_test, feat, param_best_dic)

def make_obj(model_type, feat):   # use closure to support MongoTrials
    import pandas as pd
    import numpy as np
    import cPickle as pickle
    import os

    from sklearn import preprocessing
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.svm import SVR, SVC
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
    from sklearn import neighbors

    import xgboost as xgb

    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL, pyll
    from hyperopt.mongoexp import MongoTrials


    import time
    import sys
    import multiprocessing

    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.optimizers import SGD, Adadelta, Adagrad
    class ParamConfig:
        def __init__(self, data_folder):
            # target variable to predict
            self.target = 'target'
            self.tid = 'ID'

            # (3,3), (2, 5), (5, 2)
            self.kfold = 3  # cross validation, k-fold
            self.kiter = 1  # shuffle dataset, and repeat CV

            self.DEBUG = True
            self.use_mongo = True
            self.mongo_server = 'mongo://172.16.13.7:27017/'
            self.hyper_max_evals = 200
            self.ensemble_max_evals = 200

            self.nthread = 1
            self.max_core = multiprocessing.cpu_count()


            if self.DEBUG:
                self.hyper_max_evals = 5
                #self.ensemble_max_evals = 5

            self.origin_train_path = "../data/train.csv"
            self.origin_test_path = "../data/test.csv"

            #self.feat_names = ['stand', 'label', 'dictvec', 'onehot']
            self.feat_names = ['label']

            self.data_folder = data_folder
            if not os.path.exists(self.data_folder):
                os.makedirs(self.data_folder)

            # create folder for train/test
            if not os.path.exists("%s/all"% self.data_folder):
                os.makedirs("%s/all"% self.data_folder)

            # create folder for cross validation, each iter and fold
            for i in range(self.kiter):
                for f in range(self.kfold):
                    path = "%s/iter%d/fold%d" %(self.data_folder, i, f)
                    if not os.path.exists(path):
                        os.makedirs(path)

            #self.model_list = ['xgb_fix', 'logistic', 'knn', 'ridge', 'lasso', 'xgb_rank', 'xgb_linear', 'xgb_tree', 'xgb_art', 'xgb_binary', 'xgb_log', 'xgb_auc', 'rf', 'gbf']
            self.model_list = ['knn', 'rf']
            #self.model_list = ['xgb_auc', 'xgb_log']
            #self.model_list = ['gbfC', 'xgb_binary', 'xgb_auc', 'xgb_log', 'xgb_fix', 'xgb_tree_auc', 'xgb_tree_log', 'xgb_linear_fix']
            #self.model_list = ['xgb_fix', 'xgb_linear_fix']

            self.update_model = ['']
            self.model_type = ''
            self.param_spaces = {
                'logistic': {
                    'C': hp.loguniform('C', np.log(0.001), np.log(10)),
                },
                'knn': {
                    'n_neighbors': pyll.scope.int(hp.quniform('n_neighbors', 2, 1024, 2)),
                    #'weights': hp.choice('weights', ['uniform', 'distance']),
                    'weights': 'distance',
                },
                'knnC': {
                    'n_neighbors': pyll.scope.int(hp.quniform('n_neighbors', 2, 1024, 2)),
                    #'weights': hp.choice('weights', ['uniform', 'distance']),
                    'weights': 'distance',
                },
                'ridge': {
                    'alpha': hp.loguniform('alpha', np.log(0.01), np.log(20)),
                },
                'lasso': {
                    'alpha': hp.loguniform('alpha', np.log(0.00001), np.log(0.1)),
                },
                'rf': {
                    'n_estimators': pyll.scope.int(hp.quniform('n_estimators', 100, 500, 10)),
                    'max_features': hp.quniform('max_features', 0.05, 1.0, 0.05),
                },
                'rfC': {
                    'n_estimators': pyll.scope.int(hp.quniform('n_estimators', 100, 500, 100)),
                    'max_features': hp.quniform('max_features', 0.05, 1.0, 0.05),
                },
                'extratree': {
                    'n_estimators': pyll.scope.int(hp.quniform('n_estimators', 100, 500, 10)),
                    'max_features': hp.quniform('max_features', 0.05, 1.0, 0.05),
                },
                'extratreeC': {
                    'n_estimators': pyll.scope.int(hp.quniform('n_estimators', 100, 500, 10)),
                    'max_features': hp.quniform('max_features', 0.05, 1.0, 0.05),
                },
                'gbf': {
                    'n_estimators': pyll.scope.int(hp.quniform('n_estimators', 100, 500, 10)),
                    'max_features': hp.quniform('max_features', 0.05, 1.0, 0.05),
                    'learning_rate': hp.quniform('learning_rate', 0.01, 0.5, 0.01),
                    'max_depth': hp.quniform('max_depth', 1, 15, 1),
                    'subsample': hp.quniform('subsample', 0.5, 1.0, 0.1),
                },
                'gbfC': {
                    'n_estimators': pyll.scope.int(hp.quniform('n_estimators', 10, 100, 10)),
                    #'max_features': hp.quniform('max_features', 0.05, 1.0, 0.05),
                    #'learning_rate': hp.quniform('learning_rate', 0.01, 0.5, 0.01),
                    'max_depth': hp.quniform('max_depth', 1, 15, 1),
                    'subsample': hp.quniform('subsample', 0.5, 1.0, 0.1),
                },
                'svr': {
                    'C': hp.quniform('C', 0.1, 10, 0.1),
                    'epsilon': hp.loguniform('epsilon', np.log(0.001), np.log(0.1)),
                },
                'svc': {
                    'C': hp.quniform('C', 0.1, 10, 0.1),
                    'epsilon': hp.loguniform('epsilon', np.log(0.001), np.log(0.1)),
                },
                'xgb_binary': {
                    'booster': 'gblinear',
                    'objective': 'binary:logistic',
                    'eta' : hp.quniform('eta', 0.01, 1, 0.01),
                    'lambda' : hp.quniform('lambda', 0, 5, 0.05),
                    'alpha' : hp.quniform('alpha', 0, 0.5, 0.005),
                    'lambda_bias' : hp.quniform('lambda_bias', 0, 3, 0.1),
                    'num_rounds' : 10000,
                    'silent' : 1,
                    'verbose': 0,
                    'early_stopping_rounds': 120,
                    'nthread': 1,
                },
                'xgb_log': {
                    'booster': 'gblinear',
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'eta' : hp.quniform('eta', 0.01, 1, 0.01),
                    'lambda' : hp.quniform('lambda', 0, 5, 0.05),
                    'alpha' : hp.quniform('alpha', 0, 0.5, 0.005),
                    'lambda_bias' : hp.quniform('lambda_bias', 0, 3, 0.1),
                    'num_rounds' : 10000,
                    'silent' : 1,
                    'verbose': 0,
                    'early_stopping_rounds': 120,
                    'nthread': 1,
                },
                'xgb_auc': {
                    'booster': 'gblinear',
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'eta' : hp.quniform('eta', 0.01, 1, 0.01),
                    'lambda' : hp.quniform('lambda', 0, 5, 0.05),
                    'alpha' : hp.quniform('alpha', 0, 0.5, 0.005),
                    'lambda_bias' : hp.quniform('lambda_bias', 0, 3, 0.1),
                    'num_rounds' : 10000,
                    'silent' : 1,
                    'verbose': 0,
                    'early_stopping_rounds': 120,
                    'nthread': 1,
                },
                'xgb_multi': {
                    'booster': 'gblinear',
                    'objective': 'multi:softmax',
                    'eta' : hp.quniform('eta', 0.01, 1, 0.01),
                    'lambda' : hp.quniform('lambda', 0, 5, 0.05),
                    'alpha' : hp.quniform('alpha', 0, 0.5, 0.005),
                    'lambda_bias' : hp.quniform('lambda_bias', 0, 3, 0.1),
                    'num_rounds' : 10000,
                    'num_class': 69,
                    'silent' : 1,
                    'verbose': 0,
                    'early_stopping_rounds': 120,
                    'nthread': 1,
                },
                'xgb_rank': {
                    'objective': 'rank:pairwise',
                    'eta' : hp.quniform('eta', 0.01, 1, 0.01),
                    'lambda' : hp.quniform('lambda', 0, 5, 0.05),
                    'alpha' : hp.quniform('alpha', 0, 0.5, 0.005),
                    'lambda_bias' : hp.quniform('lambda_bias', 0, 3, 0.1),
                    'num_rounds' : 10000,
                    'silent' : 1,
                    'verbose': 0,
                    'early_stopping_rounds': 120,
                    'nthread': 1,
                },
                'xgb_linear': {
                    'booster': 'gblinear',
                    'objective': 'reg:linear',
                    'eta' : hp.quniform('eta', 0.01, 1, 0.01),
                    'lambda' : hp.quniform('lambda', 0, 5, 0.05),
                    'alpha' : hp.quniform('alpha', 0, 0.5, 0.005),
                    'lambda_bias' : hp.quniform('lambda_bias', 0, 3, 0.1),
                    'num_rounds' : 10000,
                    'silent' : 1,
                    'verbose': 0,
                    'early_stopping_rounds': 120,
                    'nthread': 1,
                },
                'xgb_tree_auc': {
                    'objective': 'binary:logistic',
                    'eta': 0.01, #hp.quniform('eta', 0.01, 1, 0.01),
                    #'gamma': hp.quniform('gamma', 0, 2, 0.1),
                    'min_child_weight': pyll.scope.int( hp.quniform('min_child_weight', 5, 7, 1) ),
                    'subsample': hp.quniform('subsample', 0.6, 0.8, 0.1),
                    'eval_metric': 'auc',
                    'silent': 1,
                    'verbose': 0,
                    'max_depth': pyll.scope.int(hp.quniform('max_depth', 7, 9, 1)),
                    'colsample_bytree': 0.8, #hp.quniform('colsample_bytree', 0.1, 1, 0.1),
                    'num_rounds': 1000,
                    'early_stopping_rounds': 120,
                    'nthread': 1,
                },
                'xgb_tree_log': {
                    'objective': 'binary:logistic',
                    'eta': 0.01, #hp.quniform('eta', 0.01, 1, 0.01),
                    #'gamma': hp.quniform('gamma', 0, 2, 0.1),
                    'min_child_weight': pyll.scope.int( hp.quniform('min_child_weight', 5, 7, 1) ),
                    'subsample': hp.quniform('subsample', 0.6, 0.8, 0.1),
                    'eval_metric': 'logloss',
                    'silent': 1,
                    'verbose': 0,
                    'max_depth': pyll.scope.int(hp.quniform('max_depth', 7, 9, 1)),
                    'colsample_bytree': 0.8, #hp.quniform('colsample_bytree', 0.1, 1, 0.1),
                    'num_rounds': 1000,
                    'early_stopping_rounds': 120,
                    'nthread': 1,
                },
                'xgb_art': {
                    'objective': 'reg:linear',
                    'eta': hp.quniform('eta', 0.01, 1, 0.001),
                    'gamma': hp.quniform('gamma', 0, 2, 0.1),
                    'min_child_weight': pyll.scope.int( hp.quniform('min_child_weight', 0, 10, 1) ),
                    'subsample': hp.quniform('subsample', 0.5, 0.9, 0.05),
                    'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.1),
                    'scale_pos_weight': 1,
                    'silent': 1,
                    'verbose': 0,
                    'max_depth': pyll.scope.int(hp.quniform('max_depth', 1, 10, 1)),
                    'num_rounds': 10000,
                    'valid_size': 0.07843291,
                    'early_stopping_rounds': 120,
                    'nthread': 1,
                },
                'xgb_fix': {
                    'objective': 'binary:logistic',
                    'eta': 0.005,
                    'min_child_weight': 6,
                    'subsample':0.7, #hp.quniform('subsample', 0.5, 1.0, 0.01),
                    'eval_metric': 'logloss',
                    'colsample_bytree': 0.8,
                    'scale_pos_weight': 1,
                    'silent': 1,
                    'verbose': 0,
                    'max_depth': 8, #pyll.scope.int(hp.quniform('max_depth', 1, 10, 1)),
                    'num_rounds': 1000,
                    'early_stopping_rounds': 120,
                    'nthread': 1,
                },
                'xgb_linear_fix': {
                    'objective': 'reg:linear',
                    'eta': 0.005,
                    'min_child_weight': 6,
                    'subsample':0.7, #hp.quniform('subsample', 0.5, 1.0, 0.01),
                    'eval_metric': 'logloss',
                    'colsample_bytree': 0.8,
                    'scale_pos_weight': 1,
                    'silent': 1,
                    'verbose': 0,
                    'max_depth': 8, #pyll.scope.int(hp.quniform('max_depth', 1, 10, 1)),
                    'num_rounds': 1000,
                    'early_stopping_rounds': 120,
                    'nthread': 1,
                },
            }
    config = ParamConfig("feat")

    from sklearn.metrics import roc_auc_score
    def ml_score(y_true, y_pred):
        return roc_auc_score(y_true, y_pred)

    def get_trials_counter(model_type, feat):
        trials_counter = 1
        path = "%s/pid/%s_%s.pid.pkl" %(config.data_folder, feat, model_type)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                trials_counter = pickle.load(f)
        with open(path, 'wb') as f:
            pickle.dump(trials_counter + 1, f, -1)
        return trials_counter


    def hyperopt_obj(model_param, model_type, feat, trials_counter):
        ######
        gini_cv = np.zeros((config.kiter, config.kfold), dtype=float)

        if model_type.count('xgb') > 0:  # support xgb multi-thread
            model_param['nthread'] = config.max_core

        if config.nthread == 1 or model_type.count('xgb') > 0: # single process,  support xgboost multi
            for iter in range(config.kiter):
                for fold in range(config.kfold):
                    # load data
                    path = "%s/iter%d/fold%d" %(config.data_folder, iter, fold)
                    with open("%s/train.%s.feat.pkl" %(path, feat), 'rb') as f:
                        [x_train, y_train] = pickle.load(f)
                    with open("%s/valid.%s.feat.pkl" %(path, feat), 'rb') as f:
                        [x_test, y_test] = pickle.load(f)

                    #if model_type.count('xgb') > 0:  # rescale the positive examples
                    #    ratio = float( np.sum(y_train==0) ) / np.sum(y_train==1)
                    #    model_param['scale_pos_weight'] = ratio

                    pred_val = hyperopt_library(model_type, model_param, x_train, y_train, x_test, y_test)
                    # save the pred for cross validation
                    pred_file = "%s/%s_%s@%d.pred.pkl" %(path, feat, model_type, trials_counter)
                    with open(pred_file, 'wb') as f:
                        pickle.dump(pred_val, f, -1)

                    if model_type == 'logistic':
                        y_test = y_test / np.linalg.norm(y_test)
                    gini_cv[iter, fold] = ml_score(y_test, pred_val)
        else: # multiprocess
            manager = multiprocessing.Manager()
            gini_cv = manager.list()
            lock = multiprocessing.Lock()
            mp_list = []
            for iter in range(config.kiter):
                for fold in range(config.kfold):
                    mp = ModelProcess(lock, iter, fold, feat, model_type, model_param, gini_cv)
                    mp_list.append(mp)

            for mp in mp_list:
                mp.start()

            for mp in mp_list:
                mp.join()


        gini_cv_mean = np.mean(gini_cv)
        gini_cv_std = np.std(gini_cv)
        print "Mean %f, Std %f" % (gini_cv_mean, gini_cv_std)

        # save the pred for train/test
        # load data
        path = "%s/all" %(config.data_folder)
        with open("%s/train.%s.feat.pkl" %(path, feat), 'rb') as f:
            [x_train, y_train] = pickle.load(f)
        f.close()
        with open("%s/test.%s.feat.pkl" %(path, feat), 'rb') as f:
            [x_test, y_test] = pickle.load(f)
        f.close()
        pred_val = hyperopt_library(model_type, model_param, x_train, y_train, x_test, y_test)
        # save the pred for train/test
        pred_file = "%s/%s_%s@%d.pred.pkl" %(path, feat, model_type, trials_counter)
        with open(pred_file, 'wb') as f:
            pickle.dump(pred_val, f, -1)
        f.close()

        return gini_cv_mean, gini_cv_std
        ######



    def hyperopt_library(model_type, model_param, x_train, y_train, x_test, y_test):
        try:
            # training
            if model_type.count('dnn') > 0:
                print "%s training..." % model_type
                model = deep_model()
                model.fit(x_train, y_train, nb_epoch=2, batch_size=16)
                pred_val = model.predict( x_test, batch_size=16 )
                pred_val = pred_val.reshape( pred_val.shape[0] )

            # Nearest Neighbors, regression
            if model_type.count('knn') > 0:
                print "%s training..." % model_type
                n_neighbors = model_param['n_neighbors']
                weights = model_param['weights']
                model = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
                model.fit( x_train, y_train )
                pred_val = model.predict( x_test )

            # Nearest Neighbors, classifier
            if model_type.count('knnC') > 0:
                print "%s training..." % model_type
                n_neighbors = model_param['n_neighbors']
                weights = model_param['weights']
                model = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
                model.fit( x_train, y_train )
                pred_val = model.predict( x_test )

            # linear regression
            if model_type.count('linear') > 0:
                print "%s training..." % model_type
                model = LinearRegression()
                model.fit( x_train, y_train )
                pred_val = model.predict( x_test )

            # logistic regression
            if model_type.count('logistic') > 0:
                print "%s training..." % model_type
                model = LogisticRegression()
                y_train = y_train / np.linalg.norm(y_train)
                model.fit( x_train, y_train )
                pred_val = model.predict( x_test )

            # kernal ridge regression
            if model_type.count('ridge') > 0:
                print "%s training..." % model_type
                model = Ridge(alpha=model_param['alpha'])
                model.fit( x_train, y_train )
                pred_val = model.predict( x_test )

            # lasso regression
            if model_type.count('lasso') > 0:
                print "%s training..." % model_type
                model = Ridge(alpha=model_param['alpha'])
                model.fit( x_train, y_train )
                pred_val = model.predict( x_test )


            # SVM regression
            if model_type.count('svr') > 0:
                print "%s training..." % model_type
                model = SVR(C=model_param['C'], epsilon=model_param['epsilon'])
                model.fit( x_train, y_train )
                pred_val = model.predict( x_test )

            # SVM classification
            if model_type.count('svc') > 0:
                print "%s training..." % model_type
                model = SVC(C=model_param['C'], epsilon=model_param['epsilon'])
                model.fit( x_train, y_train )
                pred_val = model.predict( x_test )

            # random forest regression
            if model_type.count('rf') > 0:
                print "%s training..." % model_type
                model = RandomForestRegressor(n_estimators=model_param['n_estimators'])
                model.fit( x_train, y_train )
                pred_val = model.predict( x_test )

            # random forest classification
            if model_type.count('rfC') > 0:
                print "%s training..." % model_type
                model = RandomForestClassifier(n_estimators=model_param['n_estimators'])
                model.fit( x_train, y_train )
                pred_val = model.predict( x_test )

            # extra tree regression
            if model_type.count('extratree') > 0:
                print "%s training..." % model_type
                model = ExtraTreesRegressor(n_estimators=model_param['n_estimators'])
                model.fit( x_train, y_train )
                pred_val = model.predict( x_test )

            # extra tree classification
            if model_type.count('extratreeC') > 0:
                print "%s training..." % model_type
                model = ExtraTreesClassifier(n_estimators=model_param['n_estimators'])
                model.fit( x_train, y_train )
                pred_val = model.predict( x_test )

            # GBRT regression
            if model_type.count('gbf') > 0:
                print "%s training..." % model_type
                model = GradientBoostingRegressor(n_estimators=model_param['n_estimators'])
                model.fit( x_train, y_train )
                pred_val = model.predict( x_test )

            # GBRT classification
            if model_type.count('gbfC') > 0:
                print "%s training..." % model_type
                model = GradientBoostingClassifier(n_estimators=model_param['n_estimators'], subsample=model_param['subsample'], max_depth=model_param['max_depth'])
                model.fit( x_train, y_train )
                pred_val = model.predict( x_test )

            # xgboost
            if model_type.count('xgb_binary') > 0 or model_type.count('xgb_log') > 0 or model_type.count('xgb_auc') > 0:
                print "%s training..." % model_type
                params = model_param
                num_rounds = model_param['num_rounds']
                #create a train and validation dmatrices
                xgtrain = xgb.DMatrix(x_train, label=y_train)
                xgval = xgb.DMatrix(x_test)

                #train using early stopping and predict
                watchlist = [(xgtrain, "train")]
                model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
                pred_val = model.predict( xgval )

            if model_type.count('xgb_rank') > 0:
                print "%s training..." % model_type
                params = model_param
                num_rounds = model_param['num_rounds']
                #create a train and validation dmatrices
                xgtrain = xgb.DMatrix(x_train, label=y_train)
                xgval = xgb.DMatrix(x_test)

                #train using early stopping and predict
                watchlist = [(xgtrain, "train")]
                #model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=100, feval=gini_metric)
                model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
                pred_val = model.predict( xgval, ntree_limit=model.best_iteration )
                #pred_val = [0]

            if model_type.count('xgb_linear') > 0:
                print "%s training..." % model_type
                params = model_param
                num_rounds = model_param['num_rounds']
                #create a train and validation dmatrices
                xgtrain = xgb.DMatrix(x_train, label=y_train)
                xgval = xgb.DMatrix(x_test)

                #train using early stopping and predict
                watchlist = [(xgtrain, "train")]
                #model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=100, feval=gini_metric)
                model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
                pred_val = model.predict( xgval )

            if model_type.count('xgb_multi') > 0:
                print "%s training..." % model_type
                params = model_param
                num_rounds = model_param['num_rounds']
                xgtrain = xgb.DMatrix(x_train, label=(y_train - 1))
                xgval = xgb.DMatrix(x_test)

                watchlist = [(xgtrain, "train")]
                model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
                pred_val = model.predict( xgval )

            if model_type.count('xgb_tree_auc') or model_type.count('xgb_tree_log') > 0 or model_type.count('xgb_fix') > 0:
                print "%s training..." % model_type
                params = model_param
                num_rounds = model_param['num_rounds']
                xgtrain = xgb.DMatrix(x_train, label=y_train)
                xgval = xgb.DMatrix(x_test)

                watchlist = [(xgtrain, "train")]
                model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=model_param['early_stopping_rounds'])
                pred_val = model.predict( xgval, ntree_limit=model.best_iteration )

            if model_type.count('xgb_art') > 0:
                print "%s trainning..." % model_type
                params = model_param
                num_rounds = model_param['num_rounds']
                #offset = int(model_param['valid_size'] * y_train.shape[0]) + 1
                offset = int(model_param['valid_size'] * y_train.shape[0]) + 1
                if type(x_train) != np.ndarray:
                    x_train = x_train.toarray()
                    x_test = x_test.toarray()
                xgtrain = xgb.DMatrix(x_train[offset:, :], label=y_train[offset:])
                xgval = xgb.DMatrix(x_train[:offset, :], label=y_train[:offset])
                watchlist = [(xgtrain, "train"), (xgval, "val")]
                model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds = model_param['early_stopping_rounds'])

                xgtest = xgb.DMatrix(x_test)
                pred_val1 = model.predict(xgtest, ntree_limit=model.best_iteration)

                # reverse train, and log label
                x_train_tmp = x_train[::-1, :]
                y_train_tmp = np.log(y_train[::-1])
                xgtrain = xgb.DMatrix(x_train_tmp[offset:, :], label=y_train_tmp[offset:])
                xgval = xgb.DMatrix(x_train_tmp[:offset, :], label=y_train_tmp[:offset])
                watchlist = [(xgtrain, "train"), (xgval, "val")]
                model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds = model_param['early_stopping_rounds'])

                pred_val2 = model.predict(xgtest, ntree_limit=model.best_iteration)
                pred_val = pred_val1*1.5 + pred_val2*8.5
        except Exception as err:
            pred_val = [0] * len(y_test)

        return pred_val


    class ModelProcess(multiprocessing.Process):
        def __init__(self, lock, iter, fold, feat, model_type, model_param, gini_cv):
            multiprocessing.Process.__init__(self)
            self.lock = lock
            self.iter = iter
            self.fold = fold
            self.feat = feat
            self.model_type = model_type
            self.model_param = model_param
            self.gini_cv = gini_cv

        def run(self):
            path = "%s/iter%d/fold%d" %(config.data_folder, self.iter, self.fold)
            with open("%s/train.%s.feat.pkl" %(path, self.feat), 'rb') as f:
                [x_train, y_train] = pickle.load(f)
            f.close()
            with open("%s/valid.%s.feat.pkl" %(path, self.feat), 'rb') as f:
                [x_test, y_test] = pickle.load(f)
            f.close()
            pred_val = hyperopt_library(self.model_type, self.model_param, x_train, y_train, x_test, y_test)

            # save the pred for cross validation
            pred_file = "%s/%s_%s@%d.pred.pkl" %(path, self.feat, self.model_type, trials_counter)
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_val, f, -1)
            f.close()

            if self.model_type == 'logistic':
                y_test = y_test / np.linalg.norm(y_test)

            self.gini_cv.append( ml_score(y_test, pred_val) )

    def hyperopt_wrapper(model_param):
        import pandas as pd
        import numpy as np
        import cPickle as pickle
        import os

        from sklearn import preprocessing
        from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
        from sklearn.svm import SVR, SVC
        from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
        from sklearn import neighbors

        import xgboost as xgb

        from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL, pyll
        from hyperopt.mongoexp import MongoTrials


        import time
        import sys
        import multiprocessing

        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation, Flatten
        from keras.optimizers import SGD, Adadelta, Adagrad

        trials_counter = get_trials_counter(model_type, feat)
        gini_cv_mean, gini_cv_std = hyperopt_obj(model_param, model_type, feat, trials_counter)
        return {'loss': -gini_cv_mean, 'status': STATUS_OK}

    return hyperopt_wrapper


def hyperopt_main():
    feat_names = config.feat_names
    model_list = config.model_list
    param_best_dic = {}
    model_file = "%s/model_best_params" %config.data_folder
    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            param_best_dic = pickle.load(f)
        f.close()

    for feat in feat_names:
        for model in model_list:
            model_key = "%s_%s"%(feat, model)
            if param_best_dic.has_key(model_key) is False:
                print "Training model %s_%s ......" %(feat, model)
                model_param = config.param_spaces[model]

                # use Mongo to do parallel search
                if config.use_mongo:
                    trials = MongoTrials(config.mongo_server + 'kaggle/jobs', exp_key='exp_%s'%model_key)
                else:
                    trials = Trials()
                #obj = lambda p: hyperopt_wrapper(p, model, feat)
                obj = make_obj(model, feat)
                tmp_max_evals = config.hyper_max_evals
                best_params = fmin(obj, model_param, algo=tpe.suggest, trials=trials, max_evals=tmp_max_evals)
                print best_params
                param_best_dic[model_key] = best_params

            with open(model_file, 'wb') as f:
                pickle.dump(param_best_dic, f, -1)
            f.close()

if __name__ == '__main__':
    start_time = time.time()

    # write your code here
    # apply different model on different feature, generate model library
    print "Code start at %s" %time.ctime()

    flag = sys.argv[1]
    if flag == "train":
        ## generate pred by best params
        one_model()

    if flag == "hyperopt":
        ## hyper parameter search
        hyperopt_main()



    end_time = time.time()
    print "cost time %f" %( (end_time - start_time)/1000 )
