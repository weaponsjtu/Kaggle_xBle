import os

from hyperopt import hp, pyll

import numpy as np

import multiprocessing

class ParamConfig:
    def __init__(self, data_folder):
        # target variable to predict
        self.target = 'Sales'
        self.tid = 'Id'
        self.type = 'regression' # regression, classification

        # (3,3), (2, 5), (5, 2)
        self.kfold = 3  # cross validation, k-fold
        self.kiter = 1  # shuffle dataset, and repeat CV

        self.DEBUG = True
        self.hyper_max_evals = 100
        self.ensemble_max_evals = 50

        self.nthread = 1
        self.max_core = multiprocessing.cpu_count()

        if self.DEBUG:
            self.hyper_max_evals = 1
            #self.ensemble_max_evals = 5

        self.origin_train_path = "../data/tr.csv"
        self.origin_test_path = "../data/te.csv"

        #self.feat_names = ['stand', 'label', 'dictvec', 'onehot']
        #self.feat_names = ['label', 'fs', 'small', 'onehot']
        self.feat_names = ['stand']

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
        #self.model_list = ['logistic']
        #self.model_list = ['extratree', 'knn', 'rf']
        self.model_list = ['fm', 'logistic', 'rgf', 'lasagne', 'ridge', 'extratree', 'rf', 'lasso', 'gbfC', 'knn', 'xgb_binary', 'xgb_auc', 'xgb_log', 'xgb_rank', 'xgb_count', 'xgb_tree_auc', 'xgb_tree_log', 'xgb_fix', 'xgb_linear_fix', 'xgb_fix_log', 'xgb_linear_fix_log']
        #self.model_list = ['xgb_fix', 'xgb_linear_fix', 'xgb_fix_log', 'xgb_linear_fix_log']

        self.update_model = ['']
        self.model_type = ''
        self.param_spaces = {
            'logistic': {
                'C': 1.0, #hp.loguniform('C', np.log(0.001), np.log(10)),
            },
            'knn': {
                'n_neighbors': 50, #pyll.scope.int(hp.quniform('n_neighbors', 2, 1024, 2)),
                #'weights': hp.choice('weights', ['uniform', 'distance']),
                'weights': 'distance',
            },
            'lasagne': {
                #TODO
                'loss': 'lr',
            },
            'knnC': {
                'n_neighbors': pyll.scope.int(hp.quniform('n_neighbors', 2, 1024, 2)),
                #'weights': hp.choice('weights', ['uniform', 'distance']),
                'weights': 'distance',
            },
            'ridge': {
                'alpha': 1.0, #hp.loguniform('alpha', np.log(0.01), np.log(20)),
            },
            'lasso': {
                'alpha': hp.loguniform('alpha', np.log(0.00001), np.log(0.1)),
            },
            'rf': {
                'n_estimators': 300, #pyll.scope.int(hp.quniform('n_estimators', 100, 500, 10)),
                'max_features': 0.7, #hp.quniform('max_features', 0.05, 1.0, 0.05),
            },
            'rfC': {
                'n_estimators': pyll.scope.int(hp.quniform('n_estimators', 100, 500, 100)),
                'max_features': hp.quniform('max_features', 0.05, 1.0, 0.05),
            },
            'extratree': {
                'n_estimators': 300, #pyll.scope.int(hp.quniform('n_estimators', 100, 500, 10)),
                'max_features': 0.7, #hp.quniform('max_features', 0.05, 1.0, 0.05),
                'max_depth': 20,
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
                'kernel': 'linear',
                'C': 1.0, #hp.quniform('C', 0.1, 10, 0.1),
                'epsilon': 0.1, # hp.loguniform('epsilon', np.log(0.001), np.log(0.1)),
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
                'eta': 0.01,
                'min_child_weight': 6,
                'subsample':0.7, #hp.quniform('subsample', 0.5, 1.0, 0.01),
                'eval_metric': 'auc',
                'colsample_bytree': 0.8,
                'scale_pos_weight': 1,
                'silent': 1,
                'verbose': 0,
                'max_depth': 8, #pyll.scope.int(hp.quniform('max_depth', 1, 10, 1)),
                'num_rounds': 3000,
                'early_stopping_rounds': 120,
                'nthread': 1,
            },
            'xgb_count': {
                'objective': 'count:poisson',
                'eta': 0.01,
                'min_child_weight': 6,
                'subsample':0.7, #hp.quniform('subsample', 0.5, 1.0, 0.01),
                'eval_metric': 'auc',
                'colsample_bytree': 0.8,
                'scale_pos_weight': 1,
                'silent': 1,
                'verbose': 0,
                'max_depth': 8, #pyll.scope.int(hp.quniform('max_depth', 1, 10, 1)),
                'num_rounds': 3000,
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
                'num_rounds': 5000,
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
                'num_rounds': 5000,
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
                'eta': 0.01,
                'min_child_weight': 6,
                'subsample':0.7, #hp.quniform('subsample', 0.5, 1.0, 0.01),
                'eval_metric': 'auc',
                'colsample_bytree': 0.8,
                'scale_pos_weight': 1,
                'silent': 1,
                'verbose': 0,
                'max_depth': 8, #pyll.scope.int(hp.quniform('max_depth', 1, 10, 1)),
                'num_rounds': 3000,
                'early_stopping_rounds': 120,
                'nthread': 1,
            },
            'xgb_fix_log': {
                'objective': 'binary:logistic',
                'eta': 0.01,
                'min_child_weight': 6,
                'subsample':0.7, #hp.quniform('subsample', 0.5, 1.0, 0.01),
                'eval_metric': 'logloss',
                'colsample_bytree': 0.8,
                'scale_pos_weight': 1,
                'silent': 1,
                'verbose': 0,
                'max_depth': 8, #pyll.scope.int(hp.quniform('max_depth', 1, 10, 1)),
                'num_rounds': 3000,
                'early_stopping_rounds': 120,
                'nthread': 1,
            },
            'xgb_linear_fix': {
                'objective': 'reg:linear',
                'eta': 0.01,
                'min_child_weight': 6,
                'subsample':0.7, #hp.quniform('subsample', 0.5, 1.0, 0.01),
                'eval_metric': 'auc',
                'colsample_bytree': 0.8,
                'scale_pos_weight': 1,
                'silent': 1,
                'verbose': 0,
                'max_depth': 8, #pyll.scope.int(hp.quniform('max_depth', 1, 10, 1)),
                'num_rounds': 3000,
                'early_stopping_rounds': 120,
                'nthread': 1,
            },
            'xgb_linear_fix_log': {
                'objective': 'reg:linear',
                'eta': 0.01,
                'min_child_weight': 6,
                'subsample':0.7, #hp.quniform('subsample', 0.5, 1.0, 0.01),
                'eval_metric': 'logloss',
                'colsample_bytree': 0.8,
                'scale_pos_weight': 1,
                'silent': 1,
                'verbose': 0,
                'max_depth': 8, #pyll.scope.int(hp.quniform('max_depth', 1, 10, 1)),
                'num_rounds': 3000,
                'early_stopping_rounds': 120,
                'nthread': 1,
            },
        }

config = ParamConfig("feat")
