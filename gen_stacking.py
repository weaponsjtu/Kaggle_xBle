###
# gen_stacking.py
# author: Weipeng Zhang
#
#
# 1. prediction feature, LogisticRegression
# 2. prediction feature + origin feature, non-linear, GBF, KNN, ET, etc
###

from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn import cross_validation
from sklearn import manifold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

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

class StackProcess(multiprocessing.Process):
    def __init__(self, x_train, y_train, x_test, y_test, gini_cv):
        multiprocessing.Process.__init__(self)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.gini_cv = gini_cv

    def run(self):
        y_pred = xgb_train(self.x_train, self.y_train, self.x_test, self.y_test)
        self.gini_cv.append(ml_score(self.y_test, y_pred))
        print "Process %d done! ml_score is %f" %(os.getpid(),  ml_score(self.y_test, y_pred))


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
            #for num in range(config.hyper_max_evals):
            for num in range(6,7):
                model_name = "%s_%s@%d" %(feat, model, num)
                if check_model(model_name):
                    model_library.append(model_name)
                    #break

    #model_library = add_prior_models(model_library)
    return model_library

# stacking
def model_stacking():
    model_library = gen_base_model()
    print len(model_library)
    print model_library

    best_model = []
    for model in model_library:
        score_cv = []
        for iter in range(config.kiter):
            for fold in range(config.kfold):
                with open("%s/iter%d/fold%d/%s.pred.pkl"%(config.data_folder, iter, fold, model), 'rb') as f:
                    y_pred = pickle.load(f)
                with open("%s/iter%d/fold%d/valid.true.pkl"%(config.data_folder, iter, fold), 'rb') as f:
                    y_true = pickle.load(f)
                score_cv.append( ml_score(y_true, y_pred) )
        if np.mean(score_cv) > 0.7:
            print model, np.mean(score_cv)
            best_model.append(model)
    model_library = best_model
    #############################################

    # load data
    #train = pd.read_csv(config.origin_train_path, index_col=0)
    #test = pd.read_csv(config.origin_test_path, index_col=0)
    with open('%s/all/train.label.feat.pkl' %config.data_folder, 'rb') as f:
        [x_feat, x_label] = pickle.load(f)

    #x_label = np.array(train[config.target].values)
    with open('%s/all/test.label.feat.pkl' %config.data_folder, 'rb') as f:
        [t_feat, t_label] = pickle.load(f)
    y_len = len(t_label)


    blend_train = np.zeros((len(x_label), len(model_library), config.kiter))
    blend_test = np.zeros((y_len, len(model_library)))

    # load kfold object
    with open("%s/fold.pkl" % config.data_folder, 'rb') as i_f:
        skf = pickle.load(i_f)
    i_f.close()

    for iter in range(config.kiter):
        for i in range(len(model_library)):
            for j, (trainInd, validInd) in enumerate(skf[iter]):
                path = "%s/iter%d/fold%d/%s.pred.pkl" %(config.data_folder, iter, j, model_library[i])
                with open(path, 'rb') as f:
                    y_pred = pickle.load(f)
                f.close()
                #print "ml_score score is %f" % ml_score(x_label[validInd], y_pred)
                blend_train[validInd, i, iter] = y_pred
            #blend_train[:, i, iter] /= config.kfold


    for i in range(len(model_library)):
        path = "%s/all/%s.pred.pkl" %(config.data_folder, model_library[i])
        with open(path, 'rb') as f:
            y_pred = pickle.load(f)
        f.close()
        blend_test[:, i] = y_pred

    return blend_train, blend_test, x_label, model_library

from nolearn.lasagne import NeuralNet
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.updates import adagrad, nesterov_momentum
from lasagne.nonlinearities import softmax, sigmoid
from lasagne.objectives import categorical_crossentropy, binary_crossentropy
def lasagne_model(num_features, num_classes):
    layers = [('input', InputLayer),
            ('dense0', DenseLayer),
            ('dropout0', DropoutLayer),
            ('dense1', DenseLayer),
            ('dropout1', DropoutLayer),
            ('dense2', DenseLayer),
            ('dropout2', DropoutLayer),
            ('output', DenseLayer)]

    model = NeuralNet(layers=layers,
            input_shape=(None, num_features),
            #objective_loss_function=binary_crossentropy,
            dense0_num_units=1024,
            dropout0_p=0.4, #0.1,
            dense1_num_units=512,
            dropout1_p=0.4, #0.1,
            dense2_num_units=256,
            dropout2_p=0.4, #0.1,
            output_num_units=num_classes,
            output_nonlinearity=sigmoid,
            regression=True,
            update=nesterov_momentum, #adagrad,
            update_momentum=0.9,
            update_learning_rate=0.004,
            eval_size=0.2,
            verbose=1,
            max_epochs=30) #15)
    return model

def xgb_train(x_train, x_label, x_test, y_test):
    #x_train = max_min(x_train)
    #x_test = max_min(x_test)
    #x_train = np.log(x_train / (1 - x_train))
    #x_test = np.log(x_test / (1 - x_test))
    #x_train = np.log(x_train + 1)
    #x_test = np.log(x_test + 1)

    # standazition
    #sc = StandardScaler(copy=True, with_mean=True, with_std=True)
    #sc.fit(x_train)
    #x_train = sc.transform(x_train)
    #x_test = sc.transform(x_test)

    print "Feature dimension %d" %x_train.shape[1]

    #params = {}
    ##params["objective"] = "binary:logistic"
    #params["objective"] = "rank:pairwise"
    #params["eta"] = 0.01  # [0,1]
    #params["min_child_weight"] = 6#30 #6
    #params["subsample"] = 0.7
    #params["colsample_bytree"] = 1
    #params["scale_pos_weight"] = 1.0
    #params["eval_metric"] = 'auc'
    #params["silent"] = 1
    #params["max_depth"] = 8#4 #8
    #params["nthread"] = 16

    #num_rounds = 5000

    #xgtrain = xgb.DMatrix(x_train, label=x_label)
    #xgval = xgb.DMatrix(x_test, label=y_test)

    #watchlist = [(xgtrain, "train"), (xgval, "val")]
    #model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
    #pred = model.predict( xgval )


    clf = lasagne_model(x_train.shape[1], 1)
    #clf = RandomForestRegressor()
    #clf = LogisticRegression()
    #clf = GradientBoostingRegressor()
    #cal_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)

    #clf = Lasso()
    #clf = AdaBoostRegressor( ExtraTreesRegressor(max_depth=8), n_estimators=20 )
    clf.fit(x_train, x_label)
    pred = clf.predict(x_test)
    #cal_clf.fit(x_train, x_label)
    #pred = cal_clf.predict_proba(x_test)
    #pred = pred[:, 1]

    #pred = pred1 * pred2 / (pred1 + pred2)
    #pred = 0.7 * (pred1**0.01) + 0.3 * (pred2**0.01)
    #pred = (pred1.argsort() + pred2.argsort()) / 2
    #pred =  (pred1 +  pred2)/2

    return pred

def stacking_base(blend_train, blend_test, x_label, model_library):
    print "Blending..., 4->9"

    ## kfold cv
    #kf = cross_validation.KFold(n=blend_train.shape[0], n_folds=3, shuffle=False, random_state=None)
    with open("%s/fold.pkl" % config.data_folder, 'rb') as i_f:
        kf = pickle.load(i_f)


    config.nthread = 1
    if config.nthread > 1:
        mp_list = []
        manager = multiprocessing.Manager()
        gini_cv = manager.list()
        for iter in range(config.kiter):
            #for train_index, test_index in kf:
            for fold, (train_index, test_index) in enumerate(kf[iter]):
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
        for iter in range(config.kiter):
            #for train_index, test_index in kf:
            for fold, (train_index, test_index) in enumerate(kf[iter]):
                x_train, x_test = blend_train[train_index, :, iter], blend_train[test_index, :, iter]
                y_train, y_test = x_label[train_index], x_label[test_index]
                #clf.fit(x_train, y_train)
                #y_pred = clf.predict(x_test)
                y_pred = xgb_train(x_train, y_train, x_test, y_test)
                gini_cv.append(ml_score(y_test, y_pred))
                break
        print "Average gini is %f" %(np.mean(gini_cv))


    #y_sub = np.zeros((blend_test.shape[0]))
    ##for iter in range(config.kiter):
    #for iter in range(1):
    #    #clf.fit(blend_train[:, :, iter], x_label)
    #    #y_pred = clf.predict(blend_test)
    #    y_pred = xgb_train(blend_train[:, :, iter], x_label, blend_test)
    #    y_sub += y_pred
    #y_sub = y_sub / config.kiter
    #gen_subm(y_sub, 'sub/model_stack.csv')

def stacking_nonlinear(blend_train_raw, blend_test_raw, x_label, model_library):
    print "stacking nonlinear"
    feat = "label"

    # add raw feature
    with open("%s/all/train.%s.feat.pkl" %(config.data_folder, feat), 'rb') as f:
        [x_train, y_train] = pickle.load(f)
    with open("%s/all/test.%s.feat.pkl" %(config.data_folder, feat), 'rb') as f:
        [x_test, y_test] = pickle.load(f)

    print "Raw Feature Dimension is %d" %x_train.shape[1]

    if type(x_train) != np.ndarray:
        x_train = x_train.toarray()
        x_test = x_test.toarray()

    # check base
    #model_library = []

    meta_feature_dim = x_train.shape[1]
    blend_train = np.zeros((len(x_label), len(model_library) + meta_feature_dim, config.kiter))
    #blend_train[:, :len(model_library), :] = blend_train_raw
    blend_test = np.zeros((blend_test_raw.shape[0], len(model_library) + meta_feature_dim))
    #blend_test[:, :len(model_library)] = blend_test_raw

    ## KNN feature, 4 dimension
    #with open('../train_feature_engineered.pkl', 'rb') as f:
    #    df_train = pickle.load(f)
    #with open('../test_feature_engineered.pkl', 'rb') as f:
    #    df_test = pickle.load(f)

    #knn_feats = ['meanH_N_1', 'meanH_N_2', 'meanH_N_4', 'meanH_N_8']
    #for i in range(4):
    #    for iter in range(config.kiter):
    #        blend_train[:, len(model_library) + i, iter] = df_train[ knn_feats[i] ].values
    #    blend_test[:, len(model_library) + i] = df_test[ knn_feats[i] ].values

    for iter in range(config.kiter):
        blend_train[:, len(model_library):, iter] = x_train
    blend_test[:, len(model_library):] = x_test



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





    # stacking
    stacking_base(blend_train, blend_test, x_label, model_library)



if __name__ == '__main__':
    start_time = time.time()
    print "Code start at %s" %time.ctime()
    #model_stacking()
    blend_train, blend_test, x_label, model_library = model_stacking()
    print "Feature done!!!"
    stacking_base(blend_train, blend_test, x_label, model_library)
    #stacking_nonlinear(blend_train, blend_test, x_label, model_library)

    end_time = time.time()
    print "cost time %f" %( (end_time - start_time))
