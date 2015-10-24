###
# ensemble.py
# author: Weipeng Zhang
#
#
# 1. check each weight by hyperopt
# 2. apply the weight to train/test
###

from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

import cPickle as pickle
import numpy as np
import pandas as pd
import sys,os,time
import multiprocessing

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, pyll
from hyperopt.mongoexp import MongoTrials

from param import config

from utils import *




def add_prior_models(model_library):
    #prior_models = {
    #        'xgboost-art@1': {
    #            'weight': 0.463,
    #            'pow_weight': 0.01,
    #            },
    #        'xgboost-art@2': {
    #            'weight': 0.463,
    #            'pow_weight': 0.8,
    #            },
    #        'xgboost-art@3': {
    #            'weight': 0.463,
    #            'pow_weight': 0.045,
    #            'pow_weight1': 0.055,
    #            },
    #        'xgboost-art@4': {
    #            'weight': 0.463,
    #            'pow_weight': 0.98,
    #            },
    #        'xgboost-art@5': {
    #            'weight': 0.463,
    #            'pow_weight': 1,
    #            },
    #        'xgboost-art@6': {
    #            'weight': 0.47,
    #            'pow_weight': 1,
    #            },
    #        }
    prior_models = {}
    for i in range(1, 5):
        model = 'xgboost-art@%d'%i
        prior_models[model] = {'weight': 0.463, 'pow_weight': 0.01 * i}

    feat_names = config.feat_names
    model_list = config.model_list
    for iter in range(config.kiter):
        for fold in range(config.kfold):
            path = "%s/iter%d/fold%d" %(config.data_folder, iter, fold)
            with open("%s/label_encode_xgboost-art.pred.pkl" %path, 'rb') as f:
                p1 = pickle.load(f)
            with open("%s/dictvec_xgboost-art.pred.pkl" %path, 'rb') as f:
                p2 = pickle.load(f)
            for model in prior_models.keys():
                weight = prior_models[model]['weight']
                pow_weight1 = prior_models[model]['pow_weight']
                pow_weight2 = pow_weight1
                if prior_models[model].has_key('pow_weight1'):
                    pow_weight2 = prior_models[model]['pow_weight1']
                pred = weight * (p1**pow_weight1) + (1-weight) * (p2**pow_weight2)
                with open("%s/%s.pred.pkl" %(path, model), 'wb') as f:
                    pickle.dump(pred, f, -1)

    path = "%s/all" %(config.data_folder)
    with open("%s/label_encode_xgboost-art.pred.pkl" %path, 'rb') as f:
        p1 = pickle.load(f)
    with open("%s/dictvec_xgboost-art.pred.pkl" %path, 'rb') as f:
        p2 = pickle.load(f)
    for model in prior_models.keys():
        weight = prior_models[model]['weight']
        pow_weight1 = prior_models[model]['pow_weight']
        pow_weight2 = pow_weight1
        if prior_models[model].has_key('pow_weight1'):
            pow_weight2 = prior_models[model]['pow_weight1']
        pred = weight * (p1**pow_weight1) + (1-weight) * (p2**pow_weight2)
        with open("%s/%s.pred.pkl" %(path, model), 'wb') as f:
            pickle.dump(pred, f, -1)

    for model in prior_models.keys():
        model_library.append(model)
    return model_library


def ensemble_algorithm(p1, p2, weight):
    #return (p1 + weight*p2) / (1+weight)

    ### weighted linear combine ###
    #return 2.0 / (weight*(1.0/p1) + (1-weight)*(1.0/p2))
    #return (weight * np.log(p1) + (1-weight) * np.log(p2))
    return weight * p1 + (1-weight) * p2



def ensemble_selection_obj(param, model1_pred, model2_pred, labels, num_valid_matrix):
    weight = param['weight']
    gini_cv = np.zeros((config.kiter, config.kfold), dtype=float)
    for iter in range(config.kiter):
        for fold in range(config.kfold):
            p1 = model1_pred[iter, fold, :num_valid_matrix[iter, fold]]
            p2 = model2_pred[iter, fold, :num_valid_matrix[iter, fold]]
            y_pred = ensemble_algorithm(p1, p2, weight)

            y_true = labels[iter, fold, :num_valid_matrix[iter, fold]]
            score = ml_score(y_true, y_pred)
            gini_cv[iter][fold] = score
    gini_mean = np.mean(gini_cv)
    return -gini_mean


def ensemble_selection():
    # load feat, labels and pred
    feat_names = config.feat_names
    model_list = config.model_list

    # load model library
    model_library = gen_model_library()
    model_num = len(model_library)
    print model_library

    # num valid matrix
    num_valid_matrix = np.zeros((config.kiter, config.kfold), dtype=int)

    # load valid labels
    valid_labels = np.zeros((config.kiter, config.kfold, 50000), dtype=float)
    for iter in range(config.kiter):
        for fold in range(config.kfold):
            path = "%s/iter%d/fold%d" % (config.data_folder, iter, fold)
            label_file = "%s/valid.%s.feat.pkl" %(path, feat_names[0])
            with open(label_file, 'rb') as f:
                [x_val, y_true] = pickle.load(f)
            valid_labels[iter, fold, :y_true.shape[0]] = y_true
            num_valid_matrix[iter][fold] = y_true.shape[0]
    maxNumValid = np.max(num_valid_matrix)

    # load all predictions, cross validation
    # compute model's gini cv score
    gini_cv = []
    model_valid_pred = np.zeros((model_num, config.kiter, config.kfold, maxNumValid), dtype=float)

    for mid in range(model_num):
        gini_cv_tmp = np.zeros((config.kiter, config.kfold), dtype=float)
        for iter in range(config.kiter):
            for fold in range(config.kfold):
                path = "%s/iter%d/fold%d" % (config.data_folder, iter, fold)
                pred_file = "%s/%s.pred.pkl" %(path, model_library[mid])
                with open(pred_file, 'rb') as f:
                    y_pred = pickle.load(f)
                model_valid_pred[mid, iter, fold, :num_valid_matrix[iter, fold]] = y_pred
                score = ml_score(valid_labels[iter, fold, :num_valid_matrix[iter, fold]], y_pred)
                gini_cv_tmp[iter][fold] = score
        gini_cv.append(np.mean(gini_cv_tmp))

    # sort the model by their cv mean score
    gini_cv = np.array(gini_cv)
    sorted_model = gini_cv.argsort()[::-1] # large to small
    for mid in sorted_model:
        print model_library[mid]
    print len(sorted_model)


    # boosting ensemble
    # 1. initialization, use the max score model
    model_pred_tmp = np.zeros((config.kiter, config.kfold, maxNumValid), dtype=float)
    for iter in range(config.kiter):
        for fold in range(config.kfold):
            model_pred_tmp[iter, fold, :num_valid_matrix[iter][fold]] = model_valid_pred[sorted_model[0], iter, fold, :num_valid_matrix[iter][fold]]
    print "Init with best model, ml_score %f, Model %s" %(np.max(gini_cv), model_library[sorted_model[0]])

    # 2. greedy search
    best_model_list = []
    best_weight_list = []

    if config.nthread > 1:
        manager = multiprocessing.Manager()
        best_gini_tmp = manager.list()
        best_gini_tmp.append( np.max(gini_cv) )
        best_weight_tmp = manager.list()
        best_weight_tmp.append(-1)
        best_model_tmp = manager.list()
        best_model_tmp.append(-1)
    else:
        best_gini = np.max(gini_cv)
        best_weight = None
        best_model = None
    ensemble_iter = 0
    while True:
        iter_time = time.time()
        ensemble_iter += 1
        if config.nthread > 1:
            best_model_tmp[0] = -1
            model_id = 0
            while model_id < len(sorted_model):
                mp_list = []
                for i in range(model_id, min(len(sorted_model), model_id + config.max_core)):
                    mp = EnsembleProcess(ensemble_iter, sorted_model[i] , model_library, sorted_model, model_pred_tmp, model_valid_pred, valid_labels, num_valid_matrix, best_gini_tmp, best_weight_tmp, best_model_tmp)
                    mp_list.append(mp)

                model_id += config.max_core

                for mp in mp_list:
                    mp.start()

                for mp in mp_list:
                    mp.join()

            best_gini = best_gini_tmp[0]
            best_weight = best_weight_tmp[0]
            best_model = best_model_tmp[0]
            if best_model == -1:
                best_model = None

            # TODO
        else:
            for model in sorted_model:
                print "ensemble iter %d, model (%d, %s)" %(ensemble_iter, model, model_library[model])
                # jump for the first max model
                #if ensemble_iter == 1 and model == sorted_model[0]:
                #    continue

                obj = lambda param: ensemble_selection_obj(param, model_pred_tmp, model_valid_pred[model], valid_labels, num_valid_matrix)
                param_space = {
                    'weight': hp.quniform('weight', 0, 1, 0.01),
                }
                trials = Trials()
                #trials = MongoTrials('mongo://172.16.13.7:27017/ensemble/jobs', exp_key='exp%d_%d'%(ensemble_iter, model))
                best_param = fmin(obj,
                    space = param_space,
                    algo = tpe.suggest,
                    max_evals = config.ensemble_max_evals,
                    trials = trials)
                best_w = best_param['weight']

                gini_cv_tmp = np.zeros((config.kiter, config.kfold), dtype=float)
                for iter in range(config.kiter):
                    for fold in range(config.kfold):
                        p1 = model_pred_tmp[iter, fold, :num_valid_matrix[iter, fold]]
                        p2 = model_valid_pred[model, iter, fold, :num_valid_matrix[iter, fold]]
                        y_true = valid_labels[iter, fold, :num_valid_matrix[iter, fold]]
                        y_pred = ensemble_algorithm(p1, p2, best_w)
                        score = ml_score(y_true, y_pred)
                        gini_cv_tmp[iter, fold] = score


                print "Iter %d, ml_score %f, Model %s, Weight %f" %(ensemble_iter, np.mean(gini_cv_tmp), model_library[model], best_w)
                if (np.mean(gini_cv_tmp) - best_gini) >= 0.000001:
                    best_gini, best_model, best_weight = np.mean(gini_cv_tmp), model, best_w
                #### single process

        if best_model == None: #or best_weight > 0.9:
            break
        print "Best for Iter %d, ml_score %f, Model %s, Weight %f" %(ensemble_iter, best_gini, model_library[best_model], best_weight)
        best_weight_list.append(best_weight)
        best_model_list.append(best_model)

        # reset the valid pred
        for iter in range(config.kiter):
            for fold in range(config.kfold):
                p1 = model_pred_tmp[iter, fold, :num_valid_matrix[iter, fold]]
                p2 = model_valid_pred[best_model, iter, fold, :num_valid_matrix[iter, fold]]
                model_pred_tmp[iter, fold, :num_valid_matrix[iter][fold]] = ensemble_algorithm(p1, p2, best_weight)

        best_model = None
        print 'ensemble iter %d done!!!, cost time is %s' % (ensemble_iter, time.time() - iter_time)

        # save best model list, every iteration
        with open("%s/best_model_list" % config.data_folder, 'wb') as f:
            pickle.dump([model_library, sorted_model, best_model_list, best_weight_list], f, -1)

def ensemble_prediction():
    # load best model list
    with open("%s/best_model_list" % config.data_folder, 'rb') as f:
        [model_library, sorted_model, best_model_list, best_weight_list] = pickle.load(f)

    # prediction, generate submission file
    path = "%s/all" % config.data_folder
    print "Init with (%s)" %(model_library[sorted_model[0]])
    with open("%s/%s.pred.pkl" %(path, model_library[sorted_model[0]]), 'rb') as f:
        y_pred = pickle.load(f)

    # generate best single model submission
    gen_subm(y_pred, 'sub/best_single.csv')


    for i in range(len(best_model_list)):
        model = best_model_list[i]
        weight = best_weight_list[i]
        print "(%s), %f" %(model_library[model], weight)
        with open("%s/%s.pred.pkl" %(path, model_library[model]), 'rb') as f:
            y_pred_tmp = pickle.load(f)
        y_pred = ensemble_algorithm(y_pred, y_pred_tmp, weight)

    # generate ensemble submission finally
    gen_subm(y_pred)

class EnsembleProcess(multiprocessing.Process):
    def __init__(self, ensemble_iter, model, model_library, sorted_model, model_pred_tmp, model_valid_pred, valid_labels, num_valid_matrix, best_gini, best_weight, best_model):
        multiprocessing.Process.__init__(self)
        self.ensemble_iter = ensemble_iter
        self.model = model
        self.model_library = model_library
        self.sorted_model = sorted_model
        self.model_pred_tmp = model_pred_tmp
        self.model_valid_pred = model_valid_pred
        self.valid_labels = valid_labels
        self.num_valid_matrix = num_valid_matrix
        self.best_gini = best_gini
        self.best_weight = best_weight
        self.best_model = best_model

    def run(self):
        print "ensemble iter %d, model (%d, %s)" %(self.ensemble_iter, self.model, self.model_library[self.model])
        # jump for the first max model
        #if self.ensemble_iter == 1 and self.model == self.sorted_model[0]:
        #    return

        obj = lambda param: ensemble_selection_obj(param, self.model_pred_tmp, self.model_valid_pred[self.model], self.valid_labels, self.num_valid_matrix)
        param_space = {
            'weight': hp.quniform('weight', 0, 1, 0.01),
        }
        trials = Trials()
        #trials = MongoTrials('mongo://172.16.13.7:27017/ensemble/jobs', exp_key='exp%d_%d'%(ensemble_iter, model))
        best_param = fmin(obj,
            space = param_space,
            algo = tpe.suggest,
            max_evals = config.ensemble_max_evals,
            trials = trials)
        best_w = best_param['weight']

        gini_cv_tmp = np.zeros((config.kiter, config.kfold), dtype=float)
        for iter in range(config.kiter):
            for fold in range(config.kfold):
                p1 = self.model_pred_tmp[iter, fold, :self.num_valid_matrix[iter, fold]]
                p2 = self.model_valid_pred[self.model, iter, fold, :self.num_valid_matrix[iter, fold]]
                y_true = self.valid_labels[iter, fold, :self.num_valid_matrix[iter, fold]]
                y_pred = ensemble_algorithm(p1, p2, best_w)
                score = ml_score(y_true, y_pred)
                gini_cv_tmp[iter, fold] = score


        print "Iter %d, ml_score %f, Model %s, Weight %f" %(self.ensemble_iter, np.mean(gini_cv_tmp), self.model_library[self.model], best_w)
        if (np.mean(gini_cv_tmp) - self.best_gini[0]) >= 0.000001:
            self.best_gini[0], self.best_model[0], self.best_weight[0] = np.mean(gini_cv_tmp), self.model, best_w


def ensemble_rank_average():
    # load feat, labels and pred
    feat_names = config.feat_names
    model_list = config.model_list

    # load model library
    #model_library = gen_model_library()
    model_library = ['label_xgb_linear_fix@6', 'label_xgb_fix@6', 'label_xgb_count@6', 'label_xgb_rank@6', 'label_lasagne@6', 'label_extratree@6', 'label_ridge@6', 'label_rgf@6', 'label_logistic@6']
    for model in model_library:
        with open('%s/all/%s.pred.pkl'%(config.data_folder, model), 'rb') as f:
            y_pred = pickle.load(f)
        gen_subm(y_pred, 'sub/final/%s.csv'%model)
    return 0

    #label_xgb_fix_log@6 0.999108398851
    #label_xgb_linear_fix@6 0.979159034293
    #label_xgb_fix@6 1.0
    #label_xgb_fix@1 0.980509196355
    #label_xgb_tree_log@2 0.979845717419
    #label_xgb_tree_log@1 0.978312937118
    #label_xgb_count@6 0.977797411688
    #label_xgb_rank@6 0.928070928806

    model_num = len(model_library)
    print model_library
    print model_num

    # num valid matrix
    num_valid_matrix = np.zeros((config.kiter, config.kfold), dtype=int)

    # load valid labels
    valid_labels = np.zeros((config.kiter, config.kfold, 50000), dtype=float)
    for iter in range(config.kiter):
        for fold in range(config.kfold):
            path = "%s/iter%d/fold%d" % (config.data_folder, iter, fold)
            label_file = "%s/valid.%s.feat.pkl" %(path, feat_names[0])
            with open(label_file, 'rb') as f:
                [x_val, y_true] = pickle.load(f)
            valid_labels[iter, fold, :y_true.shape[0]] = y_true
            num_valid_matrix[iter][fold] = y_true.shape[0]
    maxNumValid = np.max(num_valid_matrix)
    print "valid labels done!!!"

    # load all predictions, cross validation
    # compute model's gini cv score
    gini_cv = []
    model_valid_pred = np.zeros((model_num, config.kiter, config.kfold, maxNumValid), dtype=float)

    for mid in range(model_num):
        gini_cv_tmp = np.zeros((config.kiter, config.kfold), dtype=float)
        for iter in range(config.kiter):
            for fold in range(config.kfold):
                path = "%s/iter%d/fold%d" % (config.data_folder, iter, fold)
                pred_file = "%s/%s.pred.pkl" %(path, model_library[mid])
                with open(pred_file, 'rb') as f:
                    y_pred = pickle.load(f)
                model_valid_pred[mid, iter, fold, :num_valid_matrix[iter, fold]] = y_pred
                score = ml_score(valid_labels[iter, fold, :num_valid_matrix[iter, fold]], y_pred)
                gini_cv_tmp[iter][fold] = score
        gini_cv.append(np.mean(gini_cv_tmp))
    print "gini cv done!!!"

    # sort the model by their cv mean score
    gini_cv = np.array(gini_cv)
    sorted_model = gini_cv.argsort()[::-1]


    ### rank average ###
    best_model_end = 0
    best_gini = 0
    for model_end_id in range(len(sorted_model)):
        gini_cv_tmp = np.zeros((config.kiter, config.kfold), dtype=float)
        for iter in range(config.kiter):
            for fold in range(config.kfold):
                pred_tmp = np.zeros((num_valid_matrix[iter, fold]), dtype=float)
                #pred_tmp = np.ones((num_valid_matrix[iter, fold]), dtype=float)
                for mid in range(model_end_id+1):
                    y_pred = model_valid_pred[sorted_model[mid], iter, fold, :num_valid_matrix[iter, fold]]
                    #pred_tmp += (y_pred.argsort() + 1)*1.0 / len(y_pred)
                    pred_tmp +=  y_pred # log mean, harmonic mean
                    #pred_tmp *= y_pred
                #pred_tmp = (model_end_id + 1)*1.0 / pred_tmp
                pred_tmp /= (model_end_id + 1)
                #pred_tmp = np.power(pred_tmp, 1.0/(model_end_id + 1))
                gini_cv_tmp[iter, fold] = ml_score( valid_labels[iter, fold, :num_valid_matrix[iter, fold]], pred_tmp)
        if np.mean(gini_cv_tmp) > best_gini:
            best_model_end = model_end_id
            best_gini = np.mean(gini_cv_tmp)
        print "model end id %d, best_gini %f" %(model_end_id, np.mean(gini_cv_tmp))

    print best_model_end
    print best_gini

    #path = "%s/all/%s.pred.pkl" %(config.data_folder, model_library[ sorted_model[0] ])
    #with open(path, 'rb') as f:
    #    y_pred = pickle.load(f)
    #    y_pred = (y_pred.argsort() + 1)*1.0 / len(y_pred)

    #for mid in range(1, best_model_end + 1):
    #    path = "%s/all/%s.pred.pkl" %(config.data_folder, model_library[ sorted_model[mid] ])
    #    with open(path, 'rb') as f:
    #        y_pred_tmp = pickle.load(f)
    #    y_pred += (y_pred_tmp.argsort() + 1)*1.0 / len(y_pred)
    #    #y_pred += y_pred_tmp

    #y_pred = y_pred * 1.0 / (best_model_end + 1)
    #gen_subm(y_pred, 'sub/model_rank_avg.csv')

    #TODO


if __name__ == "__main__":
    start_time = time.time()

    flag = sys.argv[1]
    print "start ", flag
    print "Code start at %s" %time.ctime()
    if flag == "ensemble":
        ensemble_selection()
    if flag == "submission":
        ensemble_prediction()
    if flag == "rankavg":
        ensemble_rank_average()

    end_time = time.time()
    print "cost time %f" %( (end_time - start_time)/1000 )
