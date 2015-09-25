import cPickle as pickle

from sklearn import cross_validation

import pandas as pd
import numpy as np
import os

from param import config

train = pd.read_csv(config.origin_train_path, header=0)

y = train[config.target].values

skf = [0]*config.kiter
for i in range(config.kiter):
    seed = 711 +(i+1) * 1000
    skf[i] = cross_validation.StratifiedKFold(y, n_folds=config.kfold, shuffle=True, random_state=seed)

with open("%s/fold.pkl" % config.data_folder, 'wb') as f:
    pickle.dump(skf, f, -1)
