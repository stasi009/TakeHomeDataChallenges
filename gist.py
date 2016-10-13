
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve


def validation_roc():
    Xtrain_only, Xvalid, ytrain_only, yvalid = train_test_split(Xtrain, ytrain, test_size=0.2, random_state=seed)

    train_only_matrix = xgb.DMatrix(Xtrain_only, ytrain_only)
    valid_matrix = xgb.DMatrix(Xvalid)

    # retrain on training set
    gbt_train_only = xgb.train(params, train_only_matrix, n_best_trees)

    # predict on validation set
    yvalid_probas = gbt_train_only.predict(valid_matrix, ntree_limit=n_best_trees)

    d = {}
    d['FPR'], d['TPR'], d['Threshold'] = roc_curve(yvalid, yvalid_probas)
    return pd.DataFrame(d)