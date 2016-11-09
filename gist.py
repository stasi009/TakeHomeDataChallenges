
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


def sort_neighbors(X):
    Xtrain = X.loc[X.index != 'Missing', :]
    countries = Xtrain.index

    neigh = NearestNeighbors(n_neighbors=Xtrain.shape[0])  # return all neighbors
    neigh.fit(Xtrain)

    distance, indices = neigh.kneighbors(X.loc[['Missing'], :])

    distance = distance[0]
    indices = indices[0]
    countries = countries[indices]

    return pd.DataFrame(zip(countries, distance), columns=['country', 'distance'])

####################################
params = {}
params['silent'] = 1
params['objective'] = 'binary:logistic'  # output probabilities
params['eval_metric'] = 'auc'
params["num_rounds"] = 300
params["early_stopping_rounds"] = 30
# params['min_child_weight'] = 2
params['max_depth'] = 6
params['eta'] = 0.1
params["subsample"] = 0.8
params["colsample_bytree"] = 0.8

cv_results = xgb.cv(params,train_matrix,
                    num_boost_round = params["num_rounds"],
                    nfold = params.get('nfold',5),
                    metrics = params['eval_metric'],
                    early_stopping_rounds = params["early_stopping_rounds"],
                    verbose_eval = True,
                    seed = seed)

watchlist = [(train_matrix, 'train')]
gbt = xgb.train(params, train_matrix, n_best_trees,watchlist)
gbt.predict(matrix, ntree_limit=n_best_trees)

xgb.plot_importance(gbt)
###############################################
dt = DecisionTreeClassifier(max_depth=3,min_samples_leaf=20,min_samples_split=20)
dt.fit(X,y)
export_graphviz(dt,feature_names=X.columns,class_names=['NotFraud','Fraud'],
                proportion=True,leaves_parallel=True,filled=True)

dot -Tpng tree.dot -o tree.png
