
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score,roc_curve,classification_report

import matplotlib.pyplot as plt
plt.style.use('ggplot')

seed = 999

datas = pd.read_csv("fraud_cleaned.csv",index_col='user_id')

X = datas.loc[:,datas.columns != 'is_fraud']
y = datas.is_fraud

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,random_state=seed)

train_matrix = xgb.DMatrix(Xtrain,ytrain)
test_matrix = xgb.DMatrix(Xtest)

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

n_best_trees = cv_results.shape[0]

watchlist = [(train_matrix, 'train')]
gbt = xgb.train(params, train_matrix, n_best_trees,watchlist)

xgb.plot_importance(gbt)
datas.groupby("age")['is_fraud'].agg(['count','mean'])

## false negative cost more
## false positive is acceptable
########### plot ROC on validation set
Xtrain_only,Xvalid,ytrain_only,yvalid = train_test_split(Xtrain,ytrain,test_size=0.3,random_state=seed)
onlytrain_matrix = xgb.DMatrix(Xtrain_only,ytrain_only)
valid_matrix = xgb.DMatrix(Xvalid,yvalid)

temp_gbt = xgb.train(params, onlytrain_matrix, n_best_trees,[(onlytrain_matrix,'train_only'),(valid_matrix,'validate')])
yvalid_proba_pred = temp_gbt.predict(valid_matrix,ntree_limit=n_best_trees)

fpr,tpr,thresholds = roc_curve(yvalid,yvalid_proba_pred)
roc = pd.DataFrame({'FPR':fpr,'TPR':tpr,'Threshold':thresholds})

plt.plot(roc.FPR,roc.TPR,marker='o')
plt.xlabel("FPR")
plt.ylabel("TPR")


roc.loc[ (roc.TPR >= 0.78) & (roc.TPR <=0.83),:]

proba_threshold = 0.1695

##############
ytest_proba_pred = gbt.predict(test_matrix,ntree_limit=n_best_trees)
ytest_pred = (ytest_proba_pred >= proba_threshold).astype(int)

print classification_report(ytest,ytest_pred)

accuracy_score(ytest,ytest_pred)

############### simple decision tree
dt = DecisionTreeClassifier(max_depth=3,min_samples_leaf=20,min_samples_split=20)
dt.fit(X,y)
export_graphviz(dt,feature_names=X.columns,class_names=['NotFraud','Fraud'],
                proportion=True,leaves_parallel=True,filled=True)