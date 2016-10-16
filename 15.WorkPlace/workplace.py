
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import  train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import chi2
import xgboost as xgb

import matplotlib.pyplot as plt
plt.style.use('ggplot')


hierarchy = pd.read_csv("company_hierarchy.csv",index_col='employee_id')
hierarchy['level'] = None

hierarchy.loc[hierarchy.dept == 'CEO','level'] = 'CEO'

hierarchy.loc[hierarchy.level == 'CEO','boss_id'] = -1
hierarchy['boss_id'] = hierarchy.boss_id.astype(int)

def set_level(boss_level,level):
    boss_ids = hierarchy.loc[hierarchy.level == boss_level,:].index
    is_subordinate = np.in1d(hierarchy.boss_id,boss_ids)
    hierarchy.loc[is_subordinate,'level'] = level

set_level('CEO','E')
set_level('E','VP')
set_level('VP','D')
set_level('D','MM')
set_level('MM','IC')

hierarchy.level.value_counts()

###################################
hierarchy['n_subordinates'] = 0

def __count_subordinates(s):
    n_direct_subordinates = s.shape[0]
    n_indirect_subordinates = s.sum()
    return n_direct_subordinates + n_indirect_subordinates

def count_subordinates(subordinate_level):
    num_subordinates = hierarchy.loc[hierarchy.level == subordinate_level,:].groupby('boss_id')['n_subordinates'].agg(__count_subordinates)
    hierarchy.loc[num_subordinates.index,'n_subordinates'] = num_subordinates

count_subordinates(subordinate_level="IC")
count_subordinates(subordinate_level="MM")
count_subordinates(subordinate_level="D")
count_subordinates(subordinate_level="VP")
count_subordinates(subordinate_level="E")

###############################################
employees = pd.read_csv("employee.csv",index_col="employee_id")

employees = employees.join(hierarchy)
employees["salary"] /= 1000

employees.to_csv("all_employees.csv",index_label="employee_id")

employees.loc[employees.level == 'IC','n_subordinates']

#####################################
X = employees.copy()
X["is_male"] = (X.sex == "M").astype(int)
del X["sex"]
del X['boss_id']

index2level = ['IC','MM',"D","VP","E","CEO"]
level2index = {l:index for index,l in enumerate(index2level)}

index2degree = ['High_School','Bachelor','Master','PhD']
degree2index = {d:index for index,d in enumerate(index2degree)}

X['level'] = X.level.map(level2index)
np.corrcoef(X.level,X.n_subordinates)

X['degree'] = X.degree_level.map(degree2index)
del X['degree_level']

X = pd.get_dummies(X)
del X['dept_CEO']

X.to_csv("preproc_employees.csv",index_label="employee_id")



# since there is only one CEO, and its salary is the highest
# either put it into train or test, it will become a outlier
X = X.loc[X.level !=5,:]
y = np.log(X['salary'])
del X['salary']

seed = 999
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,random_state=seed)

train_matrix = xgb.DMatrix(Xtrain,ytrain)
test_matrix = xgb.DMatrix(Xtest)

params = {}
params['silent'] = 1
params['objective'] = 'reg:linear'
params['eval_metric'] = 'rmse'
params["num_rounds"] = 300
params["early_stopping_rounds"] = 30
# params['min_child_weight'] = 2
# params['max_depth'] = 6
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


ytest_pred = gbt.predict(test_matrix, ntree_limit=n_best_trees)
np.sqrt(mean_squared_error(ytest,ytest_pred))

r2_score(ytest,ytest_pred)

########################################
rfg = RandomForestRegressor(n_estimators=300,oob_score=True,
                            n_jobs=-1,random_state=seed,verbose=1)
rfg.fit(Xtrain,ytrain)
pd.Series(rfg.feature_importances_,index = Xtrain.columns).sort_values(ascending=False)

ytest_pred = rfg.predict(Xtest)
np.sqrt( mean_squared_error(ytest,ytest_pred) )

(ytest_pred - ytest).hist(bins=100)

###########################################

##### residual analysis to find bias
whole_matrix = xgb.DMatrix(X)
ypred = gbt.predict(whole_matrix)

predresult = pd.DataFrame({'ytrue': np.exp(y),'ypred': np.exp(ypred)})
predresult['bias'] = predresult.ytrue - predresult.ypred

# predresult = X.join(predresult.loc[:,['bias']])


predresult = predresult.join(employees)
del predresult['ytrue']

plt.scatter(predresult.n_subordinates,predresult.bias)

predresult.loc[predresult.sex == 'M','bias'].hist(bins=100,label='Male')
predresult.loc[predresult.sex == 'F','bias'].hist(bins=100,label='Female')

predresult.groupby('sex')['bias'].agg(np.mean)
predresult.groupby('dept')['bias'].agg(np.mean)
predresult.groupby('level')['bias'].agg(np.mean)


predresult.groupby( predresult.bias>0).apply(lambda df: df.sex.value_counts(normalize=True))


overpay_ismale = ( predresult.loc[predresult.bias > 0,"sex"] == 'M' ).astype(int)
underpay_ismale = ( predresult.loc[predresult.bias < 0,"sex"] == 'M' ).astype(int)

overpay_ismale.mean()
underpay_ismale.mean()

ss.ttest_ind(overpay_ismale,underpay_ismale,equal_var=False)


predresult_old = predresult.copy()
del predresult['ypred']
del predresult['salary']

predresult['degree_level'] = predresult.degree_level.map(degree2index)
predresult['level'] = predresult.level.map(level2index)
predresult['is_male'] = (predresult.sex == 'M').astype(int)
del predresult['sex']

dept_lb_encoder = LabelEncoder()
predresult['dept'] = dept_lb_encoder.fit_transform(predresult.dept)

scores,pvalues = chi2(predresult.loc[:,predresult.columns != 'bias'],predresult.bias > 0)


pd.Series(pvalues,index = predresult.loc[:,predresult.columns != 'bias'].columns)



param_dist = {"n_estimators": [30,50,100,200],
              "max_depth": [6, 10, 20,None],
              "min_samples_split": [2,6,10],
              "min_samples_leaf": [1,5,10]}

rfg = RandomForestRegressor(random_state=seed,verbose=1,n_jobs=-1,oob_score=True)
searchcv = RandomizedSearchCV(estimator=rfg, param_distributions=param_dist,
                              n_iter=100,n_jobs=-1,
                              random_state=seed,verbose=1)
searchcv.fit(Xtrain,ytrain)

bestrfg = searchcv.best_estimator_

"""
dept_HR             0.759710
dept_engineering    0.160756
n_subordinates      0.033970
yrs_experience      0.021435
level               0.014494
degree              0.004653
signing_bonus       0.002510
is_male             0.001915
dept_sales          0.000299
dept_marketing      0.000257
"""
rf_feat_import = pd.Series(bestrfg.feature_importances_,index = X.columns).sort_values(ascending=False)

evaluate_model(bestrfg,Xtrain,ytrain,"train")
evaluate_model(bestrfg,Xtest,ytest,'test')

##########################
lrg = LinearRegression(normalize=True )
lrg.fit(Xtrain,ytrain)

"""
dept_engineering    840.477082
dept_marketing      790.039227
dept_sales          787.553029
dept_HR             678.119787
level                 3.368485
yrs_experience        0.622068
degree                0.567698
n_subordinates        0.127758
is_male              -0.541216
signing_bonus        -2.823001
"""
pd.Series(lrg.coef_,index = Xtrain.columns).sort_values(ascending=False)

lrg.score(Xtrain,ytrain)
lrg.score(Xtest,ytest)

evaluate_model(lrg,Xtrain,ytrain,'train')
evaluate_model(lrg,Xtest,ytest,'test')

###############################
rdgcv = RidgeCV(alphas=np.logspace(-3,3,7),normalize=True)
rdgcv.fit(Xtrain,ytrain)

"""
dept_engineering    54.902454
dept_marketing       8.094899
level                6.222550
dept_sales           6.022988
is_male              1.460419
yrs_experience       0.674922
degree               0.510253
n_subordinates       0.055315
signing_bonus       -0.920191
dept_HR            -94.335435
"""
pd.Series(rdgcv.coef_,index = Xtrain.columns).sort_values(ascending=False)

rdgcv.alpha_

rdgcv.score(Xtrain,ytrain)
rdgcv.score(Xtest,ytest)

evaluate_model(rdgcv,Xtrain,ytrain,"train")
evaluate_model(rdgcv,Xtest,ytest,'test')

##############################################################
pd.concat([evaluate_model(bestrfg,Xtest,ytest,'rf_test'),
           evaluate_model(lrg,Xtest,ytest,'lrg_test'),
           evaluate_model(rdgcv,Xtest,ytest,'rdg_test')],axis=1)

ytest_pred_rf = bestrfg.predict(Xtest)
ytest_pred_lrg = lrg.predict(Xtest)
ytest_pred_rdg = rdgcv.predict(Xtest)

np.corrcoef(ytest_pred_lrg,ytest_pred_rdg)
np.corrcoef(ytest_pred_rf,ytest_pred_lrg)
np.corrcoef(ytest_pred_rf,ytest_pred_rdg)

ytest_avg = (ytest_pred_rf + ytest_pred_lrg)/2

avg_metrics = {}
avg_metrics["rmse"] = np.sqrt(mean_squared_error(ytest,ytest_avg))
avg_metrics["mabse"] = mean_absolute_error(ytest,ytest_avg)
avg_metrics["r2"] = r2_score(ytest,ytest_avg)
pd.Series(avg_metrics)


