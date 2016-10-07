
import numpy as np
import pandas as pd
from sklearn.cross_validation import  train_test_split
from sklearn.grid_search import  RandomizedSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV

import matplotlib.pyplot as plt
plt.style.use('ggplot')

index2level = ['IC','MM',"D","VP","E","CEO"]
level2index = {l:index for index,l in enumerate(index2level)}

index2degree = ['High_School','Bachelor','Master','PhD']
degree2index = {d:index for index,d in enumerate(index2degree)}

D = pd.read_csv("preproc_employees.csv",index_col="employee_id")
D['log_salary'] = np.log(D.salary)

################################################
Dlow = D.loc[D.level ==0,:]
del Dlow['level']
del Dlow['n_subordinates']

Dmiddle = D.loc[(D.level >=1) & (D.level<=3),:]
Dhigh = D.loc[D.level>=4,:]

seed = 9999

def evaluate_model(model,X,y,tag):
    ypred = model.predict(X)
    metrics = {}
    metrics['r2'] = r2_score(y,ypred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y,ypred))
    metrics['mabse'] = mean_absolute_error(y,ypred)
    return pd.Series(metrics,name=tag)

def plot_residuals(model,X,ytrue):
    ypred = model.predict(X)
    residuals = ytrue - ypred

    plt.figure()
    ax1 = plt.subplot(2,1,1)
    ax1.scatter(ypred,residuals)
    ax1.set_xlabel("predict y")
    ax1.set_ylabel('residual')

    ax2 = plt.subplot(2,1,2)
    ax2.hist(residuals,bins=100,normed=True)
    ax2.set_xlabel('residual')

    plt.show()

##############################################################
def build_linear_model(model,tag,data,target_label = 'log_salary'):
    data = data.copy()
    y = data[target_label]
    del data['salary']
    del data['log_salary']
    X = data

    Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.33333,random_state=seed)

    model.fit(Xtrain,ytrain)
    print "training finished, best alpha={}".format(rdgcv.alpha_)

    feat_coef = pd.Series(model.coef_,index=X.columns)
    print "************** Feature Importance **************"
    feat_coef = pd.concat([feat_coef,np.abs(feat_coef)],axis=1,keys=['coef','importance'])
    feat_coef.sort_values(by='coef',ascending=False,inplace=True)
    print feat_coef

    print "************** Accuracy on test set **************"
    metrics = evaluate_model(model,Xtest,ytest,tag)
    print metrics

    # ********************* #
    plot_residuals(model,Xtest,ytest)

    return model,feat_coef,metrics


rdgcv = RidgeCV(alphas=np.logspace(-3, 3, 7), normalize=True, scoring='mean_squared_error' )
low_rdg,low_feat_import,_ = build_linear_model(rdgcv,'low',Dlow)

rdgcv = RidgeCV(alphas=np.logspace(-3, 3, 7), normalize=True, scoring='mean_squared_error' )
middle_rdg, middle_feat_import,_ = build_linear_model(rdgcv,'middle',Dmiddle)

low_feat_import.sort_values(by='importance',ascending=False)
middle_feat_import.sort_values(by='importance',ascending=False)


################################
lacv = LassoCV(normalize=True,random_state=seed,verbose=True,n_jobs=-1)
_,low_feat_import,_ = build_linear_model(lacv,'low',Dlow)
_, middle_feat_import,_ = build_linear_model(lacv,'middle',Dmiddle)


###################################
Dmiddle.groupby('is_male')["salary"].agg('mean')
Dlow.groupby('is_male')['salary'].agg('mean')

Dmiddle.groupby('is_male').apply(lambda df: df.level.value_counts())

#################################################
def build_random_forest(tag,data,target_label = 'log_salary'):
    data = data.copy()
    y = data[target_label]
    del data['salary']
    del data['log_salary']
    X = data

    Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.33333,random_state=seed)

    model = RandomForestRegressor(n_estimators=150,
                                  n_jobs=-1,random_state=seed,verbose=1)
    model.fit(Xtrain,ytrain)

    feat_coef = pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)
    print "************** Feature Importance **************"
    print feat_coef

    print "************** Accuracy on test set **************"
    metrics = evaluate_model(model,Xtest,ytest,tag)
    print metrics

    #####
    plot_residuals(model,Xtest,ytest)

    return model,feat_coef,metrics

_ = build_random_forest('low',Dlow)
_ = build_random_forest('middle',Dmiddle)