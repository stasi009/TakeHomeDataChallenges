
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import roc_curve
import xgboost as xgb

seed = 999

#############################
loan = pd.read_csv("loan_table.csv",index_col='loan_id')
loan.rename(columns={'loan_purpose':'purpose',
                     'loan_granted':'granted',
                     'loan_repaid':'repaid'},inplace=True)
loan['date'] = pd.to_datetime(loan.date)

borrower = pd.read_csv("borrower_table.csv",index_col='loan_id')
borrower.rename(columns={'is_first_loan':'is_first',
                         'fully_repaid_previous_loans':'repay_prev',
                         'currently_repaying_other_loans':'paying_others',
                         'total_credit_card_limit':'credit_limit',
                         'avg_percentage_credit_card_limit_used_last_year':'credit_used',
                         'saving_amount':'saving',
                         'checking_amount':'checking',
                         'yearly_salary':'salary',
                         'dependent_number':'n_depends'},inplace=True)

loan = borrower.join(loan)

###############################
def profit_and_should_grant(row):
    """
    :param row:
    :return: series with 'profit' and 'should_grant'
    """
    granted = row['granted']
    repaid = row['repaid']

    profit = 0
    should_grant = 0

    if granted == 0:
        profit = 0
        should_grant = 0
    else: # granted == 1
        assert pd.notnull(repaid)
        profit = 1 if repaid == 1 else -1
        should_grant = 1 if repaid == 1 else 0

    return pd.Series({'profit':profit,'should_grant':should_grant})

profits = loan.apply(profit_and_should_grant,axis=1)
print 'done'

loan = loan.join(profits)
loan.to_csv("loan_all.csv",index_label='loan_id')

##############################
features = [u'is_first', u'repay_prev', u'paying_others', u'credit_limit',
            u'credit_used', u'saving', u'checking', u'is_employed', u'salary',
            u'age', u'n_depends', u'purpose']
X = loan.loc[:,features]
y = loan['should_grant']

X.fillna({'repay_prev':-1,'paying_others':-1},inplace=True)
del X['is_first'] # redundant after filling missing in 'repay_prev' and 'paying_others'

del X['is_employed'] # redudant, since 'not employed' has 'salary=0'

X = pd.get_dummies(X)
del X['purpose_other'] # redudant
X.rename(columns={'purpose_emergency_funds':'purpose_emergency'},inplace=True)

(X.join(loan.loc[:,[u'date', u'granted', u'repaid',u'profit', u'should_grant']])).to_csv('cleaned_loan.csv',index_label='loan_id')

######################
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,random_state=seed)
Xtrain,Xvalid,ytrain,yvalid =  train_test_split(Xtrain,ytrain,test_size=0.3,random_state=seed)

train_matrix = xgb.DMatrix(Xtrain,ytrain)
valid_matrix = xgb.DMatrix(Xvalid,yvalid)
test_matrix = xgb.DMatrix(Xtest,ytest)

###############################
def train(params):
    params['silent'] = 1
    params['objective'] = 'binary:logistic'  # output probabilities
    params['eval_metric'] = 'auc'

    num_rounds = params["num_rounds"]
    early_stopping_rounds = params["early_stop_rounds"]

    # early stop will check on the last dataset
    watchlist = [(train_matrix, 'train'), (valid_matrix, 'validate')]
    bst = xgb.train(params, train_matrix, num_rounds, watchlist, early_stopping_rounds=early_stopping_rounds)

    print "parameters: {}".format(params)
    print "best {}: {:.2f}".format(params["eval_metric"], bst.best_score)
    print "best_iteration: %d" % (bst.best_n_trees)

    return bst

params = {}
params["num_rounds"] = 300
params["early_stop_rounds"] = 30
# params['min_child_weight'] = 2

params['max_depth'] = 6
params['eta'] = 0.1
params["subsample"] = 0.8
params["colsample_bytree"] = 0.8

bst = train(params)

######################
yvalid_true = valid_matrix.get_label()
yvalid_probas = bst.predict(valid_matrix, ntree_limit=bst.best_iteration)

fpr,tpr,thresholds = roc_curve(yvalid_true,yvalid_probas)
roc = pd.DataFrame({'FPR':fpr,'TPR':tpr,'Thresholds':thresholds})

plt.plot(roc.FPR,roc.TPR)
plt.xlabel("FPR")
plt.ylabel('TPR')
ticks = np.linspace(0,1,11)
plt.yticks(ticks)
plt.xticks(ticks)

def calc_profits(repaids,probas,threshold):
    total_profit = 0
    for (repaid,proba) in itertools.izip(repaids,probas):
        if proba > threshold:
            total_profit += (1 if repaid == 1 else -1)
    return total_profit

loan_valid = loan.loc[yvalid.index,:]
valid_profits = [ calc_profits(loan_valid.repaid,yvalid_probas,threshold) for threshold in thresholds]
print 'done'

valid_threshold_profits = pd.DataFrame({'thresholds':thresholds,'profit':valid_profits})
plt.plot(thresholds,valid_profits)


valid_threshold_profits.loc[  valid_threshold_profits.profit.argmax()   ,:]

roc.loc[ (roc.FPR > 0.199999) & (roc.FPR < 0.201),:]
proba_threshold = 0.413#0.2812

###########################################
Xalltrain = pd.concat([Xtrain,Xvalid],axis=0)
yalltrain = pd.concat([ytrain,yvalid],axis=0)

alltrain_matrix = xgb.DMatrix(Xalltrain,yalltrain)

watchlist = [(alltrain_matrix, 'train')]
all_bst = xgb.train(params, alltrain_matrix, bst.best_iteration,watchlist)

###############################
ytest_probas = bst.predict(test_matrix, ntree_limit=bst.best_iteration)
ytest_pred = (ytest_probas > proba_threshold).astype(int)

print classification_report(ytest,ytest_pred)
accuracy_score(ytest,ytest_pred)

#######################
loan_test = loan.loc[ytest.index,:]

# 4130
old_profit = loan_test.profit.sum()

# 4378
new_profit = calc_profits(loan_test.repaid,ytest_probas,proba_threshold)

xgb.plot_importance(all_bst,title='Feature Importance')









