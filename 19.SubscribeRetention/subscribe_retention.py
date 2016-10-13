
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.linear_model import LinearRegression,Ridge

subscriptions = pd.read_csv("subscription.csv",index_col='user_id')
del subscriptions['subscription_signup_date']
subscriptions.rename(columns={'subscription_monthly_cost':'monthly_cost',
                              'billing_cycles':'bill_cycles'},inplace=True)

count_by_cost = subscriptions.groupby('monthly_cost').apply(lambda df: df.bill_cycles.value_counts()).unstack()

total_by_cost = count_by_cost.apply(lambda s: s.iloc[::-1].cumsum().iloc[::-1],axis=1).transpose()

##############
def make_time_features(t):
    return pd.DataFrame({'t': t,'logt': np.log(t),'tsquare':t*t },index = t)

def fit_linear_regression(s,alpha):
    X = make_time_features(s.index)
    return Ridge(alpha=alpha).fit(X,np.log(s))

lr_by_cost = total_by_cost.apply(fit_linear_regression,alpha=0.005,axis=0)



###################
allt = np.arange(1,13)
Xwhole = make_time_features(allt)

predicts = lr_by_cost.apply(lambda lr: pd.Series(lr.predict(Xwhole),index=allt)).transpose()


combined = pd.merge(total_by_cost,predicts,how='right',left_index=True,right_index=True,suffixes = ('_true','_pred'))

combined.plot(marker='*')

#######################################
def calc_retend_rate(s):
    r = s.iloc[::-1].cumsum().iloc[::-1]
    return r/r.iloc[0]

def retend_rate_by(colname):
    counts = subscriptions.groupby(colname).apply(lambda df: df.bill_cycles.value_counts()).unstack()
    return counts.apply(calc_retend_rate, axis=1).transpose()

retend_rate_by('country').plot(marker='o')
retend_rate_by('source').plot(marker='*')























