
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import scipy.linalg as slin
from sklearn.linear_model import LinearRegression

ads = pd.read_csv("ad_table.csv")
ads['date'] = pd.to_datetime(ads.date)
ads.rename(columns={"avg_cost_per_click":"cost",'total_revenue':'revenue'},inplace=True)

################################
X = ads.loc[:,['ad','date','shown']]
firstday = X.date.min()
X['days'] = X.date.map(lambda dt: (dt - firstday).days)
X['weekday'] = X.date.map(lambda dt: dt.weekday_name)

X = pd.get_dummies(X,columns=['weekday'],prefix='',prefix_sep='')
del X['Sunday']

#######################
time_features = [u'days', u'Friday', u'Monday', u'Saturday',u'Thursday', u'Tuesday', u'Wednesday']
def fit_linear_regression(df):
    features = df.loc[:,time_features]
    target = df['shown']

    lr = LinearRegression()
    lr.fit(features,target)

    return lr

lrmodels = X.groupby(by='ad').apply(fit_linear_regression)

##########################
adname = 'ad_group_15'
data = X.loc[X.ad == adname,time_features]
ytrue = X.loc[X.ad == adname,'shown']

model = lrmodels.loc[adname]
ypred = model.predict(data)

plt.plot(data.days,ytrue,marker='o')
plt.plot(data.days,ypred,marker='*')


###############################
ads['net_revenue'] = ads.apply(lambda s: s['revenue'] - s['cost'] * s['clicked'],axis=1)

def avg_net_revenue_per_show(df):
    total_net_revenue = df.net_revenue.sum()
    total_shown = df.shown.sum()
    return total_net_revenue / total_shown

net_revenue_per_show_grps = ads.groupby("ad").apply(avg_net_revenue_per_show).sort_values(ascending=False)

###################
def __statistics(values,suffix,d):
    d['mean_{}'.format(suffix)] = values.mean()

    qs = [25, 50, 75]
    percentiles = np.percentile(values, qs)
    for q, p in itertools.izip(qs, percentiles):
        d['{}p_{}'.format(q,suffix)] = p

def statistics_changes(df):
    costs = df.sort_values(by='date')['cost'].values
    prev_cost = costs[:-1]
    curr_cost = costs[1:]

    d = {}
    __statistics(curr_cost / prev_cost,'chrate',d)
    __statistics(curr_cost - prev_cost,'abschg',d)

    return pd.Series(d)

change_statistics = ads.groupby('ad').apply(statistics_changes)

change_statistics['mean_abschg'].hist(bins=50)

change_statistics.loc[:,[u'25p_abschg', u'25p_chrate', u'50p_abschg', u'50p_chrate',u'75p_abschg', u'75p_chrate', u'mean_abschg']].hist(bins=50)


