
from datetime import datetime
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans

holders = pd.read_csv('cc_info.csv',index_col='credit_card')
holders.rename(columns={'credit_card_limit':'credit_limit'},inplace=True)

transactions = pd.read_csv('transactions.csv')
transactions['date'] = pd.to_datetime(transactions.date)
transactions.rename(columns={'transaction_dollar_amount':'amount'},inplace=True)

plt.scatter(transactions.Long,transactions.Lat)

############################# questions 1
"""
identify those users that in your dataset never went above the monthly credit card limit
(calendar month).
"""
def monthly_spent_byuser(df):
    # I have checked, all transactions happen in year 2015
    # so I can just group by month
    return df.groupby(df.date.dt.month)['amount'].agg('sum')

card_month_spents = transactions.groupby("credit_card").apply(monthly_spent_byuser).unstack(fill_value=0)
card_month_spents = card_month_spents.join(holders.credit_limit)

n_months = card_month_spents.shape[1]-1
def is_never_above_limit(s):
    limit = s.loc['credit_limit']
    return (s.iloc[0:n_months] <= limit).all()

card_month_spents.loc[ card_month_spents.apply(is_never_above_limit,axis=1),:].index

###################### question 2
class MonthSpentMonitor(object):

    def __init__(self,credit_limits):
        self.total_spent = defaultdict(float)
        # card_limits is a dictionary
        # key=cardno, value=limit
        self.credit_limits = credit_limits

    def reset(self):
        self.total_spent.clear()

    def count(self,daily_transaction):
        for cardno,amount in daily_transaction:
            self.total_spent[cardno] += amount

        # assume credit_limits always can find the cardno
        return [ cardno for cardno,total in self.total_spent.viewitems() if total > self.credit_limits[cardno]]

###################### question 3
def statistics_by_card(s):
    ps = [25, 50, 75]
    d = np.percentile(s,ps)
    d = pd.Series(d,index=['{}%'.format(p) for p in ps])
    d['n_use'] = len(s)
    return d

tran_statistics = transactions.groupby('credit_card')['amount'].apply(statistics_by_card).unstack()

temp = pd.merge(transactions,tran_statistics,how='left',left_on='credit_card',right_index=True)
transactions = pd.merge(temp,holders.loc[:,['credit_limit']],how='left',left_on='credit_card',right_index=True)
transactions['hour'] = transactions.date.dt.hour

transactions.to_csv('extend_transactions.csv',index=False)


################################################
X = transactions.loc[:,['amount','25%','50%','75%','credit_limit']]
X = scale(X)

pca = PCA(n_components=2)
X2d = pca.fit_transform(X)
X2d = pd.DataFrame(X2d,columns=['pc1','pc2'])
plt.scatter(X2d[:,0],X2d[:,1])

############################### cluster with kmeans
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters,n_jobs=-1)
kmeans.fit(X)


X2d['label'] = kmeans.labels_
print X2d.label.value_counts()

colors = itertools.cycle( ['r','g','b','c','m','y','k'] )
for label in  xrange(n_clusters) :
    temp = X2d.loc[X2d.label == label,:]
    plt.scatter(temp.pc1,temp.pc2,c=next(colors),label=label,alpha=0.3)

plt.legend(loc='best')

#############################
suspect = transactions.loc[X2d.label==3,['credit_card','amount','25%','50%','75%','credit_limit','date']]
suspect.to_csv('suspect.csv',index=False)