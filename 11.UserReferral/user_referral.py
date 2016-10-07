
import datetime
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
plt.style.use('ggplot')

referral = pd.read_csv("referral.csv")
del referral['device_id']
referral['date'] = pd.to_datetime( referral.date )

dt_referral_starts = pd.to_datetime('2015-10-31')

# pre_referral = referral.loc[referral.date < dt_referral_starts,:]
# post_referral = referral.loc[referral.date >= dt_referral_starts,:]

def count_spent(df):
    d = {}
    d['n_purchase'] = df.shape[0]
    d['total_spent'] = df.money_spent.sum()
    d['n_customer'] = df.user_id.unique().shape[0]
    return pd.Series(d)

grpby_day = referral.groupby('date').apply(count_spent)

###################
fig, axes = plt.subplots(3, 1, sharex=True)
colors = ['r','g','b']
for index,col in enumerate(['total_spent','n_purchase','n_customer']):
    data =  grpby_day.loc[:,col]
    data.plot(kind='bar',ax=axes[index],color=colors[index])
    axes[index].set_title(col)

########################### t-test to test mean difference
grpby_day_before = grpby_day.loc[grpby_day.index < dt_referral_starts,:]
grpby_day_after = grpby_day.loc[grpby_day.index >= dt_referral_starts,:]

"""
n_customer      1384.464286
n_purchase      1690.750000
total_spent    71657.000000
"""
grpby_day_before.mean(axis=0)

"""
n_customer      1686.964286
n_purchase      1785.714286
total_spent    83714.392857
"""
grpby_day_after.mean(axis=0)

ss.ttest_ind(grpby_day_before,grpby_day_after)

ss.ttest_ind(grpby_day_before.total_spent,grpby_day_after.total_spent,equal_var=False)
