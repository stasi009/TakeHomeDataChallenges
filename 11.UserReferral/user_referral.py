
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

referral.country.value_counts()

def count_spent(df):
    d = {}
    d['n_purchase'] = df.shape[0]
    d['total_spent'] = df.money_spent.sum()
    d['n_customer'] = df.user_id.unique().shape[0]
    return pd.Series(d)


def daily_statistics(df):
    grpby_day = df.groupby('date').apply(count_spent)

    grpby_day_before = grpby_day.loc[grpby_day.index < dt_referral_starts, :]
    grpby_day_after = grpby_day.loc[grpby_day.index >= dt_referral_starts, :]

    d = []
    colnames = ['total_spent','n_purchase','n_customer']
    for col in colnames:
        pre_data = grpby_day_before.loc[:,col]
        pre_mean = pre_data.mean()

        post_data = grpby_day_after.loc[:,col]
        post_mean = post_data.mean()

        result = ss.ttest_ind(pre_data, post_data, equal_var=False)
        pvalue = result.pvalue / 2

        d.append({'mean_pre':pre_mean,'mean_post':post_mean,'mean_diff':post_mean - pre_mean,'pvalue':pvalue})

    # re-order the columns
    return pd.DataFrame(d,index = colnames).loc[:,['mean_pre','mean_post','mean_diff','pvalue']]


referral.groupby('country').apply(daily_statistics)


###################
fig, axes = plt.subplots(3, 1, sharex=True)
colors = ['r','g','b']
for index,col in enumerate(['total_spent','n_purchase','n_customer']):
    data =  grpby_day.loc[:,col]
    data.plot(kind='bar',ax=axes[index],color=colors[index])
    axes[index].set_title(col)
