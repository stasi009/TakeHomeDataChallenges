
import numpy as np
import pandas as pd
import scipy.stats as ss

######################### load data
tests = pd.read_csv("test_table.csv",index_col='user_id')
users = pd.read_csv("user_table.csv",index_col='user_id')

tests = tests.join(users)
tests['date'] = pd.to_datetime(tests.date)
tests['signup_date'] = pd.to_datetime(tests.signup_date)

########################## t-test
pages_in_test = tests.loc[tests.test==1,'pages_visited']
pages_in_ctrl = tests.loc[tests.test==0,'pages_visited']

pd.concat([pages_in_ctrl.describe(),pages_in_test.describe()],keys=['CTRL','TEST'],axis=1)

# Ttest_indResult(statistic=0.55711184355547971, pvalue=0.57745231715591183)
ss.ttest_ind(pages_in_ctrl,pages_in_test,equal_var=False)

#############################
tests['sign_days'] = (tests.date - tests.signup_date).dt.days
tests['new_user'] = (tests.sign_days == 0).astype(int)

#############################
def run_ttest(df):
    vp_in_test = df.loc[tests.test == 1, 'pages_visited']
    vp_in_ctrl = df.loc[tests.test == 0, 'pages_visited']
    result = ss.ttest_ind(vp_in_ctrl, vp_in_test, equal_var=False)
    conclusion = 'Significant' if result.pvalue < 0.05 else 'Not Significant'
    return pd.Series({'n_test':vp_in_test.shape[0],
                      'n_ctrl': vp_in_ctrl.shape[0],
                      'mean_test': vp_in_test.mean(),
                      'mean_ctrl': vp_in_ctrl.mean(),
                      'mean_diff': vp_in_test.mean() - vp_in_ctrl.mean(),
                      'pvalue':result.pvalue,
                      'conclusion':conclusion})

############### t-test by group
tests.groupby('new_user').apply(run_ttest)
tests.groupby('browser').apply(run_ttest)
tests.groupby(by=['browser','new_user']).apply(run_ttest)

tests.groupby('test')['browser'].apply(lambda df:df.value_counts(normalize=True))

##############
pd.crosstab(tests.new_user,tests.browser,margins=True)



