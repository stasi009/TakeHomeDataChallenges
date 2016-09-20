
import numpy as np
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from scipy.stats import norm

dfexperiment = pd.read_csv('test_table.csv',index_col='user_id')
dfuser = pd.read_csv('user_table.csv',index_col="user_id")
dataframe = dfuser.join(dfexperiment)

df_no_spain = dataframe.loc[dataframe.country != 'Spain',:]
df_no_spain.groupby("test")[["conversion"]].mean()
#       conversion
# test
# 0       0.048292
# 1       0.043411

conv_in_test = df_no_spain.loc[dfexperiment.test==1,"conversion"]
conv_in_ctrl = df_no_spain.loc[dfexperiment.test==0,"conversion"]

ss.ttest_ind(conv_in_test,conv_in_ctrl)
# Ttest_indResult(statistic=-7.382252163053967, pvalue=1.5593292778816856e-13)