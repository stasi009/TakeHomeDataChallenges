
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2,f_classif
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix,classification_report

import matplotlib.pyplot as plt
plt.style.use("ggplot")

random_state = 999

testdata = pd.read_csv("test_results.csv",index_col="user_id")
users = pd.read_csv("user_table.csv")

testdata["timestamp"] = pd.to_datetime(testdata.timestamp)

revenues = testdata.groupby(by="test").apply(lambda df: df.price * df.converted)
ctrl_revenues = revenues[0]
test_revenues = revenues[1]

test_revenues.mean() # 0.9168
ctrl_revenues.mean() # 0.7767

ttest_result = ss.ttest_ind(test_revenues,ctrl_revenues,equal_var=False)
# ttest_ind is a two tailed
# since our HA is test_mean > ctrl_mean, so we need to divide by 2
ttest_result.pvalue/2 # 7.7037493023391909e-09

#########################
src_label_encoder = LabelEncoder()
dev_label_encoder = LabelEncoder()
os_label_encoder = LabelEncoder()

testdata["source"] = src_label_encoder.fit_transform(testdata.source)
testdata["device"] = dev_label_encoder.fit_transform(testdata.device)
testdata["operative_system"] = os_label_encoder.fit_transform(testdata.operative_system)
testdata.rename(columns = {'operative_system':'OS'},inplace=True)
del testdata["timestamp"]

#########################
# check whether the splitting is random or not
colnames = ["source","device","OS"]
X4split = testdata.loc[:,colnames]
y4split = testdata["test"]

ch2values,pvalues = chi2(X4split,y4split)
feat_chi2_pvalues = pd.Series(pvalues,index=colnames)

fvalues, pvalues = f_classif(X4split,y4split)
feat_f_pvalues = pd.Series(pvalues,index=colnames)

pvalues = pd.concat([feat_chi2_pvalues,feat_f_pvalues],axis=1)
pvalues.columns = ["chi2_pvalues","f_pvalues"]

# check distribution in test/control groups
dev_in_test = testdata.loc[testdata.test == 1,"device"]
dev_in_ctrl = testdata.loc[testdata.test == 0,"device"]
# pvalue is nearly zero, so device's distribution in two group aren't same
ss.ttest_ind(dev_in_ctrl,dev_in_test,equal_var=False)
ss.ttest_ind(dev_in_ctrl,dev_in_test)

def value_freq(df,colname,lbencoder):
    freq = df[colname].value_counts(normalize=True)
    freq.index = lbencoder.inverse_transform(freq.index)
    return freq

testdata.groupby(by="test").apply(lambda df: value_freq(df,"device",dev_label_encoder)).transpose()
testdata.groupby(by="test").apply(lambda df: value_freq(df,"OS",os_label_encoder)).transpose()

### select best features
selector = SelectKBest(score_func=f_classif,k="all")
selector.fit(X4split,y4split)

#########################
del testdata["timestamp"]
ohe = OneHotEncoder(categorical_features = [0,1,2])
ohe.fit_transform(testdata)


#################
labels = ["source","device","OS","test","price"]
X4converted = testdata.loc[:,labels]
y4converted = testdata.converted

result = chi2(X4converted,y4converted)
chi2result = pd.DataFrame({'chi2v':result[0],'chi2p':result[1]},index=labels)
chi2result.sort_values(by="chi2p",inplace=True)

result = f_classif(X4converted,y4converted)
fresult = pd.DataFrame({'fv':result[0],'fp':result[1]},index=labels).sort_values(by="fp")
converted_factors = chi2result.join(fresult)


#################
testdata = pd.read_csv("test_results.csv",index_col="user_id")
del testdata["timestamp"]
testdata.rename(columns = {'operative_system':'OS'},inplace=True)

def avg_converted_revenue(df):
    avg_convert_rate = np.mean( df.converted)
    avg_revenue = np.mean(df.price * df.converted)
    return pd.Series({'avg_convert_rate':avg_convert_rate,"avg_revenue":avg_revenue})
testdata.groupby(by="test").apply(avg_converted_revenue)

testdata_ohe = pd.get_dummies(testdata)

feat_labels = ['price',u'source_ads-bing',u'source_ads-google', u'source_ads-yahoo', u'source_ads_facebook',
               u'source_ads_other', u'source_direct_traffic',u'source_friend_referral',
               u'source_seo-bing', u'source_seo-google',u'source_seo-other', u'source_seo-yahoo',
               u'source_seo_facebook',u'device_mobile', u'device_web', u'OS_android',
               u'OS_iOS', u'OS_linux',u'OS_mac', u'OS_other', u'OS_windows']
X = testdata_ohe.loc[:,feat_labels]
y = testdata_ohe.converted

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.333,random_state=random_state)

tree = DecisionTreeClassifier()
tree.fit(Xtrain,ytrain)

ytrain_predict = tree.predict(Xtrain)
ytest_predict = tree.predict(Xtest)

print classification_report(y_true=ytrain,y_pred=ytrain_predict)
classification_report(y_true=ytest,y_pred=ytest_predict)

feat_importances = pd.Series(tree.feature_importances_,index=feat_labels).sort_values(ascending=False)

with open("whole_tree.dot",mode='wb') as outf:
    export_graphviz(tree,out_file=outf,feature_names=feat_labels)


###############
rf = RandomForestClassifier(n_estimators=30, oob_score=True)
rf.fit(Xtrain,ytrain)

ytrain_predict = tree.predict(Xtrain)
ytest_predict = tree.predict(Xtest)

print classification_report(y_true=ytrain,y_pred=ytrain_predict)

rf_feat_importances = pd.Series(rf.feature_importances_,index=feat_labels).sort_values(ascending=False)






lrcv = LogisticRegressionCV(Cs=[0.001,0.01,0.1,1,10,100],cv=5,scoring='roc_auc')
lrcv.fit(Xtrain,ytrain)

ytrain_predict = lrcv.predict(Xtrain)
ytest_predict = lrcv.predict(Xtest)

print classification_report(y_true=ytrain,y_pred=ytrain_predict)
print classification_report(y_true=ytest,y_pred=ytest_predict)

lr_feat_importances = pd.Series(lrcv.coef_[0],index=feat_labels).sort_values(ascending=False)

feat_importances = pd.concat([rf_feat_importances,lr_feat_importances],axis=1,keys = ['rf','lr'])
feat_importances.sort_values(by='rf',inplace=True,ascending=False)

feat_importances['abs_lr'] = np.abs(feat_importances.lr)


# ===============================================
feat_labels = [u'source_ads-bing',u'source_ads-google', u'source_ads-yahoo', u'source_ads_facebook',
               u'source_ads_other', u'source_direct_traffic',u'source_friend_referral',
               u'source_seo-bing', u'source_seo-google',u'source_seo-other', u'source_seo-yahoo',
               u'source_seo_facebook',u'device_mobile', u'device_web', u'OS_android',
               u'OS_iOS', u'OS_linux',u'OS_mac', u'OS_other', u'OS_windows']


X_intest = testdata_ohe.loc[testdata_ohe.test == 1,feat_labels]
y_intest = testdata_ohe.loc[testdata_ohe.test == 1,"converted"]

X_inctrl = testdata_ohe.loc[testdata_ohe.test == 0,feat_labels]
y_inctrl = testdata_ohe.loc[testdata_ohe.test == 0,"converted"]

temp = chi2(X_intest,y_intest)
temp = chi2(X_inctrl,y_inctrl)

pd.DataFrame({'chi2v':temp[0],'chi2p':temp[1]},index=feat_labels).sort_values(by='chi2p')


##################
top_feat_rf = feat_importances.rf.sort_values(ascending=False)[:5]
top_feat_lr = feat_importances.sort_values(by='abs_lr',ascending=False).iloc[:5,[1,2]]

top_feat_lr.join(top_feat_rf).dropna()

########################
lrcv = LogisticRegressionCV(penalty='l1',solver='liblinear',Cs=[0.001,0.01,0.1,1,10,100],cv=5,scoring='roc_auc')
lrcv.fit(Xtrain,ytrain)








