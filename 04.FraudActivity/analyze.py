
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import  train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.feature_selection import chi2

chi2(datas.loc[:,["purchase_value","dev_shared","ip_shared","interval_after_signup"]],datas["class"])

pd.crosstab(datas.sex,datas["class"],margins=True,normalize='index')
pd.crosstab(datas.purchase_value,datas["class"],normalize='index')

temp = pd.crosstab(datas.source,datas["class"])
temp['odd'] = temp.loc[:,1]/temp.loc[:,0]
temp

grps = datas.groupby(by="class")
for key,grp in grps:
    plt.hist(grp.purchase_value,normed=False,label='class={}'.format(key),bins=50)
plt.legend(loc='best')

########################## RF
del datas["country"]

v1data = datas.copy()
src_label_encoder = LabelEncoder()
v1data["source"] = src_label_encoder.fit_transform(v1data.source)
v1data["is_male"] = (datas_v1.sex == "M").astype(int)
del v1data["sex"]

br_label_encoder = LabelEncoder()
v1data['browser'] = br_label_encoder.fit_transform(v1data.browser)

v1data['dev_shared'] = v1data['dev_shared'].astype(int)
v1data['ip_shared'] = v1data['ip_shared'].astype(int)


X = v1data.loc[:,["source","browser","age","purchase_value","interval_after_signup","is_male","dev_shared","ip_shared"]]
y = v1data.loc[:,"class"]
chi2values = chi2(X,y)

chi2values = pd.DataFrame({'feature': X.columns.values,'chi2': chi2values[0]})
chi2values.sort_values(by='chi2',inplace=True,ascending=False)
chi2values


Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3333)

rf = RandomForestClassifier()
rf.fit(Xtrain,ytrain)

ytrain_predict = rf.predict(Xtrain)
ytest_predict = rf.predict(Xtest)

accuracy_score(y_true=ytrain,y_pred=ytrain_predict)# 0.99
accuracy_score(y_true=ytest,y_pred=ytest_predict) #0.95

classification_report(y_true=ytest,y_pred=ytest_predict)
confusion_matrix(y_true=ytest,y_pred=ytest_predict)

# 'class=0' is 0.90
pd.value_counts(y,normalize=True)

feat_importances = pd.DataFrame({'feature': X.columns.values,\
                                 'importances': rf.feature_importances_})
feat_importances.sort_values(by='importances',inplace=True,ascending=False)
feat_importances














