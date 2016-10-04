
import itertools
import cPickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.feature_selection import chi2,f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report,roc_curve,auc,precision_score,precision_recall_curve
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV

#########################
emails = pd.read_csv("email_table.csv",index_col="email_id")
emails["paragraphs"] = np.where(emails.email_text == 'short_email',2,4)
del emails["email_text"]

emails["is_personal"] = (emails.email_version == "personalized").astype(np.int)
del emails["email_version"]

weekday2index = {"Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4,
                 "Friday":5,"Saturday":6,"Sunday":7}
emails["weekday"] = emails.weekday.map(weekday2index)

emails.rename(columns={'user_past_purchases':'purchases','user_country':'country'},inplace=True)

######################
emails["result"] = "received"

open_users = pd.read_csv("email_opened_table.csv").email_id
emails.loc[open_users,"result"] = "opened"

click_users = pd.read_csv("link_clicked_table.csv").email_id
emails.loc[click_users,"result"] = 'clicked'

emails.to_csv("clean_emails.csv",index_label="email_id")

##################### Q1
# What percentage of users opened the email and
# what percentage clicked on the link within the email?
emails.result.value_counts(normalize=True)


#######################
rslt_lb_encoder = LabelEncoder()
cnty_lb_encoder = LabelEncoder()

X = emails.copy()
y = rslt_lb_encoder.fit_transform(X.result)
del X["result"]

feat_names = ["hour","weekday","country","purchases","paragraphs","is_personal" ]
X = X.loc[:,feat_names]
X["country"] = cnty_lb_encoder.fit_transform(X.country)

chi2scores,_ = chi2(X,y)
fscores,_ = f_classif(X,y)

feat_scores = pd.DataFrame({"chi2scores":chi2scores,"fscores":fscores},index=feat_names)
feat_scores.sort_values(by='chi2scores',ascending=False)
feat_scores.sort_values(by="fscores",ascending=False)

"""
top three features are:    purchases, country,is_personal
bottom three features are: weekday,hour,paragraphs
"""

############################################
del X
X = emails.copy()
X = X.loc[:,["country","purchases","paragraphs","is_personal"] ]
X['is_weekend'] = (emails.weekday>=5).astype(int)
X = pd.get_dummies(X,columns=["country"],drop_first=True)

y = (emails.result == 'clicked').astype(int)

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.33333)

###################################
rf = RandomForestClassifier(oob_score=True,verbose=1,n_jobs=-1)

param_dist = {"n_estimators": [30,50,100],
              "max_depth": [6, 10, None],
              "min_samples_split": [2,6,10],
              "criterion": ["gini", "entropy"],
              "min_samples_leaf": [1,3,10]}
searchcv = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                              scoring = "roc_auc",n_iter=100,n_jobs=-1,verbose=1)
searchcv.fit(Xtrain,ytrain)

searchcv.best_params_
searchcv.best_score_ # 0.73567509333112291
bestrf = searchcv.best_estimator_

ytrain_pred = bestrf.predict(Xtrain)
print classification_report(ytrain,ytrain_pred)

ytest_pred = bestrf.predict(Xtest)
print classification_report(ytest,ytest_pred)

with open("rf.pkl","wb") as outf:
    cPickle.dump(bestrf,outf)

with open("rf.pkl","rb") as inf:
    bestrf = cPickle.load(inf)

feat_importances = pd.Series(bestrf.feature_importances_,index=X.columns).sort_values(ascending=False)

###################################
ytest_pred_proba = bestrf.predict_proba(Xtest)[:,1]
fpr,tpr,thresholds = roc_curve(ytest,ytest_pred_proba)

auc(fpr,tpr)# 0.73663254679466483

roc_results = pd.DataFrame({'FPR':fpr,'TPR':tpr,'Thresholds':thresholds})

plt.plot(fpr,tpr,marker='o',markersize=3)
plt.xlabel("FPR")
plt.ylabel('TPR')
for fp,tp,threshold in itertools.izip(fpr,tpr,thresholds):
    plt.text(fp,tp,threshold)


roc_results.loc[(roc_results.TPR > 0.6) & (roc_results.TPR < 0.65),:]

"""
          FPR       TPR  Thresholds
148  0.302066  0.635724    0.026345
149  0.302097  0.635724    0.026221

240  0.304007  0.639045    0.027801
241  0.304160  0.639045    0.027677
"""
def adjust_predict(X,threshold):
    proba = bestrf.predict_proba(X)[:,1]
    return (proba >=threshold).astype(int)

threshold = 0.0278

ytrain_adjpred = adjust_predict(Xtrain,threshold)
print classification_report(ytrain,ytrain_adjpred)

ytest_adjpred = adjust_predict(Xtest,threshold)
print classification_report(ytest,ytest_adjpred)

##############################
precisions,recalls,thresholds = precision_recall_curve(ytest,ytest_pred_proba)
plt.plot(precisions,recalls,marker='o',markersize=3)
plt.xlabel("Precisions")
plt.ylabel('Recalls')

#####################################
def count_result_ratio(df):
    counts = df.result.value_counts(normalize=True)
    counts['total'] = df.shape[0]
    return counts

def grp_count_plotbar(key):
    grpresult = emails.groupby(key).apply(count_result_ratio)
    print grpresult
    grpresult.loc[:,["received","opened",'clicked']].plot(kind='bar')
    return grpresult

rslt_grpby_purchase = emails.groupby("purchases").apply(count_result_ratio).unstack()
rslt_grpby_purchase.fillna(value=0,inplace=True)
rslt_grpby_purchase.plot(marker='o',markersize=3)

grp_count_plotbar('country')
grp_count_plotbar('is_personal')
grp_count_plotbar('weekday')
grp_count_plotbar('hour')





