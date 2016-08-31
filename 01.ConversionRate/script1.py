
import cPickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix

def simple_dump(filename, *objects):
    """ dump single object into binary file """
    with open(filename, 'wb') as outfile:
        for obj in objects:
            cPickle.dump(obj, outfile)

def simple_load(filename,n_objs):
    """ load single object from binary file """
    objects = []
    with open(filename, "rb") as infile:
        for index in xrange(n_objs):
            objects.append(cPickle.load(infile))
        return objects

inputfilename = "conversion_data.csv"
dataframe = pd.read_csv(inputfilename)

# from ages, I saw two samples witch ages > 100
# I consider those two samples as outlier, and remove them from future analysis
dataframe = dataframe.loc[dataframe.age <=90,:]

X = dataframe.loc[:,('country', 'age', 'new_user', 'source', 'total_pages_visited')]
# OHE those categorical features
X = pd.get_dummies(X)

y = dataframe.converted
y.mean()# =0.032, only 3.2% converted


Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.333)
# ytrain.mean() ~= ytest.mean(), indicates the split is unbiased
ytrain.mean()# 0,0324
ytest.mean()# 0.0320

dt = DecisionTreeClassifier()
params = {
    "criterion":['gini','entropy'],
    "max_depth": [None,5,10],
    "min_samples_split": [2,10,20],
    "min_samples_leaf": [5,10,20]
}
searchcv = GridSearchCV(estimator=dt,
                        param_grid =params,
                        scoring="roc_auc",
                        n_jobs=-1,
                        verbose=1)
searchcv.fit(Xtrain,ytrain)

dt = searchcv.best_estimator_
simple_dump("dt.pkl",dt)

dt.score(Xtrain,ytrain)# 0.981
1 - ytrain.mean() # 0.968

dt.score(Xtest,ytest) # 0.985
1 - ytest.mean() # 0.967

featnames = Xtrain.columns.values
featimportances = dt.feature_importances_
feat_importances = pd.DataFrame({"name":featnames,"importances":featimportances})
feat_importances = feat_importances[['name','importances']]# reorder the columns
#                   name  importances
# 0                  age     0.003461
# 1             new_user     0.050609
# 2  total_pages_visited     0.909182
# 3        country_China     0.036748
# 4      country_Germany     0.000000
# 5           country_UK     0.000000
# 6           country_US     0.000000
# 7           source_Ads     0.000000
# 8        source_Direct     0.000000
# 9           source_Seo     0.000000
feat_importances.sort_values(by="importances",inplace=True,ascending=False)


############## use LR to detect feature importances
lrcv = LogisticRegressionCV(Cs = np.logspace(-3,3,7),
                            dual=False,
                            scoring='roc_auc',
                            max_iter=1000,
                            n_jobs=-1,
                            verbose=1)
lrcv.fit(Xtrain,ytrain)
lrcv.C_ # 10

ytest_predict = lrcv.predict(Xtest)
print classification_report(y_true=ytest,y_pred=ytest_predict)

ytest_proba = lrcv.predict_proba(Xtest)

feat_importances = pd.DataFrame({"name":featnames,"coef":lrcv.coef_[0]})
feat_importances = feat_importances[['name','coef']]# reorder the columns
feat_importances['importances'] = np.abs( feat_importances['coef'] )

simple_dump('lr.pkl',lrcv)





