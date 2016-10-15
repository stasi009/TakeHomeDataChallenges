
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.ensemble import  RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier,export_graphviz

vcounts = pd.read_csv("video_count.csv")
vcounts['date'] = pd.to_datetime(vcounts.date)
vcounts.rename(columns={'count':'view_count'},inplace=True)

vfeatures = pd.read_csv("video_features.csv",index_col='video_id')
vfeatures.rename(columns={'video_length':"length",
                          'video_language':"language",
                          'video_upload_date':'upload_date',
                          'video_quality':'quality'},inplace=True)
vfeatures['upload_date'] = pd.to_datetime(vfeatures.upload_date)

def extract_from_counts(df):
    counts = df.sort_values(by='date')['view_count'].astype(np.float)
    qs = [25,50,75]

    cnt_percentiles = np.percentile(counts,qs)
    d = {'cnt_{}th'.format(q):p for q,p in itertools.izip(qs,cnt_percentiles)}
    d['cnt_mean'] = counts.mean()

    # change rate = 'current view counts'/'previous view counts'
    cnts_prev = counts.iloc[:-1]
    cnts_current = counts.iloc[1:]
    # divide by values, not match by index
    change_rates = cnts_current.values / cnts_prev.values

    chg_percentiles = np.percentile(change_rates,qs)
    for q,p in itertools.izip(qs,chg_percentiles):
        # 'rch' stands for 'rate of change'
        d['rch_{}th'.format(q)] = p
    d['rch_mean'] = change_rates.mean()

    return pd.Series(d)

vstatistics = vcounts.groupby('video_id').apply(extract_from_counts)

################################
plt.hist(vstatistics.cnt_mean,bins=100,normed=True)

cnt_bins = range(0,4500000,500000)
binned_cnt_mean = pd.cut(vstatistics.cnt_mean,cnt_bins)

plt.hist(vstatistics.rch_mean,bins=100,normed=True)
plt.xticks(np.arange(0.7,1.6,0.05))

##########################
cnt_cutoff = 1000000
vstatistics['is_popular'] = vstatistics.cnt_mean >= cnt_cutoff

##############
rch_bins = [0,0.95,1.05,100]
vstatistics['trend_status'] = pd.cut(vstatistics.rch_mean,rch_bins,right=False,labels=['decrease','flat','increase'])

##############################
videos = vfeatures
videos = videos.join(vstatistics.loc[:,['trend_status','is_popular']])

#######################
vcounts = pd.merge(vcounts,vfeatures.loc[:,["upload_date"]],left_on='video_id',right_index=True)

#################
videos['quality'] = videos.quality.map(lambda s: int(s[:-1]))
videos['upload_weekday'] = videos.upload_date.dt.weekday_name
del videos['upload_date']

################################
videos.rename(columns={'is_hot':'is_popular'},inplace=True)
videos['is_hot'] = (videos.trend_status == 'increase').astype(int)

##########################
weekday_hot = videos.groupby(by='is_hot').apply(lambda df: df.upload_weekday.value_counts(normalize=True)).unstack()
weekday_hot.plot(kind='bar')
weekday_hot.transpose().plot(kind='bar')

#########################################
videos = pd.get_dummies(videos,columns=['language','upload_weekday'],prefix=['l','w'])
del videos['l_Other']

##############################
feat_names = [u'length', u'quality',
              u'l_Cn', u'l_De', u'l_En', u'l_Es', u'l_Fr',
              u'w_Friday', u'w_Monday',u'w_Saturday', u'w_Sunday', u'w_Thursday', u'w_Tuesday',u'w_Wednesday']
X = videos.loc[:,feat_names]
y = (videos.trend_status == 'increase').astype(int)

rf = RandomForestClassifier()
rf.fit(X,y)

feat_importances = pd.Series( rf.feature_importances_, index=feat_names)
feat_importances.sort_values(ascending=False,inplace=True)

######################################
hotX = X.loc[y==1,:]
nothotX = X.loc[y==0,:]

plt.hist(hotX.length,label='hot')
plt.hist(nothotX.length,label='nothot')

fig,axes = plt.subplots(2,1,sharex=True,sharey=True)
hotX.length.hist(ax=axes[0],bins=50,normed=True)
nothotX.length.hist(ax=axes[1],bins=50,normed=True)

fig,axes = plt.subplots(2,1,sharex=True,sharey=True)
hotX.quality.hist(ax=axes[0],bins=50,normed=True,label='hot')
nothotX.quality.hist(ax=axes[1],bins=50,normed=True,label='nothot')

##################################
dt = DecisionTreeClassifier(max_depth=3,min_samples_leaf=20,min_samples_split=20)
dt.fit(X,y)
export_graphviz(dt,feature_names=feat_names,class_names=['NotHot','Hot'],
                proportion=True,leaves_parallel=True,filled=True)

feat_importances = pd.Series( dt.feature_importances_, index=feat_names)
feat_importances.sort_values(ascending=False,inplace=True)