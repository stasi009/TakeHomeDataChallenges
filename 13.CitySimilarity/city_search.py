
from collections import Counter
import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors

############################################# load and clean data
with open("city_search.json",'rt') as inf:
    sessions = json.load(inf)

def clean_json(d):
    assert len(d['cities']) == 1
    d['cities'] = d['cities'][0]

    assert len(d['session_id']) == 1
    d['session_id'] = d['session_id'][0]

    assert len(d['unix_timestamp']) == 1
    d['timestamp'] = datetime.datetime.utcfromtimestamp(d['unix_timestamp'][0])
    del d['unix_timestamp']

    # -------- retrieve users
    user_dict = d['user']

    assert len(user_dict) == 1
    user_dict = user_dict[0]

    assert len(user_dict) == 1
    user_dict = user_dict[0]

    d['user_id'] = user_dict['user_id']
    d['user_country'] = user_dict['country']

    del d['user']
    return d

for d in sessions:
    clean_json(d)

sessions = pd.DataFrame(sessions)
sessions = sessions.set_index('session_id')

# empty string will be treated as NA when read back
# and some function will dropna=True by default
# so give those missing countries another tag
sessions.loc[sessions.user_country == '','user_country'] = 'Missing'

sessions.to_csv("clean_sessions.csv",index_label='session_id')

######################################################
# sessions = pd.read_csv("clean_sessions.csv",index_col='session_id')

# pay attention the space after the comma
sessions['cities'] = sessions.cities.str.split(', ')

def count_cities(df):
    c = Counter()
    for cities in df.cities:
        for city in cities:
            c[city] +=1
    return pd.Series(c)

searchcity_by_country = sessions.groupby("user_country").apply(count_cities).unstack(fill_value=0)

searchcity_by_country_normed = normalize(searchcity_by_country,axis=1)

country_similarity = searchcity_by_country_normed.dot(searchcity_by_country_normed.T)
country_similarity = pd.DataFrame(country_similarity,
                                  index = searchcity_by_country.index,
                                  columns=searchcity_by_country.index)

country_similarity['Missing'].sort_values(ascending=False)


def sort_neighbors(X):
    Xtrain = X.loc[X.index != 'Missing', :]
    countries = Xtrain.index

    neigh = NearestNeighbors(n_neighbors=Xtrain.shape[0])  # return all neighbors
    neigh.fit(Xtrain)

    distance, indices = neigh.kneighbors(X.loc[['Missing'], :])

    distance = distance[0]
    indices = indices[0]
    countries = countries[indices]

    return pd.DataFrame(zip(countries, distance), columns=['country', 'distance'])

def count_hours(df):
    df.timestamp.dt.hour.value_counts()

hours_by_country = sessions.groupby("user_country").apply(lambda df: df.timestamp.dt.hour.value_counts()).unstack(fill_value=0)


##############
pca = PCA(n_components=2)
country2d = pca.fit_transform(searchcity_by_country)
plt.scatter(country2d[:,0],country2d[:,1],s=100)
for index,country in enumerate( searchcity_by_country.index):
    x = country2d[index,0]
    y = country2d[index,1]
    plt.text(x,y,country)

######################################################################
searchcity_by_user = sessions.groupby("user_id").apply(count_cities).unstack(fill_value=0)
searchcity_by_user = searchcity_by_user.transpose()

searchcity_by_user_normed = normalize(searchcity_by_user,axis=1)

city_similarity = searchcity_by_user_normed.dot(searchcity_by_user_normed.T)
city_similarity = pd.DataFrame(city_similarity,
                               index = searchcity_by_user.index,
                               columns = searchcity_by_user.index)

city_similarity.to_csv("city_similarity.csv",index_label="city")

### find top K most similar of each song
def most_similar(s,topk):
    # [0] must be itself
    similar_ones = s.sort_values(ascending=False)[1:topk+1].index.values
    return pd.Series(similar_ones,index = ["similar#{}".format(i) for i in xrange(1,topk+1)])

city_similarity.apply(most_similar,topk=1,axis=1)

########
pca = PCA(n_components=2)
city2d = pca.fit_transform(searchcity_by_user)
plt.scatter(city2d[:,0],city2d[:,1])
for index,city in enumerate( searchcity_by_user.index):
    x = city2d[index,0]
    y = city2d[index,1]
    plt.text(x,y,city)


def search_distance(cities,similar2dist):
    sumdist = 0
    total = len(cities)

    # if total=1, then distance =0
    for i1 in xrange(total-1):
        city1 = cities[i1]

        for i2 in xrange(i1+1,total):
            city2 = cities[i2]

            similarity = city_similarity.loc[city1,city2]
            dist = similar2dist(similarity)

            sumdist += dist

    return sumdist

distances = sessions.cities.map(lambda cities: search_distance(cities,lambda s: np.sqrt(1-s*s)))
# distances = sessions.cities.map(lambda cities: search_distance(cities,lambda s: 1-s*s))
distances[ (distances>0) & (distances <1)].hist(bins=50,alpha=0.5,normed=True)
distances[ (distances>0) & (distances <1)].plot(kind='kde',style='k--')

sessions["num_searched"] = sessions.cities.map(len)

plt.scatter(sessions.num_searched,sessions.search_distance)

sessions.loc[distances<0.94,'cities'].to_csv('temp.csv')

##############################################







