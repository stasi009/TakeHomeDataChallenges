import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

def load_data():
    with open("song.json", "rt") as inf:
        data = json.load(inf)

    data = pd.DataFrame(data)
    data.set_index("id", inplace=True)
    data["time_played"] = pd.to_datetime(data.time_played)
    data['user_sign_up_date'] = pd.to_datetime(data.user_sign_up_date)

    return data


data = load_data()

# ************* What are the top 3 and the bottom 3 states in terms of number of users?
user_counts = data.groupby("user_state").user_id.agg(lambda ids: len(np.unique(ids)))
user_counts.sort_values(inplace=True, ascending=False)

print "top 3 states in #users: "
print user_counts.iloc[:3]

print "bottom 3 states in #users: "
print user_counts.iloc[:-4:-1]

# ************* top 3 and the bottom 3 states in terms of user engagement
def count_by_state(df):
    """ all data in df come from the same state """
    total_played = df.shape[0]
    first_play_dt = df.time_played.min()
    last_play_dt = df.time_played.max()
    duration = last_play_dt - first_play_dt
    duration_hours = duration.total_seconds()/60.0
    return pd.Series([first_play_dt,last_play_dt, duration, duration_hours, total_played],
                     index=["first_play_dt",'last_play_dt','duration','duration_hours','total_played'])

counts_by_states = data.groupby("user_state").apply(count_by_state)
counts_by_states["hr_average"] = counts_by_states.total_played/counts_by_states.duration_hours
counts_by_states.sort_values(by="hr_average",ascending=False,inplace=True)

# ************* ﬁrst user who signed-up for each state
def find_first_signup(df):
    idx = df.user_sign_up_date.argmin()
    return df.loc[idx,["user_id","user_sign_up_date"]]

first_users = data.groupby("user_state").apply(find_first_signup)
first_users.sort_values(by="user_sign_up_date")

# ************* recommend songs
def count_by_song(df):
    """ all data in df come from the same song"""
    return pd.Series( Counter(df.user_id) )

counts_by_songs = data.groupby("song_played").apply(count_by_song)
counts_by_songs = counts_by_songs.unstack(fill_value=0)

cnts_by_songs_normed = normalize(counts_by_songs,axis=1)
songs_similarity = cnts_by_songs_normed.dot(cnts_by_songs_normed.T)
songs_similarity = pd.DataFrame(songs_similarity,index=counts_by_songs.index,columns=counts_by_songs.index)

### find top K most similar of each song
def most_similar_songs(s,topk):
    # [0] must be itself
    similar_ones = s.sort_values(ascending=False)[1:topk+1].index.values
    return pd.Series(similar_ones,index = ["similar#{}".format(i) for i in xrange(1,topk+1)])

songs_similarity.apply(most_similar_songs,topk=1,axis=1)