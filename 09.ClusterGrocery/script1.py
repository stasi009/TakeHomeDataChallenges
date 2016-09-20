import re
from collections import Counter
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def agg_by_user(same_user_df):
    # 'sum' is adding two lists into one big list
    all_item_ids = same_user_df.id.str.split(',').sum()
    # transform from string to int, make it easier to be sorted later
    return pd.Series(Counter(int(id) for id in all_item_ids))


def find_most():
    ################### the customer who bought the most items overall in her lifetime
    user_item_counts = purchase_history.groupby("user_id").apply(agg_by_user)
    # since agg_by_user will return Series with different index
    # so 'user_item_counts' is a Hierarchical Series
    user_item_counts = user_item_counts.unstack(fill_value=0)

    # we assume each "item id" in the purchase history stands for 'item_count=1'
    user_item_total = user_item_counts.sum(axis=1)
    print "custom who bought most in lifetime is: {}, and he/she bought {} items".format(user_item_total.argmax(),
                                                                                         user_item_total.max())

    ################### for each item, the customer who bought that product the most
    max_user_byitem = user_item_counts.apply(
        lambda s: pd.Series([s.argmax(), s.max()], index=["max_user", "max_count"]))
    max_user_byitem = max_user_byitem.transpose()
    max_user_byitem.index.name = "Item_id"

    # merge with item names
    max_user_byitem = max_user_byitem.join(items).loc[:, ["Item_name", "max_user", "max_count"]]

def show_clusters(items_rotated,labels):
    # colors = iter(cm.rainbow(np.linspace(0, 1, len(labels))))
    colors =  itertools.cycle (["b","g","r","c","m","y","k"])

    grps = items_rotated.groupby(labels)
    for label,grp in grps:
        plt.scatter(grp.pc1,grp.pc2,c=next(colors),label = label)

        print "*********** Label [{}] ***********".format(label)
        names = items.loc[ grp.index,"Item_name"]
        for index, name in enumerate(names):
            print "\t<{}> {}".format(index+1,name)

    # annotate
    for itemid in items_rotated.index:
        x = items_rotated.loc[itemid,"pc1"]
        y = items_rotated.loc[itemid,"pc2"]
        name = items.loc[itemid,"Item_name"]
        name = re.sub('\W', ' ', name)
        plt.text(x,y,name)

    # plt.legend(loc="best")

def cluster_once(n_components,n_clusters):
    print "first {} PC explain {:.1f}% variances".format(n_components,
                                                         100 * sum(pca.explained_variance_ratio_[:n_components]))

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(items_rotated.values[:, :n_components])

    # display results
    show_clusters(items_rotated, kmeans.labels_)


def cluster():
    # A is |U|*|I|, and each item is normalized
    A = normalize(user_item_counts.values, axis=0)
    item_item_similarity = A.T.dot(A)
    item_item_similarity = pd.DataFrame(item_item_similarity,
                                        index=user_item_counts.columns,
                                        columns=user_item_counts.columns)

    pca = PCA()
    items_rotated = pca.fit_transform(item_item_similarity)
    items_rotated = pd.DataFrame(items_rotated,
                                 index=user_item_counts.columns,
                                 columns=["pc{}".format(index+1) for index in xrange(items.shape[0])])

    # cluster
    cluster_once(n_components=6,n_clusters=15)


if __name__ == "__main__":
    items = pd.read_csv("item_to_id.csv", index_col='Item_id')
    items.sort_index(inplace=True)

    purchase_history = pd.read_csv("purchase_history.csv")

    find_most()
