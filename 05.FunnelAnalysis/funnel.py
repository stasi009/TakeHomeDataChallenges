
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.feature_selection import chi2,f_classif

def load_user_pages(allusers,filename):
    if allusers is None:
        allusers = pd.read_csv(filename,index_col='user_id')
        print "{} pages from '{}'".format(allusers.shape[0],filename)
    else:
        current_pages = pd.read_csv(filename,index_col="user_id")
        allusers.loc[current_pages.index,"page"] = current_pages.page
        print "{} pages from '{}'".format(current_pages.shape[0], filename)

    return allusers

allusers = pd.read_csv("home_page_table.csv",index_col="user_id")
users_to_search = pd.read_csv("search_page_table.csv",index_col="user_id")
users_to_pay = pd.read_csv("payment_page_table.csv",index_col="user_id")
users_to_confirm = pd.read_csv("payment_confirmation_table.csv",index_col="user_id")

user_infos = pd.read_csv("user_table.csv",index_col="user_id")
user_infos.loc[:,"date"] = pd.to_datetime(user_infos.date)

def drop_percentage(src_table,srctag,dest_table,desttag):
    n_src_users = src_table.shape[0]
    n_dest_users = dest_table.shape[0]
    remain_ratio = n_dest_users * 100.0 / n_src_users
    print "{} users on {}".format(n_src_users,srctag)
    print "{} users on {}".format(n_dest_users,desttag)
    print "{} ==> {}, {:.2f}% left, {:.2f}% dropped".format(srctag,desttag,remain_ratio,100-remain_ratio)

drop_percentage(allusers,"home",users_to_search,"search")
drop_percentage(users_to_search,"search",users_to_pay,"payment")
drop_percentage(users_to_pay,"payment",users_to_confirm,"confirm")


allusers.loc[users_to_search.index,"page"] = users_to_search.page
allusers.loc[users_to_pay.index,"page"] = users_to_pay.page
allusers.loc[users_to_confirm.index,"page"] = users_to_confirm.page

allusers = allusers.join(user_infos)

allusers.to_csv("all_users.csv",index_label="user_id")

########################################################
desktop_users = allusers.loc[allusers.device == "Desktop",:]
mobile_users = allusers.loc[allusers.device == "Mobile",:]


desktop_page_dist = desktop_users.page.value_counts(normalize=True)
mobile_page_dist = mobile_users.page.value_counts(normalize=True)

page_dist = pd.concat([desktop_page_dist,mobile_page_dist],axis=1,keys=["desktop","mobile"])

page_dist.plot(kind="bar")

########################################
pages = ["home_page","search_page","payment_page","payment_confirmation_page"]
allusers["page"] = allusers.page.astype("category",categories = pages,ordered=True)

def conversion_rates(df):
    stage_counts = df.page.value_counts()
    convert_from = stage_counts.copy()

    total = df.shape[0]
    for page in stage_counts.index:
        n_left = stage_counts.loc[page]
        n_convert = total - n_left
        convert_from[page] = n_convert
        total = n_convert

    cr = pd.concat([stage_counts,convert_from],axis=1,keys=["n_drop","n_convert"])
    cr["convert_rates"] = cr.n_convert.astype(np.float)/(cr.n_drop + cr.n_convert)
    cr['drop_rates'] = 1 - cr.convert_rates

    return cr

all_convert_rates = conversion_rates(allusers)

desktop_convert_rates = conversion_rates(desktop_users)
mobile_convert_rates = conversion_rates(mobile_users)

convertrates_dev_contrast = pd.concat([desktop_convert_rates.convert_rates,
                                       mobile_convert_rates.convert_rates],
                                      axis=1,keys = ["desktop","mobile"])
convertrates_dev_contrast.plot(kind="bar")

droprates_dev_contrast = pd.concat([desktop_convert_rates.drop_rates,
                                    mobile_convert_rates.drop_rates],
                                   axis=1,keys = ["desktop","mobile"])
droprates_dev_contrast.plot(kind="bar")

#############################################################
D = allusers.copy()
D["converted"] = (D.page == "payment_confirmation_page").astype(np.int)
del D["page"]
del D["date"]

D["from_mobile"] = (D.device == "Mobile").astype(np.int)
del D["device"]

D["is_male"] = (D.sex == "Male").astype(np.int)
del D["sex"]

D.pivot_table("converted", index="from_mobile", columns="is_male", aggfunc="mean", margins=True)


##################################
feat_names = ["from_mobile","is_male"]
dt = DecisionTreeClassifier(max_depth=2)
dt.fit(D.loc[:,feat_names],D.loc[:,'converted'])
export_graphviz(dt,out_file="tree.dot",feature_names=feat_names,class_names=["NotConvert","Convert"])

##################### check time's importances
D["weekday"] = allusers.date.dt.weekday
D["month"] = allusers.date.dt.month

feat_names = ["from_mobile","is_male","weekday",'month']
chi2(D.loc[:,feat_names],D.converted)
f_classif(D.loc[:,feat_names],D.converted)

### group by month
D.groupby(by="month")["converted"].agg(['count','mean'])
D.groupby(by="weekday")["converted"].agg(['count','mean'])














