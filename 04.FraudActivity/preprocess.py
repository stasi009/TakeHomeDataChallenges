import bisect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ip2country = pd.read_csv("IpAddress_to_Country.csv")
datas = pd.read_csv("Fraud_Data.csv")


class IpLookupTable(object):
    def __init__(self, df):
        self._nrows = df.shape[0]
        self._ip_lowbounds = [0 for _ in xrange(self._nrows + 2)]
        self._countries = ["Unknown" for _ in xrange(self._nrows + 2)]

        for r in xrange(1, self._nrows + 1):
            self._ip_lowbounds[r] = df.iloc[r - 1, 0]
            self._countries[r] = df.iloc[r - 1, 2]
            assert self._ip_lowbounds[r] > self._ip_lowbounds[r - 1]

        # we cannot assign all ip> last low boundary to be that country
        self._ip_lowbounds[self._nrows + 1] = df.iloc[self._nrows - 1, 1] + 1

    def find_country(self, ip):
        index = bisect.bisect(self._ip_lowbounds, ip) - 1
        assert ip >= self._ip_lowbounds[index] and (index == self._nrows + 1 or ip < self._ip_lowbounds[index + 1])
        return self._countries[index]


iplookuptable = IpLookupTable(ip2country)


# ----------------- signup_time and purchase_time
def interval_after_signup(s):
    signup_time = pd.to_datetime(s["signup_time"])
    purchase_time = pd.to_datetime(s["purchase_time"])
    interval = purchase_time - signup_time
    return interval.total_seconds()


datas["interval_after_signup"] = datas.apply(interval_after_signup, axis=1)
datas.drop(["signup_time", "purchase_time"], axis=1, inplace=True)

# ----------------- ip to country
datas["country"] = datas.ip_address.map(iplookuptable.find_country)

# since the data only contains each user's first purchase
# so all user_id are unique, so if device is shared, then it is shared by different users
datas["dev_shared"] = datas.device_id.duplicated(keep=False)
datas["ip_shared"] = datas.ip_address.duplicated(keep=False)
datas.drop(["ip_address", "device_id"], axis=1, inplace=True)

# reorder the columns
datas = datas.loc[:, ["user_id", "source", "browser", "country", \
                      "sex", "age", "purchase_value", "interval_after_signup", \
                      "dev_shared", "ip_shared","class"]]
datas.set_index("user_id",inplace=True)
datas.to_csv("fraud_cleaned.csv",index_column="user_id")
