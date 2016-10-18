
import pandas as pd

#################################### constant definition
Site = 'http://www.mysearchforhotels.com/shop/hotelsearch?'
LenSite = len(Site)

ParamPrefix = 'hotel.'
LenParaPrefix = len(ParamPrefix)

####################################
def parse_url(url):
    assert url[LenSite-1] == '?'
    segments = url[LenSite:].split('&')

    params = {}
    for segment in segments:
        kvpairs = segment.split('=')
        assert len(kvpairs) == 2

        k = kvpairs[0]
        assert k[LenParaPrefix-1] == '.'
        k = k[LenParaPrefix:]

        params[k] = kvpairs[1]

    return params


####################################
def load_parse():
    succ_urls = []
    fail_urls = []
    with open("url_list.txt",'rt') as inf:
        for index,line in enumerate(inf):
            try:
                url = parse_url(line.strip())
                succ_urls.append(url)
            except:
                fail_urls.append(line)
                print "failed to parse: {}".format(line)

            if index%100 ==0:
                print '{} lines parsed'.format(index)

    print "************ ALL DONE ************"
    return succ_urls,fail_urls

succ_urls,fail_urls = load_parse()
assert len(fail_urls) == 0

####################################
urls = pd.DataFrame(succ_urls)

######################## clean
urls['checkin'] = pd.to_datetime(urls.checkin)
urls['checkout'] = pd.to_datetime(urls.checkout)
urls["children"].fillna(0,inplace=True)
urls['city'] = urls.city.str.replace('+',' ')
urls['search_page'] = urls.search_page.astype(int)
urls.to_csv("urls.csv",index=False)

##
urls = pd.read_csv('urls.csv')

###################
urls.amenities[pd.notnull(urls.amenities)]

# only single amenity each time
urls.amenities.value_counts()

urls.amenities.map(lambda s: 0 if pd.isnull(s) else 1)

##############################
def firstpage_ratio(s):
    total = s.shape[0]
    n_firstpage = (s == 1).sum()
    return float(n_firstpage)/total

city_page1_ratio = urls.groupby('city')['search_page'].agg(firstpage_ratio).sort_values()


