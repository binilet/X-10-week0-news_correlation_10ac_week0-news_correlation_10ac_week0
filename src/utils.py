import os
import sys
import glob
import json
import datetime
from collections import Counter
from collections import Counter

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords



def find_top_websites(data,url_column='url',top=10):
    """
        this function will get the top [top] websites with highest article counts
    """
    data['domain'] = data[url_column].apply(lambda x: x.split('/')[2])

    #count occurences of each domain
    domain_counts = data['domain'].value_counts()

    top_domains = domain_counts.head(top)
    return top_domains

def find_high_traffic_websites(data,top=10):
    """
    this function will return websites with high reference ips(assuming the ips are the number of traffic)
    """

    print(data.head(2))
    traffic_per_domain = data.groupby(['Domain'])['RefIPs'].sum()
    traffic_per_domain = traffic_per_domain.sort_values(ascending=False)
    return traffic_per_domain.head(top)

