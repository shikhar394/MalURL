#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 11:49:24 2017

@author: shikhar
"""


#!/usr/bin/env python
# Description: A quick script to check a domain's Alexa rank.
# Author: c0dist
# Usage: python alexa.py <domain_to_quei

from bs4 import BeautifulSoup
from urllib.request import urlopen
Test_URL = [line.split(',') for line in open("URL.txt")]
print(Test_URL)
Rank = {}
for URL in Test_URL:
    soup = BeautifulSoup(urlopen("http://data.alexa.com/data?cli=10&dat=snbamz&url="+URL[0]).read(), "lxml")
#==============================================================================
#     Rank[soup.popularity['url']] = soup.popularity['text']
#     #Rank['Rank'] = 
#     Rank['Country'] = soup.country['name']
#     Rank['CountryRank'] = soup.country['rank']
#==============================================================================
    print(URL[1] , ":", URL[0], end = ' ')    
    try:
        print(soup.popularity['text'], end = " ")
    except:
        print(-1, end =" ")
    try:
        print(soup.country['rank'])
    except:
        print(-1)