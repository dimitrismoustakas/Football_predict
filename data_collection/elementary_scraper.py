# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:41:37 2023

@author: mouts
"""

import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import re

# Entering the league's  link
link = "https://understat.com/league/La_liga/2021"
res = requests.get(link)
soup = BeautifulSoup(res.content,'lxml')
scripts = soup.find_all('script')
# Get the fixtures data
strings = scripts[1].string 
# Getting rid of unnecessary characters from json data
ind_start = strings.index("('")+2 
ind_end = strings.index("')") 
json_data = strings[ind_start:ind_end] 
json_data = json_data.encode('utf8').decode('unicode_escape')
data = json.loads(json_data)

df = pd.DataFrame(data)
