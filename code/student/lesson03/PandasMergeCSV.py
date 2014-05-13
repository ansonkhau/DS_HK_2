from __future__ import division
import urllib
from urlparse import urlparse
import pandas as pd
import os.path

#Define DataDir
Data_Dir = '/Users/ansonau/DS_HK_2/data'

def download_series(base_url,extension,limit):
	file_list = []
	for i in range(limit)