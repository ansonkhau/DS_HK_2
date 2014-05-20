#matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set some Pandas options
pd.set_option('max_columns', 30)
pd.set_option('max_rows', 20)


# Store data in a consistent place
DATA_DIR = '../data/'
import re
url = 'http://www-958.ibm.com/software/analytics/manyeyes/datasets/af-er-beer-dataset/versions/1.txt'
beer = pd.read_csv(url, delimiter="\t")
beer = beer.dropna()

def good(x):
    if x > 4.3:
        return 1
    else:
        return 0
beer['Good'] = beer['WR'].apply(good)

beer_type = ['Ale', 'Stout', 'IPA', 'Lager'] 



beer['Ale'] = beer['Name'].str.contains('Ale')
beer['Stout'] = beer['Name'].str.contains('Stout')
beer['IPA'] = beer['Name'].str.contains('IPA')
beer['Lager'] = beer['Name'].str.contains('Lager')


def convertbeer(x):
    if x == True:
        return 1
    else:
        return 0

for i in beer_type:
	beer[beer_type[i]] = beer[beer_type[i]].apply(convertbeer)

#beer['Ale'] = beer['Ale'].apply(convertbeer)
#beer['Stout'] = beer['Stout'].apply(convertbeer)
#beer['IPA'] = beer['IPA'].apply(convertbeer)
#beer['Lager'] = beer['Lager'].apply(convertbeer)

from sklearn import linear_model
logm = linear_model.LogisticRegression(penalty = 'l1')

X = beer[ ['Ale', 'Stout', 'IPA', 'Lager', 'Reviews', 'ABV'] ].values
y = beer['Good'].values

logm.fit(X, y)
logm.predict(X)
logm.score(X, y)
