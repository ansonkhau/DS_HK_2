dgiimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nytimes = pd.read_csv('/Users/ansonau/DS_HK_2/data/' + 'nyagg.csv')

from sklearn import linear_model
regr = linear_model.LinearRegression()
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()

#Gender = nytimes['Gender'].values
Gender = [[x] for x in nytimes['Gender'].values]
#Gender = vec.fit_transform(nytimes['Gender'].values).toarray()
Ctr = nytimes['Ctr'].values

regr.fit(Gender,Ctr)
#Display the Coefficients
print regr.coef_
#Display our SSE:
print np.mean((regr.predict(Gender)-Ctr)**2)
#Display regr.score
print regr.score(Gender,Ctr)

plt.scatter(Gender, Ctr)
plt.plot(Gender, regr.predict(Gender), color='blue', linewidth=3)
plt.show()