import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mammals = pd.read_csv('/Users/ansonau/DS_HK_2/data/' + 'mammals.csv')
mammals.describe()

from sklearn import linear_model
regr = linear_model.LinearRegression()

body = [[x] for x in mammals['body'].values]
brain = mammals['brain'].values

regr.fit(body, brain)

# Display the coefficients
print regr.coef_

# Display our SSE:
print np.mean((regr.predict(body) - brain) ** 2)

# Scoring our model (closer to 1 is better!)
print regr.score(body, brain)

# figure(figsize=(20,8))
# plt.scatter(mammals['body'], mammals['brain'])
# plt.show()

from numpy import log
# figure(figsize=(20,8))

mammals['log_body'] = log(mammals['body'])
mammals['log_brain'] = log(mammals['brain'])

body_log= [[x] for x in mammals['log_body'].values]
brain_log = mammals['log_brain'].values 
regr.fit(body_log, brain_log)

# Display the coefficients
print regr.coef_
# Display our SSE:
print np.mean((regr.predict(body) - brain) ** 2)

# Scoring our model (closer to 1 is better!)
print regr.score(body, brain)
