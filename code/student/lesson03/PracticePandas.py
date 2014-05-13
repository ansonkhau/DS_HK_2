import pandas as pd 
import numpy as np
df = pd.read_csv('/Users/ansonau/DS_HK_2/data/nytimes.csv')
#print df.describe()
#print df[:10]
dfg = df[['Age','Impressions','Clicks']].groupby(['Age']).agg([np.mean])
print dfg