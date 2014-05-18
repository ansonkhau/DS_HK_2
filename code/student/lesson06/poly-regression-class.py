
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from numpy import arange,array,ones#,random,linalg
from pylab import plot,show
from scipy import stats

xi = arange(0,9)
A = array([ xi, ones(9)])
# linearly generated sequence
y = [19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24]
slope, intercept, r_value, p_value, std_err = stats.linregress(xi,y)

# H0
print 'response mean', np.mean(y)

# Standard Deviation of Y
print 'standard Deviation of Y', np.std(y)

# Coefficient of Determination
print 'r-squared value', r_value**2

# Is the statistic significant?
print 'p_value', p_value

print 'standard deviation of error terms', std_err

line = slope*xi+intercept
plot(xi,line,'r-',xi,y,'o')
show()