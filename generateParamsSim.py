import numpy as np
from scipy.integrate import odeint
import os
from simCoronavirusSeattle2 import runModel2
import time
from matplotlib import pyplot as plt
import scipy.stats as stats
import pickle
myseed = np.abs(int((np.random.normal()) * 1000000))
np.random.seed(myseed)
print(myseed)
today = time.strftime("%d%b%Y", time.localtime())
numSimulations = 1000

#distribution for R0
lower, upper = 1.6,  3
mu, sigma = 2.2, 0.7
X = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
# X.rvs(1)

#distribution for gamma
lower, upper = 3, 9
mu, sigma = 5, 0.7
Y = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
# Y.rvs(1)

R0vals = X.rvs(numSimulations)  # np.random.normal((5.17/0.096),0.096, 1)
# print(R0)


gammaEvals = np.random.gamma((5.17 / 0.096), 0.096, numSimulations) #distribution for gammaE
gammaVals = Y.rvs(numSimulations)  # np.random.normal(8, 3, 1000)



fig, ax = plt.subplots(3, sharex=True)
ax[0].hist(X.rvs(10000), normed=True)
ax[1].hist(gammaVals, normed=True)
ax[2].hist(gammaEvals, normed=True)
# plt.show()
# params = [R0vals, gammaEvals, gammaVals,myseed]
# myfilename = 'randomParams' + today + '.py'
# myfile = open(myfilename, 'wb')
# pickle.dump(params, myfile)
# myfile.close()

plt.show()