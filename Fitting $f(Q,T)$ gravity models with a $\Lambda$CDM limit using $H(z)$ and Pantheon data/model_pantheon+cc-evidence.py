import numpy as np
import matplotlib.pyplot as plt
import math
import emcee
import getdist
import h5py

values = np.loadtxt('Pantheon_data.txt', delimiter=',',unpack='True')

x,y,yerr = values [ :, values[0].argsort()]

data = np.array([[0, 0.07,0.1,0.12,0.17,0.179,0.199,0.2,0.27,0.28,
            0.352, 0.3802, 0.4, 0.4004, 0.4247, 0.4497, 0.47,
            0.4783, 0.48, 0.593, 0.68, 0.781, 0.875, 0.88, 0.9,
            1.037, 1.3, 1.363, 1.43, 1.53, 1.75, 1.965, 0.24, 0.3, 0.31,
            0.35, 0.36, 0.38, 0.43, 0.44, 0.51, 0.52, 0.56, 0.57, 0.59,
            0.6, 0.61, 0.64, 0.73, 2.33, 2.34, 2.36], [73.24, 69.0, 69.0,
            68.6, 83.0, 75.0, 75.0, 72.9, 77.0, 88.8, 83.0, 83.0, 95.0,
            77.0, 87.1, 92.8, 89.0, 80.9, 97.0, 104.0, 92.0, 105.0, 125.0,
            90.0, 117.0, 154.0, 168.0, 160.0, 177.0, 140.0, 202.0, 186.5,
            79.69, 81.7, 78.17, 82.7, 79.93, 81.5, 86.45, 82.6, 90.4, 94.35,
            93.33, 92.9, 98.48, 87.9, 97.3, 98.82, 97.3, 224, 222, 226], [1.74,
            19.6, 12.0, 26.2,8.0,4.0,5.0,29.6,14.0,36.6,14.0,
            13.5, 17.0, 10.2, 11.2, 12.9, 49.6, 9.0, 62.0, 13.0,
            8.0, 12.0, 17.0, 40.0, 23.0, 20.0, 17.0, 33.6, 18.0,
            14.0, 40.0, 50.4, 2.65, 6.22, 4.74, 8.4, 3.39, 1.9, 3.68, 7.8,
            1.9, 2.65, 2.32, 7.8, 3.19, 6.1, 2.1, 2.99, 7, 8, 7, 8]])

x_2,y_2,yerr_2 = data [ :, data[0].argsort()]

from scipy.integrate import quad

def log_likelihood(theta):
    h, Omega_Lambda, b = theta
    def func(z, E):
        a = np.sqrt(1 + b*(E**2-Omega_Lambda)) - 1
        m = 3 / (E * b * (1+z)) * (1+2/5.*a) * a
        return m
    def Runge_kutta(E0, x):
        z = np.linspace(0,x,100)
        l = x/100
        E = np.zeros(z.shape)
        E[0] = E0
        for i in range(len(z)-1):
            k1 = l * (func(z[i], E0))
            k2 = l * (func((z[i]+l/2), (E0+k1/2)))
            k3 = l * (func((z[i]+l/2), (E0+k2/2)))
            k4 = l * (func((z[i]+l), (E0+k3)))
            k = (k1+2*k2+2*k3+k4)/6
            En = E0 + k
            E0 = En
            E[i+1] = E0
        return E, l
    E_vec = np.zeros(x.shape)
    for i in range(len(x)):
        E, l = Runge_kutta(1, x[i])
        E = 1/E
        suma = 0.5 * l * (2 * np.sum(E) - E[0] - E[-1])
        E_vec[i] = suma
    model = 5 * np.log10((3e8/h*(1+x))*E_vec)
    sigma2 = yerr ** 2
    def Runge_kutta_2(E0, z):
        z0 = z[0]
        E = np.zeros(z.shape)
        E[0] = E0
        for i in range(len(z)-1):
            h = z[i+1] - z[i]
            k1 = h * (func(z0, E0))
            k2 = h * (func((z0+h/2), (E0+k1/2)))
            k3 = h * (func((z0+h/2), (E0+k2/2)))
            k4 = h * (func((z0+h), (E0+k3)))
            k = (k1+2*k2+2*k3+k4)/6
            En = E0 + k
            E0 = En
            z0 = z0+h
            E[i+1] = E0
        return E
    model_2 = 100*h*Runge_kutta_2(1, x_2)
    sigma2_2 = yerr_2 ** 2
    a1 = -0.5*52*np.log(2.*np.pi)
    a2 = -0.5*1048*np.log(2.*np.pi)
    b1 = -np.sum(np.log(yerr_2))
    b2 = -np.sum(np.log(yerr))
    return a1 + a2 + b1 + b2 -0.5 * np.sum((y - model) ** 2 / sigma2 ) -0.5 * np.sum((y_2 - model_2) ** 2 / sigma2_2 )

def prior_transform(theta):
    """
    A function defining the tranform between the parameterisation in the unit hypercube
    to the true parameters.

    Args:
        theta (tuple): a tuple containing the parameters.

    Returns:
        tuple: a new tuple or array with the transformed parameters.
    """

    hprime, Omega_Lambdaprime, bprime = theta # unpack the parameters (in their unit hypercube form)

    hmin = 0.0  # lower bound on uniform prior on c
    hmax = 1.0  # upper bound on uniform prior on c

    Omega_Lambdamin = 0.0     # mean of Gaussian prior on m
    Omega_Lambdamax = 1.0 # standard deviation of Gaussian prior on m

    bmin = -0.01  # lower bound on uniform prior on c
    bmax = 20

    b = bprime*(bmax-bmin) + bmin # convert back to m
    Omega_Lambda = Omega_Lambdaprime*(Omega_Lambdamax-Omega_Lambdamin) + Omega_Lambdamin  # convert back to c
    h = hprime*(hmax-hmin) + hmin

    return (h, Omega_Lambda, b)

from scipy.optimize import minimize

import dynesty
import time

print('dynesty version: {}'.format(dynesty.__version__))

nlive = 1024      # number of live points
bound = 'multi'   # use MutliNest algorithm for bounds
ndims = 3         # two parameters
sample = 'unif'   # uniform sampling
tol = 0.1         # the stopping criterion

from dynesty import NestedSampler

from schwimmbad import MPIPool
#from multiprocessing import Pool

#with Pool() as pool:
with MPIPool() as pool:
    sampler = NestedSampler(log_likelihood, prior_transform, ndims,
                        bound=bound, sample=sample, nlive=nlive, pool=pool, queue_size=256)
    t0 = time.time()
    sampler.run_nested(dlogz=tol, print_progress=True) # don't output progress bar
    t1 = time.time()

timedynesty = (t1-t0)
print(timedynesty)

res = sampler.results # get results dictionary from sampler

logZdynesty = res.logz[-1]        # value of logZ
logZerrdynesty = res.logzerr[-1]  # estimate of the statistcal uncertainty on logZ

print("log(Z) = {} ± {}".format(logZdynesty, logZerrdynesty))

from dynesty.utils import resample_equal
from matplotlib import pyplot as pl # import pyplot from matplotlib
import matplotlib as mpl
rcparams = {}
rcparams['text.usetex'] = True
rcparams['axes.linewidth'] = 0.5
rcparams['font.family'] = 'sans-serif'
rcparams['font.size'] = 22

# functions for plotting posteriors

from scipy.stats import gaussian_kde

resdict = {}

# draw posterior samples
weights = np.exp(res['logwt'] - res['logz'][-1])
samples_dynesty = resample_equal(res.samples, weights)

resdict['mdynesty_mu'] = np.mean(samples_dynesty[:,0])      # mean of m samples
resdict['mdynesty_sig'] = np.std(samples_dynesty[:,0])      # standard deviation of m samples
resdict['cdynesty_mu'] = np.mean(samples_dynesty[:,1])      # mean of c samples
resdict['cdynesty_sig'] = np.std(samples_dynesty[:,1])      # standard deviation of c samples
resdict['ccdynesty'] = np.corrcoef(samples_dynesty.T)[0,1]  # correlation coefficient between parameters
resdict['dynesty_npos'] = len(samples_dynesty)              # number of posterior samples
resdict['dynesty_time'] = timedynesty                       # run time
resdict['dynesty_logZ'] = logZdynesty                       # log marginalised likelihood
resdict['dynesty_logZerr'] = logZerrdynesty                 # uncertainty on log(Z)

print('Number of posterior samples is {}'.format(len(samples_dynesty)))

from getdist import plots, MCSamples

samples = MCSamples(samples=samples_dynesty,names = ["h","\Omega_\Lambda", "b"], labels = ["h","\Omega_\Lambda", "b"])
fig = plots.get_subplot_plotter()
fig.settings.figure_legend_frame = False
fig.settings.alpha_filled_add=0.4
fig.settings.title_limit_fontsize = 14
fig.triangle_plot(samples, ["h","\Omega_\Lambda", "b"], filled=True, contour_colors=['red'], title_limit=1)
fig.export("model-evidence.png")
