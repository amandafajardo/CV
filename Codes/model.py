import numpy as np
import matplotlib.pyplot as plt
import math
import emcee
import getdist
import h5py

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

x,y,yerr = data [ :, data[0].argsort()]

def log_likelihood(theta, x, y, yerr):
    h, b, m, n = theta
    a = (b+(16*np.pi+3*b)*(1+m*(1+x)**n))/(b+(16*np.pi+3*b)*(1+m))
    l = (3*(8*np.pi+b))/(n*(16*np.pi+3*b))
    model = 100*h*np.power(a,l)
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 )

from scipy.optimize import minimize

nll = lambda *args: -log_likelihood(*args)
initial = np.array([0.7, 0.0, 0.5, 3.0])
soln = minimize(nll, initial, bounds=((0.05,0.95),(-3.95,3.95),(0.05,0.95),(0.05,4.95)), args=(x, y, yerr))
h_ml, b_ml, m_ml, n_ml = soln.x

print("Maximum likelihood estimates:")
print("Hubble Constant = {0:.5f}".format(h_ml))
print("b = {0:.5f}".format(b_ml))
print("m = {0:.5f}".format(m_ml))
print("n = {0:.5f}".format(n_ml))

a_ml = (b_ml+(16*np.pi+3*b_ml)*(1+m_ml*(1+x)**n_ml))/(b_ml+(16*np.pi+3*b_ml)*(1+m_ml))
l_ml = (3*(8*np.pi+b_ml))/(n_ml*(16*np.pi+3*b_ml))
model_ml = 100*h_ml*np.power(a_ml,l_ml)

fig = plt.errorbar(x,y,yerr=yerr, fmt='o', color='red', label='Data')
plt.plot(x, model_ml, color='green', label="Fit Model" )
plt.legend(fontsize=14)
plt.xlabel('z')
plt.ylabel('$H(z)$ / (km / (s Mpc))')

plt.legend(loc='upper left')
plt.show()

def log_prior(theta):
    h, b, m, n = theta
    if -4 < b < 4 and 0 < h < 1 and 0 < m < 1 and 0 < n < 5:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

import os

os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool

with Pool() as pool:
    pos = soln.x + 1e-4 * np.random.randn(32, 4)
    nwalkers, ndim = pos.shape
    filename = "CC.h5"
    backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x,y,yerr), backend=backend)
sampler.run_mcmc(pos, 30000, progress=True)

fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["$h$", "$b$","$m$","$n$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("Step number")

tau = sampler.get_autocorr_time()
print(tau)

flat_samples = sampler.get_chain(discard=500, thin=150, flat=True)

#plt.show()

from getdist import plots, MCSamples

samples = MCSamples(samples=flat_samples,names = ["h","b","m","n"] , labels = ["h","b","m","n"])
fig = plots.get_subplot_plotter()
fig.settings.figure_legend_frame = False
fig.settings.alpha_filled_add=0.4
fig.settings.title_limit_fontsize = 14
fig.triangle_plot(samples, ["h","b","m","n"], filled=True, contour_colors=['orange'], title_limit=1)
fig.export("model.png")
