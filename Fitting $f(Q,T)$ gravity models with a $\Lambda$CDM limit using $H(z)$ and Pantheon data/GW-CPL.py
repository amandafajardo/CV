import numpy as np
import matplotlib.pyplot as plt
import math
import emcee
import getdist
import h5py

GWData = np.array([[0.09,0.21,0.09,0.20,0.07,0.49,0.20,0.12,0.01,0.21,0.35,0.30,0.15,
            0.66,0.80,0.53,0.45,0.03,0.08,0.29,0.28,0.39,0.77,0.38,0.49,0.72,0.25,
            0.53,0.51,0.54,0.19,0.38,0.79,0.16,0.18,0.73,0.16,0.60,0.18,0.65,0.61,
            0.05,0.40,0.31,0.75,0.29,0.32,0.12,0.61,0.16], [0.440,1.080,0.450,0.990,
            0.320,2.840,1.030,0.600,0.040,1.060,1.940, 1.58,0.74,4.10,5.15,3.15,2.55,
            0.16,0.38,1.52,1.49,2.16,4.93,2.11,2.85, 4.53,1.28,3.10,2.99,3.16,0.93,
            2.14,5.07,0.80,0.90,4.61,0.81,3.60,0.89, 3.97,3.69,0.24,2.22,1.66,4.77,
            1.57,1.70,0.57,3.68,0.78], [0.03,0.09,0.04,0.08,0.02,0.21,0.07,0.04,0.00,
            0.07,0.15,0.10,0.03, 0.30,0.31,0.21,0.22,0.02,0.04,0.11,0.10,0.14,0.34,
            0.26,0.27,0.29,0.10, 0.61,0.27,0.22,0.10,0.12,0.31,0.07,0.07,0.35,0.12,
            0.22,0.07,0.32,0.26, 0.010,0.15,0.10,0.45,0.17,0.11,0.04,0.38,0.06], [0.170,
            0.550,0.190,0.440,0.120,1.400,0.390,0.220,0.015,0.420,0.970, 0.59,0.17,
            2.41,2.44,1.42,1.56,0.07,0.19,0.71,0.59,0.94,2.76,1.79,2.02, 2.30,0.57,4.85,
            2.02,1.67,0.56,0.79,2.57,0.38,0.40,2.84,0.71,1.56,0.37, 2.56,2.04,0.05,0.95,
            0.63,3.70,1.07,0.71,0.22,2.98,0.37]])

x, y, xerr, yerr = GWData [ :, GWData[0].argsort()]

print(x.shape)
print(y.shape)
print(xerr.shape)
print(yerr.shape)

from scipy.integrate import quad

def log_likelihood(theta, x, y, yerr):
    h, Omega_M, w_0, w_1, Xi_0, n = theta
    def integrand(x, Omega_M, w_0, w_1):
        return 1/np.sqrt( Omega_M * (1+x)**3 + (1-Omega_M) * np.exp(-(3*w_1*x)/(1+x)) * np.power(1+x,3*(1+w_0+w_1)) )
    def integral(x,Omega_M, w_0, w_1):
        return quad(integrand,0,x, args=(Omega_M, w_0, w_1))[0]
    vector = np.vectorize(integral)
    model = (3.00065/h*(1+x))*vector(x, Omega_M, w_0, w_1) * (Xi_0 + (1-Xi_0)/(1+x)**n)#* np.exp(-vector2(x,n,Xi_0))
    sigma2 = yerr ** 2 + xerr**2
    return np.sum((y - model) ** 2 / sigma2)

from scipy.optimize import minimize

nll = lambda *args: log_likelihood(*args)
initial = np.array([0.7, 0.3, -1, 0, 1, 2.5])
soln = minimize(nll, initial, bounds=((0.05,0.80),(0.05,0.45), (-4.95,4.95), (-4.95,4.95), (0.05,1.95), (0.05,4.95)), args=(x, y, yerr))
h_ml, Omega_M_ml, w_0_ml, w_1_ml, Xi_0_ml, n_ml = soln.x

print("Maximum likelihood estimates:")
print("Hubble Constant = {0:.5f}".format(h_ml))
print("Omega Matter = {0:.5f}".format(Omega_M_ml))
print("w_0 = {0:.5f}".format(w_0_ml))
print("w_1 = {0:.5f}".format(w_1_ml))
print("n = {0:.5f}".format(n_ml))
print("Xi_0 = {0:.5f}".format(Xi_0_ml))

theta_ml = h_ml, Omega_M_ml, w_0_ml, w_1_ml, Xi_0_ml, n_ml
print("Xi^2=", log_likelihood(theta_ml, x, y, yerr))
print("AIC =", log_likelihood(theta_ml, x, y, yerr)+2*6)
print("BIC =", log_likelihood(theta_ml, x, y, yerr)
 + 6 * math.log(50))

"""
def integrand(x, Omega_M, w_0, w_1):
    return 1/np.sqrt( Omega_M * (1+x)**3 + (1-Omega_M) * np.exp(-(3*w_1*x)/(1+x)) * np.power(1+x,3*(1+w_0+w_1)) )
def integral(x,Omega_M, w_0, w_1):
    return quad(integrand,0,x, args=(Omega_M, w_0, w_1))[0]
vector = np.vectorize(integral)
model_ml = (3.00065/h_ml*(1+x))*vector(x, Omega_M_ml, w_0_ml, w_1_ml) * (Xi_0_ml + (1-Xi_0_ml)/(1+x)**n_ml)
fig = plt.errorbar(x,y,xerr=xerr,yerr=yerr, fmt='o', color='red', label='Data')
plt.plot(x, model_ml, color='green', label="BA Fit Model" )
plt.legend(fontsize=14)
plt.xlabel('z')
plt.ylabel('$H(z)$ / (km / (s Mpc))')

plt.legend(loc='upper left')
plt.show()

def log_prior(theta):
    h, Omega_M, w_0, w_1, Xi_0, n  = theta
    if 0 < Omega_M < 0.5 and 0 < h < 0.85 and -5 < w_0 < 5 and -5 < w_1 < 5 and 0 < n < 5 and 0 < Xi_0 < 2:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

from multiprocessing import Pool

import os

os.environ["OMP_NUM_THREADS"] = "4"

with Pool() as pool:
    pos = soln.x + 1e-4 * np.random.randn(32, 6)
    nwalkers, ndim = pos.shape
    filename = "LCDM-GW1.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x,y,yerr), pool=pool, backend=backend)
    sampler.run_mcmc(pos, 30000, progress=True)

fig, axes = plt.subplots(6, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["$h$", "$\Omega_M$", "$w_0$", "$w_1$", "$\Xi_0$", "$n$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("Step number")

tau = sampler.get_autocorr_time()
print(tau)

flat_samples = sampler.get_chain(discard=250, thin=50, flat=True)

plt.show()

from getdist import plots, MCSamples

samples = MCSamples(samples=flat_samples,names = ["h","\Omega_M", "w_0", "w_1", "\Xi_0", "n"] , labels = ["h","\Omega_M", "w_0", "w_1", "\Xi_0", "n"])
fig = plots.get_subplot_plotter()
fig.settings.figure_legend_frame = False
fig.settings.alpha_filled_add=0.4
fig.settings.title_limit_fontsize = 14
fig.triangle_plot(samples, ["h","\Omega_M", "w_0", "w_1", "\Xi_0", "n"], filled=True, contour_colors=['red'], title_limit=1)
fig.export("resultsCPL-GW1.png")"""
