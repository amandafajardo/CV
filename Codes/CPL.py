import numpy as np
import matplotlib.pyplot as plt
import math
import emcee
import getdist
import time
import h5py

start=time.time()

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

values = np.loadtxt('Pantheon_data.txt', delimiter=',',unpack='True')

x_1 = []
y_1 = []
yerr_1 = []

x_1,y_1,yerr_1 = values [ :, values[0].argsort()]

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

x_3, y_3, xerr_3, yerr_3 = GWData [ :, GWData[0].argsort()]

from scipy.integrate import quad

def log_likelihood(theta, x_1, y_1, yerr_1, x_2, y_2, yerr_2, x_3, y_3, xerr_3, yerr_3):
    h, Omega_M, w_0, w_1, Xi_0, n = theta
    def integrand(x, Omega_M, w_0, w_1):
        return 1/np.sqrt( Omega_M * (1+x)**3 + (1-Omega_M) * np.exp(-(3*w_1*x)/(1+x)) * np.power(1+x,3*(1+w_0+w_1)) )
    def integral(x,Omega_M, w_0, w_1):
        return quad(integrand,0,x, args=(Omega_M, w_0, w_1))[0]
    vector = np.vectorize(integral)
    model_1 = 5 * np.log10((3e8/h*(1+x_1))*vector(x_1, Omega_M, w_0, w_1))
    sigma2_1 = yerr_1 ** 2
    model_2 = 100 * h * np.sqrt( Omega_M * (1+x_2)**3 + (1-Omega_M) * np.exp(-(3*w_1*x)/(1+x)) * np.power(1+x,3*(1+w_0+w_1)) )
    sigma2_2 = yerr_2 ** 2
    model_3 = (3.00065/h*(1+x_3))*vector(x_3, Omega_M, w_0, w_1) * (Xi_0 + (1-Xi_0)/(1+x_3)**n)#* np.exp(-vector2(x,n,Xi_0))
    sigma2_3 = yerr_3 ** 2 + xerr_3**2
    return -0.5 * np.sum((y_1 - model_1) ** 2 / sigma2_1 + np.log(sigma2_1)) -0.5 * np.sum((y_2 - model_2) ** 2 / sigma2_2 + np.log(sigma2_2)) - 0.5 * np.sum((y_3 - model_3)**2 /sigma2_3 + np.log(sigma2_3))

from scipy.optimize import minimize

nll = lambda *args: -log_likelihood(*args)
initial = np.array([0.7, 0.3, -1, 0, 0.5, 2.5])
soln = minimize(nll, initial, bounds=((0.05,0.95),(0.05,0.95), (-4.95,4.95), (-4.95,4.95), (0.05, 1.95), (0.05, 4.95)), args=(x_1,y_1,yerr_1,x_2,y_2,yerr_2, x_3, y_3, xerr_3, yerr_3))
h_ml, Omega_M_ml, w_0_ml, w_1_ml, Xi_0_ml, n_ml = soln.x

def log_likelihood_1(theta, x_1, y_1, yerr_1, x_2, y_2, yerr_2, x_3, y_3, xerr_3, yerr_3):
    h, Omega_M, w_0, w_1, Xi_0, n = theta
    def integrand(x, Omega_M, w_0, w_1):
        return 1/np.sqrt( Omega_M * (1+x)**3 + (1-Omega_M) * np.exp(-(3*w_1*x)/(1+x)) * np.power(1+x,3*(1+w_0+w_1)) )
    def integral(x,Omega_M, w_0, w_1):
        return quad(integrand,0,x, args=(Omega_M, w_0, w_1))[0]
    vector = np.vectorize(integral)
    model_1 = 5 * np.log10((3e8/h*(1+x_1))*vector(x_1, Omega_M, w_0, w_1))
    sigma2_1 = yerr_1 ** 2
    model_2 = 100 * h * np.sqrt( Omega_M * (1+x_2)**3 + (1-Omega_M) * np.exp(-(3*w_1*x)/(1+x)) * np.power(1+x,3*(1+w_0+w_1)) )
    sigma2_2 = yerr_2 ** 2
    model_3 = (3.00065/h*(1+x_3))*vector(x_3, Omega_M, w_0, w_1) * (Xi_0 + (1-Xi_0)/(1+x_3)**n)
    sigma2_3 = yerr_3 ** 2 + xerr_3**2
    return 0.5 * np.sum((y_1 - model_1) ** 2 / sigma2_1)  + 0.5 * np.sum((y_2 - model_2) ** 2 / sigma2_2) + 0.5 * np.sum((y_3 - model_3)**2 /sigma2_3)

theta_ml = h_ml, Omega_M_ml, w_0_ml, w_1_ml, Xi_0_ml, n_ml
print(log_likelihood_1(theta_ml, x_1, y_1, yerr_1, x_2, y_2, yerr_2, x_3, y_3, xerr_3, yerr_3))

def integrand(x,Omega_M, w_0, w_1):
    return 1/np.sqrt( Omega_M * (1+x)**3 + (1-Omega_M) * np.exp(-(3*w_1*x)/(1+x)) * np.power(1+x,3*(1+w_0+w_1)) )

def integral(x,Omega_M, w_0, w_1):
    return quad(integrand,0,x,args=(Omega_M, w_0, w_1))[0]

vector_ml = np.vectorize(integral)

model_ml = 5 * np.log10((3e8/h_ml*(1+x_1))*vector_ml(x_1,Omega_M_ml, w_0_ml, w_1_ml))

print("Maximum likelihood estimates:")
print("h = {0:.5f}".format(h_ml))
print("Omega Matter = {0:.5f}".format(Omega_M_ml))
print("w_0 = {0:.5f}".format(w_0_ml))
print("w_1 = {0:.5f}".format(w_1_ml))
print("Xi_0 = {0:.5f}".format(Xi_0_ml))
print("n = {0:.5f}".format(n_ml))

fig = plt.errorbar(x_1,y_1,yerr_1, fmt='o', color='red', label='Data')
plt.plot(x_1, model_ml, color='green', label="CPL Fit Model" )
plt.legend(fontsize=14)
plt.xlabel('z')
plt.ylabel('$\mu$')

plt.legend(loc='upper left')

def log_prior(theta):
    h, Omega_M, w_0, w_1, Xi_0, n = theta
    if 0 < Omega_M < 1 and 0 < h < 0.85 and -5 < w_0 < 5 and -5 < w_1 < 5 and 0 < Xi_0 < 2 and 0 < n < 5:
        return 0.0
    return -np.inf

def log_probability(theta, x_1, y_1, yerr_1, x_2, y_2, yerr_2, x_3, y_3, xerr_3, yerr_3):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x_1, y_1, yerr_1, x_2, y_2, yerr_2, x_3, y_3, xerr_3, yerr_3)

from schwimmbad import MPIPool

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    pos = soln.x + 1e-4 * np.random.randn(32, 6)
    nwalkers, ndim = pos.shape
    filename = "CPL-Pantheon+CC+GW.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, args=(x_1,y_1,yerr_1,x_2,y_2,yerr_2, x_3, y_3, xerr_3, yerr_3), backend=backend)
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

flat_samples = sampler.get_chain(discard=600, thin=100, flat=True)

from getdist import plots, MCSamples

samples = MCSamples(samples=flat_samples,names = ["h","\Omega_M", "w_0", "w_1", "\Xi_0", "n"], labels = ["h","\Omega_M", "w_0", "w_1", "\Xi_0", "n"])
fig = plots.get_subplot_plotter()
fig.settings.figure_legend_frame = False
fig.settings.alpha_filled_add=0.4
fig.settings.title_limit_fontsize = 14
fig.triangle_plot(samples, ["h","\Omega_M", "w_0", "w_1", "\Xi_0", "n"], filled=True, contour_colors=['red'], title_limit=1)
fig.export("resultsCPL-CC+Pantheon+GW.png")

omega = plots.get_single_plotter()
omega.plot_2d(samples, "h", "\Omega_M", colors=['red'], filled=True)
omega.export("Omega-M-h-LCDM.png")

end= time.time()
print("El tiempo del código fue:",end-start,"segundos")
