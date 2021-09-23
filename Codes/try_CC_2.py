import numpy as np
import matplotlib.pyplot as plt
import emcee

data = np.array([[0, 0.07,0.1,0.12,0.17,0.179,0.199,0.2,0.27,0.28,
            0.352, 0.3802, 0.4, 0.4004, 0.4247, 0.4497, 0.47,
            0.4783, 0.48, 0.593, 0.68, 0.781, 0.875, 0.88, 0.9,
            1.037, 1.3, 1.363, 1.43, 1.53, 1.75, 1.965, 0.24, 0.3, 0.31,
            0.35, 0.36, 0.38, 0.43, 0.44, 0.51, 0.52, 0.56, 0.57, 0.59,
            0.6, 0.61, 0.64, 0.73], [73.24, 69.0, 69.0,
            68.6, 83.0, 75.0, 75.0, 72.9, 77.0, 88.8, 83.0, 83.0, 95.0,
            77.0, 87.1, 92.8, 89.0, 80.9, 97.0, 104.0, 92.0, 105.0, 125.0,
            90.0, 117.0, 154.0, 168.0, 160.0, 177.0, 140.0, 202.0, 186.5,
            79.69, 81.7, 78.17, 82.7, 79.93, 81.5, 86.45, 82.6, 90.4, 94.35,
            93.33, 92.9, 98.48, 87.9, 97.3, 98.82, 97.3], [1.74,
            19.6, 12.0, 26.2,8.0,4.0,5.0,29.6,14.0,36.6,14.0,
            13.5, 17.0, 10.2, 11.2, 12.9, 49.6, 9.0, 62.0, 13.0,
            8.0, 12.0, 17.0, 40.0, 23.0, 20.0, 17.0, 33.6, 18.0,
            14.0, 40.0, 50.4, 2.65, 6.22, 4.74, 8.4, 3.39, 1.9, 3.68, 7.8,
            1.9, 2.65, 2.32, 7.8, 3.19, 6.1, 2.1, 2.99, 7]])

z, Hz, Hz_err = data [ :, data[0].argsort()]

def dy_H(y_H, y_R, z):
    dery_H = -1/(1+z) * ( 1/3. * y_R - 4 * y_H )
    return dery_H

def dy_R(y_H, y_R, z, FR, F, f, m2, h, Omega_M, Lambda, R, b):
    dery_R = -1/(1+z)
    dery_R1 = 9*np.power(1+z,3)
    dery_R2 = -(1/(y_H+np.power(1+z,3)))*(1/(m2(h, Omega_M)*FR(Lambda, R, b, h, Omega_M, y_R, m2, z)))
    dery_R2_1 = (y_H+1/6.*f(Lambda, R, b, h, Omega_M, y_R, m2, z)/m2(h, Omega_M))
    dery_R2_2 = -F(Lambda, R, b, h, Omega_M, y_R, m2, z)*(1/6.*y_R-y_H-1/2.*np.power(1+z,3))
    return dery_R * (dery_R1 + dery_R2*(dery_R2_1+dery_R2_2))

def m2(h, Omega_M):
    return 10013.24056 * np.power(h,2) * Omega_M

def Lambda(h, Omega_M):
    return 30000 * np.power(h,2) * (1-Omega_M)

def R(y_R,m2,z, h, Omega_M):
    return m2(h, Omega_M)*(y_R+3*np.power(1+z,3))

def f(Lambda, R, b, h, Omega_M, y_R, m2, z):
    if R(y_R,m2,z, h, Omega_M)/(Lambda(h, Omega_M)*b) > 10.53:
        return 0
    else: return -2*Lambda(h, Omega_M) * (1 - np.exp(-R(y_R,m2,z, h, Omega_M)/(Lambda(h, Omega_M)*b)))

def F(Lambda, R, b, h, Omega_M, y_R, m2, z):
    if R(y_R,m2,z, h, Omega_M)/(Lambda(h, Omega_M)*b) > 10.53:
        return -2/b
    else: return -2/b * np.exp(-R(y_R,m2,z, h, Omega_M)/(Lambda(h, Omega_M)*b))

def FR(Lambda, R, b, h, Omega_M, y_R, m2, z):
    if R(y_R,m2,z, h, Omega_M)/(Lambda(h, Omega_M)*b) > 10.53:
        return 2/(Lambda(h, Omega_M)*np.power(b,2))
    else: return 2/(Lambda(h, Omega_M)*np.power(b,2)) * np.exp(-R(y_R,m2,z, h, Omega_M)/(Lambda(h, Omega_M)*b))

def runge_kutta_4th_order(z, y_H_0, y_R_0, FR, F, f, m2, h, Omega_M, R, b):
    y_H = y_H_0
    y_R = y_R_0
    y_H_res = np.zeros(z.shape)
    y_R_res = np.zeros(z.shape)
    y_H_res[0] = y_H
    y_R_res[0] = y_R
    for i in range(len(z)-1):
        h = z[i+1]-z[i]
        k1=h*dy_H(y_H, y_R, z[i])
        l1=h*dy_R(y_H, y_R, z[i], FR, F, f, m2, h, Omega_M, Lambda, R, b)
        k2=h*dy_H(y_H+0.5*k1, y_R+0.5*l1, z[i]+0.5*h)
        l2=h*dy_R(y_H+0.5*k1, y_R+0.5*l1, z[i]+0.5*h, FR, F, f, m2, h, Omega_M, Lambda, R, b)
        k3=h*dy_H(y_H+0.5*k2, y_R+0.5*l2, z[i]+0.5*h)
        l3=h*dy_R(y_H+0.5*k2, y_R+0.5*l2, z[i]+0.5*h, FR, F, f, m2, h, Omega_M, Lambda, R, b)
        k4=h*dy_H(y_H+k3, y_R+l3, z[i]+h)
        l4=h*dy_R(y_H+k3, y_R+l3, z[i]+h, FR, F, f, m2, h, Omega_M, Lambda, R, b)
        y_H += 1/6. * (k1+2*k2+2*k3+k4)
        y_R += 1/6. * (l1+2*l2+2*l3+l4)
        y_H_res[i+1] = y_H
        y_R_res[i+1] = y_R
    return [y_H_res, y_R_res]

def chi_squared(theta, z, Hz, Hz_err):
    h, Omega_M, b = theta
    y_H_0 = (10000*np.power(h,2))/m2(h, Omega_M) - 1
    y_R_0 = (84600*np.power(h,2))/m2(h, Omega_M) - 3
    y_H_sol = runge_kutta_4th_order(z, y_H_0, y_R_0, FR, F, f, m2, h, Omega_M, R, b)[0]
    Hubble_param = np.sqrt(m2(h, Omega_M) * (y_H_sol + np.power(1+z,3)))
    sigma2 = np.power(Hz_err, 2)
    return np.sum(np.power(Hubble_param-Hz,2)/sigma2)
"""
def log_likelihood(theta, z, Hz, Hz_err):
    h, Omega_M, b = theta
    y_H_0 = (10000*np.power(h,2))/m2(h, Omega_M) - 1
    y_R_0 = (84600*np.power(h,2))/m2(h, Omega_M) - 3
    y_H_sol , y_R_sol = runge_kutta_4th_order(z, y_H_0, y_R_0, FR, F, f, m2, h, Omega_M, R, b)
    Hubble_param = np.sqrt(m2(h, Omega_M) * (y_H_sol + np.power(1+z,3)))
    sigma2 = np.power(Hz_err, 2)
    return - 0.5 * np.sum(np.power(Hubble_param-Hz,2)/sigma2 + np.log(sigma2))"""

from scipy.optimize import minimize

nll = lambda *args: chi_squared(*args)
initial = np.array([0.7, 0.3, 0.5])
soln = minimize(nll, initial, bounds=((0.05,0.95),(0.05,0.95), (0.05,0.95)), args=(z, Hz, Hz_err))
h_ml, Omega_M_ml, b_ml = soln.x

print("Maximum likelihood estimates:")
print("h = {0:.5f}".format(h_ml))
print("Omega Matter = {0:.5f}".format(Omega_M_ml))
print("b = {0:.5f}".format(b_ml))

theta_ml = h_ml, Omega_M_ml, b_ml
print(chi_squared(theta_ml, z, Hz, Hz_err))

y_H_0 = (10000*np.power(h_ml,2))/m2(h_ml, Omega_M_ml) - 1
y_R_0 = (84600*np.power(h_ml,2))/m2(h_ml, Omega_M_ml) - 3
y_H_sol = runge_kutta_4th_order(z, y_H_0, y_R_0, FR, F, f, m2, h_ml, Omega_M_ml, R, b_ml)[0]
Hubble_param = np.sqrt(m2(h_ml, Omega_M_ml) * (y_H_sol + np.power(1+z,3)))

fig = plt.errorbar(z,Hz,Hz_err, fmt='o', color='red', label='Data')
plt.plot(z, Hubble_param, color='green', label="Model" )
plt.legend(fontsize=14)
plt.xlabel('$z$')
plt.ylabel('$H(z)$')
plt.legend(loc='upper left')
plt.show()
"""
def log_prior(theta):
    h, Omega_M, b = theta
    if 0 < Omega_M < 1 and 0 < h < 0.85 and 0 < b < 1:
        return 0.0
    return -np.inf

def log_probability(theta, z, Hz, Hz_err):
    lp = log_prior(theta)
    if not np.isfinite(lp+ log_likelihood(theta, z, Hz, Hz_err)):
        return -np.inf
    return lp+ log_likelihood(theta, z, Hz, Hz_err)

from multiprocessing import Pool
import os

os.environ["OMP_NUM_THREADS"] = "1"

with Pool() as pool:
    pos = soln.x + 1e-4 * np.random.randn(32, 3)
    nwalkers, ndim = pos.shape
    filename = "try.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(z,Hz,Hz_err), backend=backend, pool=pool)
    sampler.run_mcmc(pos, 30000, progress=True)

fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["$h$", "$\Omega_M$","$b$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("Step number")

tau = sampler.get_autocorr_time()
print(tau)

flat_samples = sampler.get_chain(discard=500, thin=70, flat=True)

plt.show()

from getdist import plots, MCSamples

samples = MCSamples(samples=flat_samples,names = ["h","\Omega_M","b"] , labels = ["h","\Omega_M","b"])
fig = plots.get_subplot_plotter()
fig.settings.figure_legend_frame = False
fig.settings.alpha_filled_add=0.4
fig.settings.title_limit_fontsize = 14
fig.triangle_plot(samples, ["h","\Omega_M","b"], filled=True, contour_colors=['red'], title_limit=1)
fig.export("try.png")"""
