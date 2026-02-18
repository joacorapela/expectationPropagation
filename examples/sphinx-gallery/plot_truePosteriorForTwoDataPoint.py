

"""
True and estimated posteriors
=============================

The code below plots the true and estimated posteriors of a sample of two data
points from the clutter problem.

"""

#%%
# Import requirments
# ------------------

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from IPython.display import display
import plotly.graph_objects as go

import ep.examples.clutter.utils
import ep.examples.clutter.core
import ep.examples.clutter.plot

#%%
# Generate two samples from the clutter model
# -------------------------------------------

theta = 3.0
a = 10.0
b = 100.0
w = 0.5
N = 2
x_min = -5.0
x_max = 10.0
x_dt = 0.01

samples_list = ep.examples.clutter.utils.sample(theta=theta, a=a, w=w, n_samples=N)
samples = [np.array([sample]) for sample in samples_list]

#%%
# Compute true posterior
# ----------------------

D = len(samples[0])

x1 = samples[0]
x2 = samples[1]

sigma2_SS = b / (1 + 2 * b)
mu_SS = sigma2_SS * (x1 + x2)

sigma2_SC = b / (1 + b)
mu_SC = sigma2_SC * x1

sigma2_CS = sigma2_SC
mu_CS = sigma2_CS * x2

sigma2_CC = b
mu_CC = 0.0

pi_SS = ((1 - w)**2 * (1 / (2 * np.pi)**D) * (1 / (1 + 2 * b)**(D / 2)) *
         np.exp((b * np.linalg.norm(x1 + x2)**2 -
                 (1 + 2 * b) * (np.linalg.norm(x1)**2 +
                                np.linalg.norm(x2)**2)) /
                (2 * (1 + 2 * b))))

pi_SC = ((1 - w) * w *
         multivariate_normal(np.zeros(shape=D),
                             a * np.eye(D)).pdf(x2) *
         multivariate_normal(np.zeros(shape=D),
                             (b + 1) * np.eye(D)).pdf(x1))

pi_CS = (w * (1 - w) *
         multivariate_normal(np.zeros(shape=D),
                             a * np.eye(D)).pdf(x1) *
         multivariate_normal(np.zeros(shape=D),
                             (b + 1) * np.eye(D)).pdf(x2))

pi_CC = (w**2 *
         multivariate_normal(np.zeros(shape=D),
                             a * np.eye(D)).pdf(x1) *
         multivariate_normal(np.zeros(shape=D),
                             a * np.eye(D)).pdf(x2))

K = 1.0 / (pi_SS + pi_SC + pi_CS + pi_CC)


def true_posterior(theta):
    answer = K * (pi_SS * multivariate_normal(mu_SS, sigma2_SS *
                                              np.eye(D)).pdf(theta) +
                  pi_SC * multivariate_normal(mu_SC, sigma2_SC *
                                              np.eye(D)).pdf(theta) +
                  pi_CS * multivariate_normal(mu_CS, sigma2_CS *
                                              np.eye(D)).pdf(theta) +
                  pi_CC * multivariate_normal(mu_CC, sigma2_CC *
                                              np.eye(D)).pdf(theta))
    return answer


x_dense = np.arange(x_min, x_max, x_dt)
true_posterior_values = true_posterior(theta=x_dense)


#%%
# Expectation Propagation script
# ------------------------------

num_iter = 10

D = len(samples[0])
m, v, m_f, v_f, s_f = ep.examples.clutter.core.init(b=b, D=D, N=N)
log_evidences = []
snapshots = []

for iter_num in range(num_iter):
    for n in range(N):
        v_cn = ep.examples.clutter.core.get_cavity_var(v=v, v_fn=v_f[n])
        m_cn = ep.examples.clutter.core.get_cavity_mean(m=m, m_fn=m_f[n], v_fn=v_f[n], v_cn=v_cn)
        Z_n = ep.examples.clutter.core.get_zeroth_moment(w=w, a=a, m_cn=m_cn, v_cn=v_cn,
                                x_n=samples[n])
        rho_n = ep.examples.clutter.core.get_site_strength(w=w, a=a, D=D, Z_n=Z_n, x_n=samples[n])
        m = ep.examples.clutter.core.get_q_mean(m_cn=m_cn, v_cn=v_cn, rho_n=rho_n, x_n=samples[n])
        v = ep.examples.clutter.core.get_q_var(m_cn=m_cn, v_cn=v_cn, rho_n=rho_n, x_n=samples[n])
        v_f[n] = ep.examples.clutter.core.get_factor_var(v_cn=v_cn, v=v)
        m_f[n] = ep.examples.clutter.core.get_factor_mean(m_cn=m_cn, v_cn=v_cn, v_fn=v_f[n], m=m)
        s_f[n] = ep.examples.clutter.core.get_factor_scale(Z_n=Z_n, m_fn=m_f[n],
                                  v_fn=v_f[n], m_cn=m_cn, v_cn=v_cn)
    snapshots.append({
        "iter": iter_num,
        "v_cn": v_cn,
        "m_cn": m_cn.copy(),
        "v": v,
        "m": m.copy(),
        "v_fn": v_f[n],
        "m_fn": m_f[n].copy(),
    })
    log_evidence = ep.examples.clutter.core.get_log_evidence(m=m, v=v, m_f=m_f, v_f=v_f, s_f=s_f,
                                    b=b)
    log_evidences.append(log_evidence)

#%%
# Plot EP probability density functions after iteration 0
# -------------------------------------------------------

x_min = -10
x_max = 10
x_dt = 0.1

iter_num = 0
s = snapshots[iter_num]
ep.examples.clutter.plot.plot_pdfs(theta=theta, m_cn=s["m_cn"], v_cn=s["v_cn"],
                                   m=s["m"], v=s["v"], m_fn=s["m_fn"],
                                   v_fn=s["v_fn"], samples=samples[:N],
                                   x_min=x_min, x_max=x_max, x_dt=x_dt,
                                   title=f"Iteration {iter_num}, Factor {N-1}")

#%%
# Plot EP probability density functions after iteration 3
# -------------------------------------------------------

iter_num = 3
s = snapshots[iter_num]
ep.examples.clutter.plot.plot_pdfs(theta=theta, m_cn=s["m_cn"], v_cn=s["v_cn"],
                                   m=s["m"], v=s["v"], m_fn=s["m_fn"],
                                   v_fn=s["v_fn"], samples=samples[:N],
                                   x_min=x_min, x_max=x_max, x_dt=x_dt,
                                   title=f"Iteration {iter_num}, Factor {N-1}")

#%%
# Plot EP probability density functions after iteration 6
# -------------------------------------------------------

iter_num = 6
s = snapshots[iter_num]
ep.examples.clutter.plot.plot_pdfs(theta=theta, m_cn=s["m_cn"], v_cn=s["v_cn"],
                                   m=s["m"], v=s["v"], m_fn=s["m_fn"],
                                   v_fn=s["v_fn"], samples=samples[:N],
                                   x_min=x_min, x_max=x_max, x_dt=x_dt,
                                   title=f"Iteration {iter_num}, Factor {N-1}")

#%%
# Plot EP probability density functions after iteration 9
# -------------------------------------------------------

iter_num = 9
s = snapshots[iter_num]
ep.examples.clutter.plot.plot_pdfs(theta=theta, m_cn=s["m_cn"], v_cn=s["v_cn"],
                                   m=s["m"], v=s["v"], m_fn=s["m_fn"],
                                   v_fn=s["v_fn"], samples=samples[:N],
                                   x_min=x_min, x_max=x_max, x_dt=x_dt,
                                   title=f"Iteration {iter_num}, Factor {N-1}")


#%%
# Plot log evidences
# ------------------

ep.examples.clutter.plot.plot_log_evidences(log_evidences)
