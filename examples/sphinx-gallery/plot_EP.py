

"""
Clutter example for expectation propagation
===========================================

The code below uses the Expectation Propagation algorithm to solve the Clutter
problem, Bishop et al., 2006, section 10.7.1

"""

#%%
# Import requirments
# ------------------

import numpy as np
from scipy.stats import norm
from IPython.display import display
import plotly.graph_objects as go

import ep.examples.clutter.utils
import ep.examples.clutter.core
import ep.examples.clutter.plot


#%%
# Sample from the clutter model
# -----------------------------

theta = 3.0
a = 10.0
b = 100.0
w = 0.5
N = 30
num_iter = 20

samples = ep.examples.clutter.utils.sample(theta=theta, a=a, w=w, n_samples=N)

#%%
# Plot sampled data
# -----------------

x_min = -10
x_max = 10
x_dt = 0.1

x_dense = np.arange(x_min, x_max, x_dt)
signal_pdf_values = norm.pdf(x_dense, loc=theta, scale=1.0)
noise_pdf_values = norm.pdf(x_dense, loc=0, scale=np.sqrt(a))

fig = go.Figure()
trace = go.Scatter(x=samples, y=np.zeros(shape=samples.shape),
                   mode="markers", marker=dict(symbol="x", color="black"))
fig.add_trace(trace)
trace = go.Scatter(x=x_dense, y=signal_pdf_values, mode="lines",
                   line=dict(color="green"))
fig.add_trace(trace)
trace = go.Scatter(x=x_dense, y=noise_pdf_values, mode="lines",
                   line=dict(color="red"))
fig.add_trace(trace)
fig.update_xaxes(title=r"$\theta$")
fig.update_layout(showlegend=False)
fig

#%%
# Expectation Propagation script
# ------------------------------

samples = [np.array([sample]) for sample in samples]
D = len(samples[0])
m, v, m_f, v_f, s_f = ep.examples.clutter.core.init(b=b, D=D, N=N)
log_evidences = []
snapshots = []

for iter_num in range(num_iter):
    for n in range(N):
        v_cn = ep.examples.clutter.core.get_cavity_var(v=v, v_fn=v_f[n])
        m_cn = ep.examples.clutter.core.get_cavity_mean(m=m, m_fn=m_f[n], v_fn=v_f[n],
                                                v_cn=v_cn)
        Z_n = ep.examples.clutter.core.get_zeroth_moment(w=w, a=a, m_cn=m_cn,
                                                         v_cn=v_cn,
                                                         x_n=samples[n])
        rho_n = ep.examples.clutter.core.get_site_strength(w=w, a=a, D=D,
                                                           Z_n=Z_n,
                                                           x_n=samples[n])
        m = ep.examples.clutter.core.get_q_mean(m_cn=m_cn, v_cn=v_cn,
                                                rho_n=rho_n, x_n=samples[n])
        v = ep.examples.clutter.core.get_q_var(m_cn=m_cn, v_cn=v_cn,
                                               rho_n=rho_n, x_n=samples[n])
        v_f[n] = ep.examples.clutter.core.get_factor_var(v_cn=v_cn, v=v)
        m_f[n] = ep.examples.clutter.core.get_factor_mean(m_cn=m_cn, v_cn=v_cn,
                                                          v_fn=v_f[n], m=m)
        s_f[n] = ep.examples.clutter.core.get_factor_scale(Z_n=Z_n, m_fn=m_f[n],
                                                           v_fn=v_f[n],
                                                           m_cn=m_cn, v_cn=v_cn)
    snapshots.append({
        "iter": iter_num,
        "v_cn": v_cn,
        "m_cn": m_cn.copy(),
        "v": v,
        "m": m.copy(),
        "v_fn": v_f[n],
        "m_fn": m_f[n].copy(),
    })
    log_evidence = ep.examples.clutter.core.get_log_evidence(m=m, v=v, m_f=m_f,
                                                    v_f=v_f, s_f=s_f, b=b)
    log_evidences.append(log_evidence)

#%%
# Plot EP probability density functions after iteration 0
# -------------------------------------------------------

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
# Plot EP probability density functions after iteration 19
# -------------------------------------------------------

iter_num = 19
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
