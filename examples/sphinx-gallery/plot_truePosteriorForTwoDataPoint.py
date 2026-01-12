

"""
Plot true and estimated posteriors for a sample with two data points
====================================================================

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

#%%
# Function to draw samples for the clutter problem
# ------------------------------------------------

def sample(theta, a, w, n_samples):
    uniform_samples = np.random.uniform(size=n_samples)
    answer = np.empty(shape=n_samples)
    for i in range(n_samples):
        if uniform_samples[i] < w:
            answer[i] = np.random.normal(loc=0, scale=np.sqrt(a))
        else:
            answer[i] = np.random.normal(loc=theta, scale=1.0)
    return answer

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

samples_list = sample(theta=theta, a=a, w=w, n_samples=N)
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
# Functions for the Expectation Propagation algorithm
# ---------------------------------------------------

def init(b, D, N):
    m_f = np.zeros(shape=(N, D))
    v_f = np.ones(shape=N)*np.inf
    s_f = np.ones(shape=N)
    m = np.array([0.0])
    v = b
    return m, v, m_f, v_f, s_f


def get_cavity_var(v, v_fn):
    """Returns the cavity variance :math:`v^{\\setduf n}`.

    : param v: approximate posterior q variance
    : type v: float
    : param v_fn: factor variance
    : type v_fn: float
    : return: cavity variance :math:`v^{\\setduf n}`
    : rtype: float
    """
    if np.isposinf(v_fn):
        v_c = v
    else:
        v_c = v_fn * v / (v_fn - v)
    return v_c


def get_cavity_mean(m, m_fn, v_fn, v_cn):
    """Returns the cavity mean :math:`m^{\\setduf n}`.

    : param m: approximate posterior q mean
    : type m: array of length D
    : param m_fn: factor mean
    : type m_fn: array of length D
    : param v_fn: factor variance
    : type v_fn: float
    : param v_cn: cavity variance
    : type v_cn: float
    : return: cavity mean :math:`m^{\\setduf b}`
    : rtype: array of length D
    """
    m_c = m + v_cn / v_fn * (m - m_fn)
    return m_c


def get_zeroth_moment(w, a, m_cn, v_cn, x_n):
    """Returns the zeroth moment :math:`Z_n`.

    : param w: weight in the clutter problem likelihood
    : type w: float
    : param a: signal variance in clutter problem
    : type a: float
    : param m_c: cavity mean
    : type m: array of length D
    : param v_c: cavity variance
    : type v_c: float
    : return: zeorth moment :math:`Z_n`
    : rtype: float
    """
    D = len(m_cn)
    Z_n = ((1 - w) * multivariate_normal(
        mean=m_cn, cov=(v_cn + 1) * np.eye(D)).pdf(x=x_n) +
        w * multivariate_normal(mean=np.zeros(D),
                                cov=a * np.eye(D)).pdf(x=x_n))
    return Z_n


def get_site_strength(w, a, D, Z_n, x_n):
    """Returns the zeroth moment :math:`\rho_n`.

    : param m: approximate posterior q mean
    : type m: float
    : param m_fn: factor mean
    : type m_fn: float
    : param v_fn: factor variance
    : type v_fn: float
    : param v_cn: cavity variance
    : type v_cn: float
    : return: site strength :math:`\\rho_n`
    : rtype: float
    """
    rho_n = 1 - w / Z_n * multivariate_normal(mean=np.zeros(D),
                                              cov=a * np.eye(D)).pdf(x_n)
    return rho_n


def get_q_mean(m_cn, v_cn, rho_n, x_n):
    """Returns the mean of the approximate distribution :math:`q`.

    : param m_cn: cavity mean
    : type m_cn: array of length D
    : param v_cn: cavity variance
    : type v_cn: float
    : param rho_n: site strength
    : type rho_n: float
    : param x_n: sample
    : type x_n: array of dimension D
    : return: approximate posterior distribution mean
    : rtype: array of length D
    """
    m = m_cn + v_cn / (1 + v_cn) * rho_n * (x_n - m_cn)
    return m


def get_q_var(m_cn, v_cn, rho_n, x_n):
    """Returns the variance of the approximate distribution :math:`q`.

    : param m_cn: cavity mean
    : type m_cn: array of length D
    : param v_cn: cavity variance
    : type v_cn: float
    : param rho_n: site strength
    : type rho_n: float
    : param x_n: sample
    : type x_n: array of dimension D
    : return: approximate posterior variance
    : rtype: float
    """
    D = len(m_cn)
    v = (-(v_cn)**2 * rho_n / (1 + v_cn) + rho_n * (1 - rho_n) * (v_cn)**2 *
         np.linalg.norm(x_n - m_cn)**2 / (D * (1 + v_cn)**2) + v_cn)
    return v


def get_factor_var(v_cn, v, tol=1e-9):
    """Returns the factor variance.

    : param v_cn: cavity variance
    : type v_cn: float
    : param v: approximate posterior distribution variance
    : rtype: float
    : return: factor variance
    : rtype: float
    """

    if np.abs(v_cn - v) < tol:
        # v_n = np.inf
        v_n = 1e10
    else:
        v_n = v_cn * v / (v_cn - v)
    if np.isinf(v_n):
        raise ValueError("v_n is infty")
    return v_n


def get_factor_mean(m_cn, v_cn, v_fn, m, tol=1e-9):
    """Returns the factor mean.

    : param m_cn: cavity mean
    : type m_cn: array of dimension D
    : param v_cn: cavity variance
    : type v_cn: float
    : param v_fn: factor variance
    : rtype: float
    : param m: approximate posterior
    : type m: array of dimension D
    : return: factor variance
    : rtype: float
    """

    # inf x 0 problem
    if v_fn == np.inf and (v_fn == 0 or any(m - m_cn) < tol):
        m_n = m_cn
    else:
        m_n = m_cn + (v_fn + v_cn) / v_cn * (m - m_cn)
    return m_n


def get_factor_scale(Z_n, m_fn, v_fn, m_cn, v_cn):
    """Returns the factor scale.

    : param Z_n: zeroth moment
    : type Z_n: float
    : param m_fn: factor mean
    : type m_fn: array of length D
    : param v_fn: factor variance
    : type v_fn: float
    : param m_cn: cavity mean
    : type m_cn: array of dimenion D
    : param v_cn: cavity variance
    : type v_cn: float
    : return: factor scale
    : rtype: float
    """

    D = len(m_cn)
    var = v_cn + v_fn
    if var > 0:
        pdf_value = multivariate_normal(m_cn, var * np.eye(D)).pdf(m_fn)
    else:
        pdf_value = -multivariate_normal(m_cn, -var * np.eye(D)).pdf(m_fn)
    # pdf_value = (1.0 / (2 * np.pi * var)**(D/2.0) *
    #              np.exp(-1.0 / (2 * var) * np.linalg.norm(m_fn - m_cn)**2))
    # if np.isnan(pdf_value):
    #     raise ValueError("NaN pdf_value in get_factor_scale")
    sn = Z_n / pdf_value
    return sn


def get_log_evidence(m, v, m_f, v_f, s_f, b):
    """Returns the model evidence

    : param m: approximate posterior mean
    : type m: array of dimension D
    : param v: approximate posterior variance
    : type v: float
    : param m_f: factor means
    : type m_f: N \\times D array
    : param v_f: factor variance
    : type v_f: array of dimension N
    : param s_f: factor scale
    : type s_f: array of dimension N
    : param b: prior variance
    : type b: float
    : return: model evidence
    : rtype: float
    """

    D = len(m_f[0])
    N = len(s_f)

    constants_terms = D / 2 * (np.log(v) - np.log(b) - N * np.log(2*np.pi))

    log_s_term = (np.sum(np.log(np.abs(s_f))) -
                  D / 2 * np.sum(np.log(np.abs(v_f))))

    exp_terms = 0.5 * (np.sum(m**2) / v - np.sum(np.sum(m_f**2, axis=1) / v_f))

    log_evidence = constants_terms + log_s_term + exp_terms

    return log_evidence


#%%
# Plotting functions
# ------------------

def plot_pdfs(theta, m_cn, v_cn, m, v, m_fn, v_fn, samples,
              x_min=-10.0, x_max=20.0, x_dt=0.1, title=""):
    samples = [sample[0] for sample in samples]
    x = np.arange(x_min, x_max, x_dt)
    pdf_c = norm.pdf(x=x, loc=m_cn[0], scale=np.sqrt(v_cn))
    pdf_q = norm.pdf(x=x, loc=m[0], scale=np.sqrt(v))
    pdf_fn = norm.pdf(x=x, loc=m_fn[0], scale=np.sqrt(v_fn))
    true_posterior_samples = true_posterior(theta=x)
    fig = go.Figure()
    fig.add_vline(theta)
    trace = go.Scatter(x=samples, y=np.zeros(len(samples)), mode="markers",
                       name="samples")
    fig.add_trace(trace)
    trace = go.Scatter(x=x, y=pdf_c, mode="lines", name="cavity")
    fig.add_trace(trace)
    trace = go.Scatter(x=x, y=pdf_q, mode="lines", name="posterior")
    fig.add_trace(trace)
    trace = go.Scatter(x=x, y=pdf_fn, mode="lines", name="factor")
    fig.add_trace(trace)
    trace = go.Scatter(x=x, y=true_posterior_samples, mode="lines",
                       name="true posterior")
    fig.add_trace(trace)
    fig.update_xaxes(title=r"$\theta$")
    fig.update_yaxes(title="density")
    fig.update_layout(title=title)
    return fig


def plot_log_evidences(log_evidences):

    iter_nos = np.arange(len(log_evidences))
    fig = go.Figure()
    trace = go.Scatter(x=iter_nos, y=log_evidences, mode="lines+markers")
    fig.add_trace(trace)
    fig.update_xaxes(title="Iteration Number")
    fig.update_yaxes(title="Log Evidence")
    return fig


#%%
# Expectation Propagation script
# ------------------------------

num_iter = 10

D = len(samples[0])
m, v, m_f, v_f, s_f = init(b=b, D=D, N=N)
log_evidences = []
snapshots = []

for iter_num in range(num_iter):
    for n in range(N):
        v_cn = get_cavity_var(v=v, v_fn=v_f[n])
        m_cn = get_cavity_mean(m=m, m_fn=m_f[n], v_fn=v_f[n], v_cn=v_cn)
        Z_n = get_zeroth_moment(w=w, a=a, m_cn=m_cn, v_cn=v_cn,
                                x_n=samples[n])
        rho_n = get_site_strength(w=w, a=a, D=D, Z_n=Z_n, x_n=samples[n])
        m = get_q_mean(m_cn=m_cn, v_cn=v_cn, rho_n=rho_n, x_n=samples[n])
        v = get_q_var(m_cn=m_cn, v_cn=v_cn, rho_n=rho_n, x_n=samples[n])
        v_f[n] = get_factor_var(v_cn=v_cn, v=v)
        m_f[n] = get_factor_mean(m_cn=m_cn, v_cn=v_cn, v_fn=v_f[n], m=m)
        s_f[n] = get_factor_scale(Z_n=Z_n, m_fn=m_f[n],
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
    log_evidence = get_log_evidence(m=m, v=v, m_f=m_f, v_f=v_f, s_f=s_f,
                                    b=b)
    log_evidences.append(log_evidence)

#%%
# Plot EP probability density functions after iteration 0
# -------------------------------------------------------

iter_num = 0
s = snapshots[iter_num]
plot_pdfs(theta=theta, m_cn=s["m_cn"], v_cn=s["v_cn"], m=s["m"], v=s["v"],
          m_fn=s["m_fn"], v_fn=s["v_fn"], samples=samples[:N],
          title=f"Iteration {iter_num}, Factor {N-1}")

#%%
# Plot EP probability density functions after iteration 3
# -------------------------------------------------------

iter_num = 3
s = snapshots[iter_num]
plot_pdfs(theta=theta, m_cn=s["m_cn"], v_cn=s["v_cn"], m=s["m"], v=s["v"],
          m_fn=s["m_fn"], v_fn=s["v_fn"], samples=samples[:N],
          title=f"Iteration {iter_num}, Factor {N-1}")

#%%
# Plot EP probability density functions after iteration 6
# -------------------------------------------------------

iter_num = 6
s = snapshots[iter_num]
plot_pdfs(theta=theta, m_cn=s["m_cn"], v_cn=s["v_cn"], m=s["m"], v=s["v"],
          m_fn=s["m_fn"], v_fn=s["v_fn"], samples=samples[:N],
          title=f"Iteration {iter_num}, Factor {N-1}")

#%%
# Plot EP probability density functions after iteration 9
# -------------------------------------------------------

iter_num = 9
s = snapshots[iter_num]
plot_pdfs(theta=theta, m_cn=s["m_cn"], v_cn=s["v_cn"], m=s["m"], v=s["v"],
          m_fn=s["m_fn"], v_fn=s["v_fn"], samples=samples[:N],
          title=f"Iteration {iter_num}, Factor {N-1}")


#%%
# Plot log evidences
# ------------------

plot_log_evidences(log_evidences)
