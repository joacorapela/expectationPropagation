
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go


def plot_pdfs(theta, m_cn, v_cn, m, v, m_fn, v_fn, samples,
              x_min, x_max, x_dt, title="",
              true_posterior_func=None):
    samples = [sample[0] for sample in samples]
    x = np.arange(x_min, x_max, x_dt)
    pdf_c = norm.pdf(x=x, loc=m_cn[0], scale=np.sqrt(v_cn))
    pdf_q = norm.pdf(x=x, loc=m[0], scale=np.sqrt(v))
    pdf_fn = norm.pdf(x=x, loc=m_fn[0], scale=np.sqrt(v_fn))

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
    if true_posterior_func is not None:
        true_posterior_samples = true_posterior_func(theta=x)
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


