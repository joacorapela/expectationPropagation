
import sys
import argparse
import numpy as np
import scipy.stats
import plotly.graph_objects as go

import clutterUtils


def logLike(theta, w, a, samples):
    log_like = 0.0
    for sample in samples:
        log_like += np.log((1-w) *
                           scipy.stats.norm.pdf(sample, loc=theta, scale=1.0) +
                           w *
                           scipy.stats.norm.pdf(sample, loc=0.0, scale=np.sqrt(a)))
    return log_like


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", help="theta value", type=float,
                        default=3.0)
    parser.add_argument("--a", help="variance of background noise", type=float,
                        default=10.0)
    parser.add_argument("--w", help="mixture coefficient", type=float, default=0.5)
    parser.add_argument("--theta_min", help="minimum theta value", type=float,
                        default=1.0)
    parser.add_argument("--theta_max", help="maximum theta value", type=float,
                        default=5.0)
    parser.add_argument("--delta_theta", help="increment value for theta", type=float,
                        default=0.1)
    parser.add_argument("--data_filename_pattern", help="data filename", type=str,
                        default="../../results/clutter_data_theta{:.2f}_a{:.2f}_w{:.2f}.npz")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", type=str,
                        default="../../figures/logLikeTheta_theta{:.2f}_a{:.2f}_w{:.2f}.{:s}")
    args = parser.parse_args()

    theta = args.theta
    a = args.a
    w = args.w
    theta_min = args.theta_min
    theta_max = args.theta_max
    delta_theta = args.delta_theta
    data_filename = args.data_filename_pattern.format(theta, a, w)
    fig_filename_pattern = args.fig_filename_pattern

    load_res = np.load(data_filename)
    samples = load_res["samples"]

    test_thetas = np.arange(theta_min, theta_max, delta_theta)
    log_likes = np.empty_like(test_thetas)
    for i, test_theta in enumerate(test_thetas):
        log_likes[i] = logLike(theta=test_theta, w=w, a=a, samples=samples)

    fig = go.Figure()
    trace = go.Scatter(x=test_thetas, y=log_likes, mode="lines+markers")
    fig.add_trace(trace)
    fig.add_vline(x=theta)
    fig.update_xaxes(title="theta")
    fig.update_yaxes(title="log likelihood")
    fig.update_layout(showlegend=False)

    fig.write_image(fig_filename_pattern.format(theta, a, w, "png"))
    fig.write_html(fig_filename_pattern.format(theta, a, w, "html"))

    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
