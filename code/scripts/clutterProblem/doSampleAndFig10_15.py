
import sys
import argparse
import numpy as np
import scipy.stats
import plotly.graph_objects as go

import clutterUtils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", help="theta value", type=float,
                        default=3.0)
    parser.add_argument("--a", help="variance of background noise", type=float,
                        default=10.0)
    parser.add_argument("--w", help="mixture coefficient", type=float,
                        default=0.5)
    parser.add_argument("--n_samples", help="number of samples", type=int,
                        default=30)
    parser.add_argument("--x_min", help="minimum value for abscissa",
                        type=float, default=-5.0)
    parser.add_argument("--x_max", help="maximum value for abscissa",
                        type=float, default=10.0)
    parser.add_argument("--x_dt", help="sampling period for abscissa",
                        type=float, default=0.01)
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern", type=str,
                        default="../../../figures/fig_10_15.{:s}")
    parser.add_argument("--data_filename_pattern", help="data filename",
                        type=str,
                        default="../../../results/clutter_data_theta{:.2f}_a{:.2f}_w{:.2f}_nSamples{:04d}.npz")
    args = parser.parse_args()

    theta = args.theta
    a = args.a
    w = args.w
    n_samples = args.n_samples
    x_min = args.x_min
    x_max = args.x_max
    x_dt = args.x_dt
    fig_filename_pattern = args.fig_filename_pattern
    data_filename = args.data_filename_pattern.format(theta, a, w, n_samples)

    samples = clutterUtils.sample(theta=theta, a=a, w=w, n_samples=n_samples)

    x_dense = np.arange(x_min, x_max, x_dt)
    signal_pdf_values = scipy.stats.norm.pdf(x_dense, loc=theta, scale=1.0)
    noise_pdf_values = scipy.stats.norm.pdf(x_dense, loc=0, scale=np.sqrt(a))

    np.savez(data_filename, samples=samples)
    print(f"Samples saved to {data_filename}")

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
    fig.update_xaxes(title="x")
    fig.update_layout(showlegend=False)

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
