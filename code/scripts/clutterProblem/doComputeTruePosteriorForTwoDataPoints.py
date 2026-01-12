
import sys
import argparse
import numpy as np
from scipy.stats import multivariate_normal

import plotly.graph_objects as go


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", help="theta value", type=float,
                        default=3.0)
    parser.add_argument("--a", help="variance of background noise", type=float,
                        default=10.0)
    parser.add_argument("--b", help="prior variance", type=float,
                        default=100.0)
    parser.add_argument("--w", help="mixture coefficient", type=float,
                        default=0.5)
    parser.add_argument("--n_samples", help="number of samples", type=int,
                        default=2)
    parser.add_argument("--x_min", help="minimum value for abscissa",
                        type=float, default=-5.0)
    parser.add_argument("--x_max", help="maximum value for abscissa",
                        type=float, default=10.0)
    parser.add_argument("--x_dt", help="sampling period for abscissa",
                        type=float, default=0.01)
    parser.add_argument("--data_filename_pattern", help="data filename",
                        type=str,
                        default="../../../results/clutter_data_theta{:.2f}_a{:.2f}_w{:.2f}_nSamples{:04d}.npz")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern", type=str,
                        default="../../../figures/truePosterior_theta{:.2f}_a{:.2f}_w{:.2f}_nSamples{:04d}.{{:s}}")
    args = parser.parse_args()

    theta = args.theta
    a = args.a
    b = args.b
    w = args.w
    n_samples = args.n_samples
    x_min = args.x_min
    x_max = args.x_max
    x_dt = args.x_dt
    data_filename = args.data_filename_pattern.format(theta, a, w, n_samples)
    fig_filename_pattern = args.fig_filename_pattern.format(theta, a, w,
                                                            n_samples)

    samples = np.load(data_filename)["samples"]
    samples = [np.array([sample]) for sample in samples]
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

    # Plot
    fig = go.Figure()
    trace = go.Scatter(x=x_dense, y=true_posterior_values)
    fig.add_trace(trace)
    fig.update_xaxes(title=r"$\theta$")
    fig.update_yaxes(title="Posterior Value")

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))
    fig.show()


if __name__ == "__main__":
    main(sys.argv)
