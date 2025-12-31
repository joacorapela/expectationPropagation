
import sys
import argparse
import numpy as np
from scipy.stats import multivariate_normal


def init(b, N):
    m_f = np.zeros(shape=N)
    v_f = np.ones(shape=N) * np.inf
    s_f = np.zeros(shape=N)
    m = 0.0
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


def get_zeroth_moment(w, a, m_cn, v_cn):
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
        mean=m_cn, cov=(v_cn + 1) * np.eye(D) +
        w * multivariate_normal(mean=np.zeros(D),
                                cov=a * np.eye(D))))
    return Z_n


def get_site_strength(w, a, D, Z_n):
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
                                              cov=a * np.diag(D))
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


def get_factor_var(v_cn, v):
    """Returns the factor variance.

    : param v_cn: cavity variance
    : type v_cn: float
    : param v: approximate posterior distribution variance
    : rtype: float
    : return: factor variance
    : rtype: float
    """

    v_n = v_cn * v / (v_cn - v)
    return v_n


def get_factor_mean(m_cn, v_cn, v_fn, m):
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

    m = m_cn + (v_fn + v_cn) / v_cn * (m - m_cn)
    return m


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
    sn = Z_n / multivariate_normal(mean=m_cn, cov=(1 + v_cn) * np.eye(D))
    return sn


def get_evidence(m, v, m_f, v_f, s_f, b):
    """Returns the model evidence

    : param m: approximate posterior mean
    : type m: array of dimension D
    : param v: approximate posterior variance
    : type v: float
    : param m_f: factor means
    : type m_f: array of arrays of length D
    : param v_f: factor variance
    : type v_f: array of floats
    : param s_f: factor scale
    : type s_f: array of floats
    : param b: prior variance
    : type b: float
    : return: model evidence
    : rtype: float
    """

    D = len(m_f[0])
    N = len(s_f)

    s_prod = 1.0
    m_sum = 0.0
    for n in range(N):
        s_prod *= s_f[n] / (2 * np.pi * v_f[n])**{D/2}
        m_sum += np.linalg.norm(m_f[n])**2 / v_f[n]
    evidence = ((v / b)**(D/2) * s_prod *
                multivariate_normal(m, v * np.eye(D)) *
                np.exp(0.5 * (np.linalg.norm(m)**2 / v - m_sum)))
    return evidence


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", help="theta value", type=float,
                        default=3.0)
    parser.add_argument("--a", help="variance of background noise", type=float,
                        default=10.0)
    parser.add_argument("--b", help="prior variance", type=float, default=1.0)
    parser.add_argument("--w", help="mixture coefficient", type=float,
                        default=0.5)
    parser.add_argument("--n_samples", help="number of samples", type=int,
                        default=100)
    parser.add_argument("--tol", help="convergence tolverance", type=float,
                        default=1e-4)
    parser.add_argument("--data_filename_pattern", help="data filename",
                        type=str,
                        default="../../../results/clutter_data_theta{:.2f}_a{:.2f}_w{:.2f}_nSamples{:04d}.npz")
    args = parser.parse_args()

    theta = args.theta
    a = args.a
    b = args.b
    w = args.w
    n_samples = args.n_samples
    tol = args.tol
    data_filename = args.data_filename_pattern.format(theta, a, w, n_samples)

    samples = np.load(data_filename)
    N = len(samples)
    D = len(samples[0])
    m, v, m_f, v_f, s_f = init(b=b, N=N)
    evidence_cur = -np.inf
    evidence_diff = np.inf
    while evidence_diff > tol:
        for n in range(n_samples):
            v_cn = get_cavity_var(v=v, v_fn=v_f[n])
            m_cn = get_cavity_mean(m=m, v=v, m_fn=m_f[n], v_fn=v_f[n],
                                   v_cn=v_cn)
            Z_n = get_zeroth_moment(w=w, a=a, m_cn=m_cn, v_cn=v_cn)
            rho_n = get_site_strength(w=w, a=a, D=D, Z_n=Z_n)
            m = get_q_mean(m_cn=m_cn, v_cn=v_cn, rho_n=rho_n, x_n=samples[n])
            v = get_q_var(m_cn=m_cn, v_cn=v_cn, rho_n=rho_n, x_n=samples[n])
            v_f[n] = get_factor_var(m_cn=m_cn, v=v)
            m_f[n] = get_factor_mean(m_cn=m_cn, v_cn=v_cn, v_fn=v_f[n], m=m)
            s_f[n] = get_factor_scale(Z_n=Z_n, m_fn=m_f[n], v_fn=v_f[n],
                                      m_cn=m_cn, v_cn=v_cn)
        evidence_prev = evidence_cur
        evidence_cur = get_evidence(m=m, v=v, m_f=m_f, v_f=v_f, s_f=s_f, b=b)
        evidence_diff = evidence_cur - evidence_prev

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
