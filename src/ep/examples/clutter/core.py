import numpy as np
from scipy.stats import multivariate_normal


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
    """Returns the site strength :math:`\\rho_n`.

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


