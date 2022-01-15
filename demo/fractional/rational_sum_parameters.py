import numpy as np

def bp_sum(lmbda, k, s):
    M = np.ceil((np.pi**2)/(4.*s*k**2))
    N = np.ceil((np.pi**2)/(4.*(1.-s)*k**2))

    constant = (2.*np.sin(np.pi*s)*k)/np.pi

    ls = np.arange(-M, N+1, 1)
    num_param_pbms = len(ls)
    q = 0.
    c_1s = []
    c_2s = []
    weights = []
    for l in ls:
        c_1 = np.exp(2.*k*l)
        c_1s.append(c_1)
        c_2 = 1.
        c_2s.append(c_2)
        weight = np.exp(2.*s*k*l)
        weights.append(weight)
        q += weight/(c_2+c_1*lmbda)
    q *= constant
    return q, c_1s, c_2s, weights, constant

'''
rational_param is based on the assumption that the rational approximation error
is the largest for lmbda = lmbda_0. Numerical evidence show that this
assumption seems true, but there is no mathematical proof available to our
knowledge.

Parameters:
    tol: tolerance for the finite element discretization error
    s: fractional power
    lmbda_0: lower bound of the spectrum
Returns:
    coefficients of the rational sum such that rational sum discr. error < tol * 1e-3
'''
def rational_param(tol, s, lmbda_0, l2_norm_data):
    tol_rs = tol * 1e-3 * l2_norm_data
    ks = np.flip(np.arange(1e-2, 1., step=0.01))
    for k in ks:
        q, c_1s, c_2s, weights, constant = bp_sum(lmbda_0, k, s)
        diff = np.abs(lmbda_0 ** (-s) - q)
        if np.less(diff, tol_rs):
            break
    np.save("./rational_sum_parameters/c_1s.npy", c_1s)
    np.save("./rational_sum_parameters/c_2s.npy", c_2s)
    np.save("./rational_sum_parameters/weights.npy", weights)
    np.save("./rational_sum_parameters/constant.npy", constant)
