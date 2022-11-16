import baryrat as br
import numpy as np

def BP_rational_approximation(kappa, s):
    """
    Generates the parameters for the rational sum according to exponentially
    convergent scheme in Bonito and Pasciak 2013.

    Parameters:
        kappa: fineness parameter
        s: fractional power

    Returns a dict containing:
        rational_parameters: a dict containing:
            c_1s: diffusion coefficients of the rational sum
            c_2s: reaction coefficients of the rational sum
            weights: multiplicative coefficients of the rational sum
            constant: multiplicative constant in front of the sum
        err: the rational approximation error estimation
    """

    M = np.ceil((np.pi**2) / (4. * s * kappa**2))
    N = np.ceil((np.pi**2) / (4. * (1. - s) * kappa**2))

    ls = np.arange(-M, N + 1, 1, dtype=np.float64)
    c_1s = np.exp(2. * kappa * ls)
    c_2s = np.ones_like(c_1s)
    weights = np.exp(2. * s * kappa * ls)
    constant = (2. * np.sin(np.pi * s) * kappa) / np.pi

    rational_parameters = {"c_1s": c_1s,
                           "c_2s": c_2s,
                           "weights": weights,
                           "constant": constant,
                           "initial constant": 0.}   # There is no initial term in this method so initial_constant must be 0.
    
    # Rational error estimation
    xs = np.linspace(1., 1e8, 10000)

    ys = []
    for x in xs:
        bp_terms = np.multiply(weights, np.reciprocal(c_2s + c_1s * x))
        bp_sum = np.sum(bp_terms)
        bp_sum *= constant
        ys.append(bp_sum)
    
    err = np.max(np.abs(np.power(xs, -s) - ys))

    return rational_parameters, err


def BURA_rational_approximation(degree, s):
    """
    Generates the parameters for the BURA using the BRASIL method from Hofreither 2020.

    Parameters:
        degree: degree of the rational approximation (= number of parametric solves - 1)
        s: fractional power
    Returns a dict containing:
        rational_parameters: dict containing
            c_1s: diffusion coefficients of the rational sum
            c_2s: reaction coefficients of the rational sum
            weights: multiplicative coefficients of the rational sum
            constant: multiplicative constant in front of the sum
            initial_constant: once the parametric solutions are added initial_constant * f must be added to this sum to obtain the fractional approximation
        err: the rational approximation error estimation
    """

    def r(x):           # BURA method approximate x^s instead of x^{-s}
        return x**s

    domain = [1e-8, 1.] # The upper bound is lambda_1^{-1} where lambda_1 is the lowest eigenvalue, in this case lambda_1 = 1
    xs = np.linspace(domain[0], 1., 10000)

    r_brasil = br.brasil(r, domain, degree)
    pol_brasil, res_brasil = r_brasil.polres()

    c_1s = -pol_brasil
    c_2s = np.ones_like(c_1s)
    weights = res_brasil/pol_brasil
    constant = 1.

    rational_parameters = {"c_1s": c_1s,
                           "c_2s": c_2s,
                           "weights": weights,
                           "constant": constant,
                           "initial constant": r_brasil(0.)}

    # Rational error estimation
    def rational_sum(xs):
        ys = []
        for x in xs:
            ys.append(r_brasil(0.) + constant * sum([weight * 1./(c_2 + x * c_1) for weight, c_1, c_2 in zip(weights, c_1s, c_2s)]))
        return ys
    xs = np.linspace(1., 1.e2, 10000)
    err = np.max(np.abs(1./r(xs) - rational_sum(xs)))

    return rational_parameters, err

if __name__=="__main__":
    BP_parameters, BP_err = BP_rational_approximation(0.3, 0.5)
    BURA_parameters, BURA_err = BURA_rational_approximation(6, 0.5)