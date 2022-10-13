import numpy as np
from scipy.stats import norm


def phi_small(x):
    return np.exp(-x*x/2)/np.sqrt(numpy.pi*2)

def bls_d_plus(s0, K, r, sigma, q, T):
    return (np.log(s0 * np.exp(-q * T) / (K * np.exp(-r * T))) + sigma * sigma * T / 2) / (sigma * np.sqrt(T))


def bls_d_minus(s0, K, r, sigma, q, T):
    return (np.log(s0 * np.exp(-q * T) / (K * np.exp(-r * T))) - sigma * sigma * T / 2) / (sigma * np.sqrt(T))


def generate_mc(s0, r, sigma, q, nsims, steps, T, seed=None):
    """
    Genera una matriz de montecarlo con los parametros especificados.
    Se puede pasar una lista de subyacentes que seran simulados en una tercera dimension, pero usando los mismos
    parametros en todos (se puede mejorar esta especificacion para que el subyacente i tome la volatilidad i, etc...)

    :param s0: Valor del subyacente en t0. Puede ser una lista de subyacentes
    :param r: Valor del tipo libre de riesgo
    :param sigma: Valor de la volatilidad
    :param q: Valor de la tasa continua de dividendos
    :param nsims: Numero de simulaciones
    :param steps: Numero de saltos intermedios
    :param T: Vencimiento en anos
    :param seed: Semilla para fijar el aleatorio (por defecto vacio)
    :return: Devuelve una matriz (nsims, steps+1) con las sendas simuladas
    """

    # Calculo del salto
    dt = T / steps
    if seed is not None:
        np.random.seed(seed)

    if not type(s0) is list:
        s0 = [s0]

    if len(s0) == 1:
        aleat = np.random.standard_normal((nsims, steps + 1))
    else:
        aleat = np.random.standard_normal((nsims, steps + 1, len(s0)))

    # Montamos la matriz de simulaciones
    msim = np.exp((r - q - sigma * sigma / 2) * dt + sigma * np.sqrt(dt) * aleat)

    if len(s0) == 1:
        msim[:, 0] = s0[0]
    else:
        for i in range(len(s0)):
            msim[:, 0, i] = s0[i]

    # Necesitamos hacer el producto acumulado (en coordenada 2)
    msim = np.cumprod(msim, 1)
    return msim

def get_implied_vol(sigma0, opt_value, s0, K, r, q, T, opt_type):
    f_error = lambda sigma, opt_value, s0, K, r, q, T, opt_type: abs(bls_option_price(s0, K, r, sigma, q, T, opt_type) - opt_value)
    res = fsolve(f_error, sigma0, args=(opt_value, s0, K, r, q, T, opt_type))
                                                                     
    return res[0]

def bls_option_price(s0, K, r, sigma, q, T, opt_type: str='call'):
    """
    Calcula el valor de una opcion plain vanilla mediante Black Scholes
    :param s0: Valor del subyacente en t0
    :param K: Valor del Strike
    :param r: Valor del tipo libre de riesgo
    :param sigma: Valor de la volatilidad
    :param q: Valor de la tasa continua de dividendos
    :param T: Vencimiento en anos
    :param opt_type: Tipo de opcion ('call' o 'put')
    :return: Devuelve el valor de la opcion
    """
    d_plus = bls_d_plus(s0, K, r, sigma, q, T)
    d_minus = bls_d_minus(s0, K, r, sigma, q, T)

    if opt_type.lower() == 'call' or opt_type.lower() == 'c':
        return s0 * np.exp(-q * T) * norm.cdf(d_plus) - K * norm.cdf(d_minus) * np.exp(-r * T)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d_minus) - s0 * np.exp(-q * T) * norm.cdf(-d_plus)


def bls_option_delta(s0, K, r, sigma, q, T, opt_type: str='call'):
    """
    Calcula la delta de una opcion plain vanilla mediante Black Scholes
    :param s0: Valor del subyacente en t0
    :param K: Valor del Strike
    :param r: Valor del tipo libre de riesgo
    :param sigma: Valor de la volatilidad
    :param q: Valor de la tasa continua de dividendos
    :param T: Vencimiento en anos
    :param opt_type: Tipo de opcion ('call' o 'put')
    :return: Devuelve el valor de la opcion
    """
    d_plus = bls_d_plus(s0, K, r, sigma, q, T)

    if opt_type.lower() == 'call' or opt_type.lower() == 'c':
        return np.exp(-q * T) * norm.cdf(d_plus)
    else:
        return np.exp(-q * T) * (norm.cdf(d_plus)-1)


def bls_option_gamma(s0, K, r, sigma, q, T, opt_type: str='call'):
    """
    Calcula la gamma de una opcion plain vanilla mediante Black Scholes
    :param s0: Valor del subyacente en t0
    :param K: Valor del Strike
    :param r: Valor del tipo libre de riesgo
    :param sigma: Valor de la volatilidad
    :param q: Valor de la tasa continua de dividendos
    :param T: Vencimiento en anos
    :param opt_type: Tipo de opcion ('call' o 'put')
    :return: Devuelve el valor de la opcion
    """
    d_plus = bls_d_plus(s0, K, r, sigma, q, T)

    return np.exp(-q * T) * phi_small(d_plus) / (s0 * sigma * np.sqrt(T))


def bls_option_vega(s0, K, r, sigma, q, T, opt_type: str='call'):
    """
    Calcula la vega de una opcion plain vanilla mediante Black Scholes
    :param s0: Valor del subyacente en t0
    :param K: Valor del Strike
    :param r: Valor del tipo libre de riesgo
    :param sigma: Valor de la volatilidad
    :param q: Valor de la tasa continua de dividendos
    :param T: Vencimiento en anos
    :param opt_type: Tipo de opcion ('call' o 'put')
    :return: Devuelve el valor de la opcion
    """
    d_plus = bls_d_plus(s0, K, r, sigma, q, T)

    # Igual en call y put
    return s0 * np.exp(-q * T) * phi_small(d_plus) * np.sqrt(T)


def bls_option_theta(s0, K, r, sigma, q, T, opt_type: str='call'):
    """
    Calcula la theta de una opcion plain vanilla mediante Black Scholes
    :param s0: Valor del subyacente en t0
    :param K: Valor del Strike
    :param r: Valor del tipo libre de riesgo
    :param sigma: Valor de la volatilidad
    :param q: Valor de la tasa continua de dividendos
    :param T: Vencimiento en anos
    :param opt_type: Tipo de opcion ('call' o 'put')
    :return: Devuelve el valor de la opcion
    """
    d_plus = bls_d_plus(s0, K, r, sigma, q, T)
    d_minus = bls_d_minus(s0, K, r, sigma, q, T)

    if opt_type.lower() == 'call' or opt_type.lower() == 'c':
        return -np.exp(-q * T) * s0 * phi_small(d_plus) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d_minus) + q * s0 * np.exp(-q * T) * norm.cdf(d_plus)
    else:
        return -np.exp(-q * T) * s0 * phi_small(d_plus) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d_minus) - q * s0 * np.exp(-q * T) * norm.cdf(-d_plus)



def bls_option_price_sim(s0, K, r, sigma, q, T, nsims, nsteps, opt_type='call'):
    """
    Calcula el valor de una opcion plain vanilla mediante simulacion
    :param s0: Valor del subyacente en t0
    :param K: Valor del Strike
    :param r: Valor del tipo libre de riesgo
    :param sigma: Valor de la volatilidad
    :param q: Valor de la tasa continua de dividendos
    :param T: Vencimiento en anos
    :param nsims: Numero de simulaciones
    :param nsteps: Numero de saltos intermedios
    :param opt_type: Tipo de opcion ('call' o 'put')
    :return: Devuelve el valor de la opcion
    """
    msims = generate_mc(s0, r, sigma, q, nsims, nsteps, T)
    if opt_type.lower() == 'call' or opt_type.lower() == 'c':
        return numpy.mean(np.maximum(msims[:, -1] - K, 0) * np.exp(-r * T))
    else:
        return numpy.mean(np.maximum(K - msims[:, -1], 0) * np.exp(-r * T))

