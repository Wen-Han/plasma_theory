import numpy as np
import scipy
from scipy import optimize


def zfunction(z):
    return scipy.special.wofz(z) * np.sqrt(np.pi) * 1j


def zfunction_prime(z):
    return -2.0 * (1.0 + z * zfunction(z))


def zfunction_prime2(z):
    return -2.0 * (zfunction(z) + z * zfunction_prime(z))


def plasma_wave_w(ne, vth, k, inital_root_guess=None):
    """
    calculate plasma wave frequency for given density and wavenumber
    :param ne: normalized plasma density
    :param vth: thermal velocity
    :param k: wavenumber of the plasma wave, can be a number or a function of plasma wave frequency (i.e. k:=k(w))
    :param inital_root_guess: initial value for the root finder
    :return: plasma wave frequency (usually a complex number)
    """
    wp = np.sqrt(ne)
    if callable(k):
        def kvth(x):
            return vth * k(x)
    else:
        def kvth(_):
            return vth * k

    def plasma_epsilon(x):
        # chi = chi_e(x)
        # print(chi)
        kv = kvth(x)
        val = 1.0 - zfunction_prime(x / (kv * np.sqrt(2.))) * ne / (2 * kv * kv)
        return val

    if inital_root_guess is None:
        #    # use the Bohm-Gross dispersion formulas to get an initial guess for w
        if callable(k):
            inital_root_guess = np.sqrt(ne + 3 * (k(wp) * vth)**2)
        else:
            inital_root_guess = np.sqrt(ne + 3 * (k * vth)**2)
    epsilon_root = scipy.optimize.newton(plasma_epsilon, inital_root_guess)
    return epsilon_root


def plasma_wave_vg(vth, k, w=None, ne=None):
    """
    calculate the group velocity of the plasma wave
    :param vth: thermal velocity
    :param k: wavenumber of the plasma wave, can be a number or a function of plasma wave frequency (i.e. k:=k(w))
    :param w: angular frequency of the plasma wave, if w=w0=0 then w0 will hold the value of omega satisfying dispersion
    :param ne: plasma density
    :return:
    """
    if not w:
        w = plasma_wave_w(ne, vth, k)
    kk = k(np.real(w)) if callable(k) else k
    z = np.real(w) / (kk * vth * np.sqrt(2))
    return np.real(w / kk + 2 * np.sqrt(2) * vth * zfunction_prime(z) / zfunction_prime2(z))


def light_wave_vg(k=None, w=None, ne=None):
    if w is None:
        return 1. / np.sqrt(ne / (k * k) + 1.)
    else:
        if k is None:
            return np.sqrt(w * w - ne) / w
        return k / w

