"""
A collection of laser-plasma interaction theoretical results.
The reference to the literature should be in the docstring if a function is calculating some quantities.
Different authors have their own definitions/normalizations convention; here we would stick to OSIRIS convention
unless mentioned otherwise.
"""
import plasma_dispersion as plds
import numpy as np
from functools import partial


def x2dens(dens_max, dens_min, x_max, x_min, x):
    """
    convert coordinate to density assuming linear density profile
    """
    return x * (dens_max - dens_min) / (x_max - x_min) + dens_min


def scale_length(dens_max, dens_min, x_max, x_min, x=None, ne=None):
    """
    return the scale length at x assuming linear density profile
    if ne is None, x is used to calculate the density
    default value of x is set to middle of the region
    returned value has the same unit as x_max and x_min
    """
    if ne is not None:
        return (x_max - x_min) * ne / (dens_max - dens_min)
    if x is None:
        return (x_max - x_min) * (dens_max + dens_min) / (2 * (dens_max - dens_min))
    return (dens_max - dens_min) / ((x_max - x_min) * (x * (dens_max - dens_min) / (x_max - x_min) + dens_min))


def vosc(intensity, wavelength=0.351):
    """
    convert intensity to electron oscillation velocity
    :param intensity: in 10^14 Watt/cm^2
    :param wavelength: wavelength of the laser in micron
    :return: oscillation velocity
    """
    return 0.008544 * np.sqrt(intensity) * wavelength


def phys_dens(ne, wavelength=0.351):
    """
    :param ne: normalized plasma density
    :param wavelength: wavelength of the laser in micron
    :return: electron number density in cm^{-3}
    """
    return 1.12e21 / (wavelength * wavelength) * ne


def phys_omega0(wavelength=0.351):
    """
    :param wavelength: wavelength of the laser in micron
    :return: angular frequency in rad/s
    """
    return 2 * np.pi * phys_freq0(wavelength)


def phys_freq0(wavelength=0.351):
    """
    :param wavelength: wavelength of the laser in micron
    :return: laser frequency in Hz
    """
    return 2.998E14 / wavelength


def linear_landaudamping(wp=0, kld=None, k=None, vth=None, ne=None):
    """
    calculate linear landau damping according to <Introduction to Plasma Physics, I.H.Hutchinson, chapter 5, (2001)
    http://silas.psfc.mit.edu/introplasma/> Eq. (5.232)
    :param wp: Langmuir frequency. users have to set either w or ne. if both are set then the value of w will be used.
    :param kld: value of k * Debye length
    :param k: wavenumber of plasma wave
    :param vth: thermal velocity
    :param ne: plasma density
    :return: the same unit as w
    """
    if not wp:
        wp = np.sqrt(ne)
    if not kld:
        kld = k * vth / wp
    return np.sqrt(np.pi / 8) * wp / kld**3 * np.exp(-(1 / (2 * kld * kld) + 1.5))


#
# ======================================== Stimulated Raman Scattering ================================================
#
def backward_srs_growthrate_h_ud(intensity, ne, vth=None, k=None, w=None, circular_laser=False):
    """
    Raman back scattering growth rate in homogeneous plasmas when damping is not important.
    See <Forslund, D. W., Kindel, J. M., & Lindman, E. L. (1975).
    Theory of stimulated scattering processes in laser-irradiated plasmas. Physics of Fluids, 18(8), 1002.>
    Eq. (32)
    :param intensity: in 10^14 Watt/cm^2
    :param ne: density normalized to critical density
    :param vth: thermal velocity
    :param k: wavenumber of the plasma daughter wave
    :param w: frequency of the plasma daughter wave
    :param circular_laser: if the laser is circular polarized (as in the original paper). Default is False
    :return: growth rate normalized to laser frequency
    """
    if w is None:
        w = np.real(plds.plasma_wave_w(ne, vth, partial(backward_srs_k_epw, ne=ne)))
    if k is None:
        k = backward_srs_k_epw(w, ne)
    if circular_laser:
        return __backward_srs_growthrate_h_ud_qutcr(k, intensity) * np.sqrt(2 * ne / (w * (1 - w)))
    return __backward_srs_growthrate_h_ud_qutcr(k, intensity) * np.sqrt(ne / (w * (1 - w)))


def __backward_srs_growthrate_h_ud_qutcr(k, intensity, wavelength=0.351):
    return k * vosc(intensity, wavelength=wavelength) / 4


def backward_srs_beta_forslung(intensity, ne, vth):
    """
    A parameter quantifying whether absolute SRS is suppressed by damping
    <Forslund, D. W., Kindel, J. M., & Lindman, E. L. (1975).
    Theory of stimulated scattering processes in laser-irradiated plasmas. Physics of Fluids, 18(8), 1002.>
    Eq. (43)
    :param intensity: in 10^14 Watt/cm^2
    :param ne: density normalized to critical density
    :param vth: thermal velocity
    """
    w = None
    vg = plds.plasma_wave_vg(vth, partial(backward_srs_k_epw, ne=ne), w=w, ne=ne)
    wr = np.real(w)
    vm, kp = plds.light_wave_vg(backward_srs_k_epw(wr, ne), wr), backward_srs_k_epw(wr, ne)
    return np.abs(np.imag(w)) / backward_srs_growthrate_h_ud(intensity, ne, k=kp, w=wr) * np.sqrt(vm / vg)


def backward_srs_k_epw(w, ne):
    """
    solve plasma/light wave dispersions and matching conditions in backward SRS. k:=k(w)
    :param w: plasma wave frequency
    :param ne: plasma density
    :return: wavenumber of plasma wave
    """
    return np.sqrt(1 - ne) + np.sqrt((1 - w) * (1 - w) - ne)


def forward_srs_k_epw(w, ne):
    """
    solve plasma/light wave dispersions and matching conditions in forward SRS. k:=k(w)
    :param w: plasma wave frequency
    :param ne: plasma density
    :return: wavenumber of plasma wave
    """
    return np.sqrt(1 - ne) - np.sqrt((1 - w) * (1 - w) - ne)


def backward_srs_convective_gain(intensity, ne, ln, wavelength=0.351, nu1=0, nu2=0, author='Albright',
                                 gamma0=0, kappa_prime=None, v1=None, v2=None):
    """
    calculate linear convective gain factor for backward SRS.
    depending on the author parameter, see
    <Albright, B. J., Yin, L., & Afeyan, B. (2014). Control of stimulated Raman scattering in the
    strongly nonlinear and kinetic regime using spike trains of uneven duration and delay.
    Physical Review Letters, 113(4), 45002>
    <Rosenbluth, M. N. (1972). Parametric Instabilities in Inhomogeneous Media. Phys. Rev. Lett., 29(9), 565.>
    <Williams, E. A. (1991). Convective growth of parametrically unstable modes in inhomogeneous media.
    Physics of Fluids B, 3(6), 1504–1506.>
    <Afeyan, B., & Hüller, S. (2013). Optimal Control of Laser-Plasma Instabilities Using Spike Trains of Uneven
    Duration and Delay: STUD Pulses. EPJ Web of Conferences, 59, 5009>
    :param intensity: in 10^14 Watt/cm^2
    :param ne: plasma density normalized to critical density
    :param ln: plasma density scale length in micron
    :param nu1: damping frequency of the light wave, normalized to laser frequency. Default is 0.
    :param nu2: damping frequency of the plasma wave, normalized to laser frequency. Default is 0.
    :param wavelength: wavelength of the laser in micron
    :param author: select which formulation/convention
    :param gamma0: growth rate/coupling strength in homogeneous plasmas
    :param kappa_prime: mismatch in wavenumber
    :param v1: group velocity of the light daughter wave
    :param v2: group velocity of the plasma daughter wave
    :return:
    """
    def factor(_gamma0=0, _kappa_prime=None, _v1=None, _v2=None):
        return _gamma0 * _gamma0 / np.abs(_kappa_prime * _v1 * _v2)

    if author.lower() == 'albright':
        gamma0 = 0.0043 * wavelength * np.sqrt(intensity)
        gn = np.sqrt(1 - 2 * np.sqrt(ne)) / (1 / np.sqrt(ne) - 1)
        return 8 * np.pi * np.pi * gamma0 * gamma0 * ln / wavelength / gn * (1 - nu1 * nu2 / gamma0)

    elif author.lower() == 'Rosenbluth':
        return 2 * np.pi * factor(gamma0, kappa_prime, v1, v2)

    elif author.lower() == 'willian':
        g = nu1 * nu2 / gamma0
        return 2 * factor(gamma0, kappa_prime, v1, v2) * (np.arccos(g) - g * np.sqrt(1 - g * g)) if g < 1 else 0

    elif author.lower() == 'afeyan':
        g = nu1 * nu2 / gamma0
        return 2 * np.pi * factor(gamma0, kappa_prime, v1, v2) * (1 - g) if g < 1 else 0


def _wavenumber_mismatch_backward_srs(ln, ne, vth, k0=None, k=None, w=None):
    """
    calculate wavenumber mismatch as in <Liu, C. S., Rosenbluth, M. N., & White, R. B. (1974).
    Raman and Brillouin scattering of electromagnetic waves in inhomogeneous plasmas.
    The Physics of Fluids, 17(6), 1211–1219.> Eq. (28)
    :param ln: plasma density scale length in micron
    :param ne: plasma density normalized to critical density
    :param vth: thermal velocity
    :param k0: wavenumber of the laser pump wave. If k0=None then sqrt(1-ne) will be used.
    :param k: wavenumber of the plasma daughter wave. If not set then k will be obtained by solving dispersion relation
    :param w: angular frequency of the plasma wave. If not set then w will be obtained by solving dispersion relation
    :return: unit is OSIRIS unit * micron^{-1}
    """
    if k is None:
        k = backward_srs_k_epw(w, ne)
    if k0 is None:
        k0 = np.sqrt(1 - ne)
    return ne / (2 * ln) * (1 / (k - k0) + 1 / (3 * k * vth * vth) - 1 / k0)


def backward_srs_lint_wdl(intensity, ln, ne, w=0, v1=0, v2=0, vth=None, wavelength=0.351, approx=True,
                          nu1=0, nu2=0, kappa_prime_approx=True, gamma0_approx=True):
    """
    Return the interaction length in the weak damping limit.
    See <Afeyan, B., & Hüller, S. (2013). Optimal Control of Laser-Plasma Instabilities Using Spike Trains of Uneven
    Duration and Delay: STUD Pulses. EPJ Web of Conferences, 59, 5009.>
    See also <Williams, E. A. (1991). Convective growth of parametrically unstable modes in inhomogeneous media.
    Physics of Fluids B, 3(6), 1504–1506.>

    if approx is true then the formula from <Albright, B. J., Yin, L., & Afeyan, B. (2014). Control of stimulated
    Raman scattering in the strongly nonlinear and kinetic regime using spike trains of uneven duration and delay.
    Physical Review Letters, 113(4), 45002> is used

    if kappa_prime_approx is false then we calculate the wavenumber mismatch using definition. See for detail:
    <Liu, C. S., Rosenbluth, M. N., & White, R. B. (1974). Raman and Brillouin scattering of electromagnetic
    waves in inhomogeneous plasmas. The Physics of Fluids, 17(6), 1211–1219.> Eq. (28)
    :param intensity: in 10^14 Watt/cm^2
    :param ln: plasma density scale length in micron
    :param ne: plasam density normalized to critical density
    :param w: plasma wave frequency
    :param v1: group velocity of the light daughter wave
    :param v2: group velocity of the plasma daughter wave
    :param vth: thermal velocity. user must supply this parameter if v2=None.
    :param wavelength: wavelength of the laser in micron
    :param approx: use formula from [Albright et al (2014)]
    :param nu1: damping frequency of the light wave, normalized to laser frequency. Default is 0.
    :param nu2: damping frequency of the plasma wave, normalized to laser frequency. Default is 0.
    :param kappa_prime_approx: assume very low density plasma so k_mismatch = w_epw / (2 * v2 * ln). Default is true.
    :param gamma0_approx: assume scattered light wave is near its turning point. Default is true.
    :return: The unit is micron
    """
    if not v2 or not v1:
        v2 = plds.plasma_wave_vg(vth, k=partial(backward_srs_k_epw, ne=ne), w=w, ne=ne)
        v1 = plds.light_wave_vg(backward_srs_k_epw(np.real(w), ne=ne) - np.sqrt(1 - ne), ne=ne)

    if approx:
        return 4 * ln * wavelength * np.sqrt(1.14E-4 * np.abs(v2 / v1) / ne * intensity)
    else:
        if w == 0:
            w = np.real(plds.plasma_wave_w(ne, vth, k=partial(backward_srs_k_epw, ne=ne)))
        wr = np.real(w)
        gamma0 = __backward_srs_growthrate_h_ud_qutcr(backward_srs_k_epw(wr, ne), intensity, wavelength=wavelength)
        if not gamma0_approx:
            gamma0 *= np.sqrt(ne / (wr * (1 - wr)))
        if kappa_prime_approx:
            kappa_prime = wr / (2 * v2 * ln)
        else:
            kappa_prime = _wavenumber_mismatch_backward_srs(ln, ne, vth, w=wr)
        return np.sqrt(backward_srs_convective_gain(intensity, ne, ln, nu1, nu2, author='afeyan', gamma0=gamma0,
                                                    kappa_prime=kappa_prime, v1=v1, v2=v2) /
                       (kappa_prime * np.sqrt(np.pi))) * 2


def backward_srs_lint_sdl(ln, ne, w=0., nu1=0., nu2=0., v1=None, v2=None, vth=None, kappa_prime_approx=True):
    """
    Return the interaction length in the strong damping limit.
    See <Afeyan, B., & Hüller, S. (2013). Optimal Control of Laser-Plasma Instabilities Using Spike Trains of Uneven
    Duration and Delay: STUD Pulses. EPJ Web of Conferences, 59, 5009>

    if kappa_prime_approx is true then the formula from <Albright, B. J., Yin, L., & Afeyan, B. (2014).
    Control of stimulated Raman scattering in the strongly nonlinear and kinetic regime using spike trains
    of uneven duration and delay. Physical Review Letters, 113(4), 45002> is used
    :param ln: plasma density scale length in micron
    :param ne: plasam density normalized to critical density
    :param w: plasma wave frequency
    :param nu1: damping frequency of the light wave, normalized to laser frequency. Default is 0.
    :param nu2: damping frequency of the plasma wave, normalized to laser frequency. Default is 0.
    :param v1: group velocity of the light daughter wave
    :param v2: group velocity of the plasma daughter wave
    :param vth: thermal velocity. user must supply this parameter if v2=None.
    :param kappa_prime_approx:  assume very low density plasma so k_mismatch = w_epw / (2 * v2 * ln). Default is true.
    :return: The unit is micron
    """
    if not w or not nu2:
        w_tmp = plds.plasma_wave_w(ne, vth, k=partial(backward_srs_k_epw, ne=ne))
        if not nu2:
            nu2 = np.abs(np.imag(w_tmp))
        if not w:
            w = np.real(w_tmp)

    if kappa_prime_approx:
        return 4 * ln * nu2 / w
    else:
        if not v2 or not v1:
            v2 = plds.plasma_wave_vg(vth, k=backward_srs_k_epw(w, ne), w=w, ne=ne)
            v1 = plds.light_wave_vg(backward_srs_k_epw(np.real(w), ne=ne) - np.sqrt(1 - ne), ne=ne)
        kappa_prime = _wavenumber_mismatch_backward_srs(ln, ne, vth, w=np.real(w))
        return 2 * (nu1 / np.abs(v1) + nu2 / np.abs(v2)) / np.abs(kappa_prime)


def backward_srs_lint(ln, intensity, ne, vth, v1=0, v2=0, wavelength=0.351, accuracy_level=0):
    if accuracy_level <= 0:
        w = plds.plasma_wave_w(ne, vth, partial(backward_srs_k_epw, ne=ne))
        k = backward_srs_k_epw(np.real(w), ne=ne)
        nu2 = linear_landaudamping(k=k, vth=vth, ne=ne)
        return np.sqrt(backward_srs_lint_wdl(intensity, ln, ne, 0, v1, v2, vth, wavelength)**2 +
                       backward_srs_lint_sdl(ln, ne, w=np.real(w), nu2=nu2, v1=v1, v2=v2)**2)

    elif accuracy_level == 1:
        w = plds.plasma_wave_w(ne, vth, partial(backward_srs_k_epw, ne=ne))
        k = backward_srs_k_epw(np.real(w), ne=ne)
        nu2 = linear_landaudamping(k=k, vth=vth, ne=ne)
        return np.sqrt(backward_srs_lint_wdl(intensity, ln, ne, 0, v1, v2, vth, wavelength, approx=False)**2 +
                       backward_srs_lint_sdl(ln, ne, w=np.real(w), nu2=nu2, v1=v1, v2=v2)**2)

    elif accuracy_level == 2:
        w = plds.plasma_wave_w(ne, vth, partial(backward_srs_k_epw, ne=ne))
        k = backward_srs_k_epw(np.real(w), ne=ne)
        nu2 = linear_landaudamping(k=k, vth=vth, ne=ne)
        return np.sqrt(backward_srs_lint_wdl(intensity, ln, ne, 0, v1, v2, vth, wavelength, approx=False,
                                             kappa_prime_approx=False)**2 +
                       backward_srs_lint_sdl(ln, ne, w=np.real(w), nu2=nu2, v1=v1, v2=v2)**2)

    elif accuracy_level == 3:
        w = plds.plasma_wave_w(ne, vth, partial(backward_srs_k_epw, ne=ne))
        k = backward_srs_k_epw(np.real(w), ne=ne)
        nu2 = linear_landaudamping(k=k, vth=vth, ne=ne)
        return np.sqrt(backward_srs_lint_wdl(intensity, ln, ne, 0, v1, v2, vth, wavelength, approx=False,
                                             kappa_prime_approx=False, gamma0_approx=False)**2 +
                       backward_srs_lint_sdl(ln, ne, w=np.real(w), nu2=nu2, v1=v1, v2=v2)**2)
    
    elif accuracy_level >= 4:
        w = plds.plasma_wave_w(ne, vth, partial(backward_srs_k_epw, ne=ne))
        k = backward_srs_k_epw(np.real(w), ne=ne)
        nu2 = linear_landaudamping(k=k, vth=vth, ne=ne)
        return np.sqrt(backward_srs_lint_wdl(intensity, ln, ne, 0, v1, v2, vth, wavelength, approx=False,
                                             kappa_prime_approx=False, gamma0_approx=False)**2 +
                       backward_srs_lint_sdl(ln, ne, w=np.real(w), nu2=nu2, v1=v1, v2=v2, vth=vth,
                                             kappa_prime_approx=False)**2)


def srs_side_scattering_growthrate(ln, intensity, ne, vth, k0=None, wavelength=0.351):
    """
    calculate the growth rate of Raman side scattering (absolutely unstable) modes for a given density.
    see <Liu, C. S., Rosenbluth, M. N., & White, R. B. (1974). Raman and Brillouin scattering of electromagnetic
    waves in inhomogeneous plasmas. The Physics of Fluids, 17(6), 1211–1219.> Eq. (57)

    Here we assume the scattered light waves propagate exactly 90-degree to the incident laser
    :param ln: plasma density scale length in micron
    :param intensity: in 10^14 Watt/cm^2
    :param ne: plasma density normalized to critical density
    :param vth: thermal velocity
    :param k0: wavenumber of the laser pump wave. If k0=None then sqrt(1-ne) will be used
    :param wavelength: wavelength of the laser in micron
    :return: normalized to \omega_0
    """
    # ve2 = TkeV / 511.0
    # den = yaxis
    ve2 = vth ** 2
    kp = np.sqrt((1 - 3 * (ne - 2) * ve2 + 9 * (ne - 1) * ve2**2 -
                  2 * np.sqrt(ne + 6 * ve2 - 9 * ne * ve2 - 9 * ve2**2 + 18 * ne * ve2**2)) / (1 - 3 * ve2)**2)
    wpe = np.sqrt(ne)
    if k0 is None:
        k0 = np.sqrt(1 - ne)
    # here we assume it is exact 90-degree sidescattering
    nkc = np.sqrt(kp**2 + k0**2)
    kL = nkc * ln / wavelength
    v0 = 4.27e-3 * np.sqrt(intensity) * wavelength
    drkN = 6.283 * ln / wavelength * v0**1.5 * np.sqrt(0.75)
    gm = np.imag(complex(0.0, 1.0) * v0 * nkc * wpe * (1 - 0.5 * complex(np.sqrt(2) / 2.0, np.sqrt(2.) / 2.0) *
                                                       np.sqrt(wpe / nkc) / drkN))
    return gm


#
# ============================================= Two Plasmon Decay =====================================================
#
def tpd_small_k(w, ne):
    pass


def tpd_large_k(w, ne):
    pass


# print(backward_srs_lint(scale_length(0.15, 0.09, 500, 0), 5, ne=0.12, vth=np.sqrt(2.6/511), accuracy_level=0))

# print(backward_srs_lint_wdl(1000, 5, vth=np.sqrt(2.6/511), ne=0.12))
