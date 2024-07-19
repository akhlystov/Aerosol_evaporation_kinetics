# Andrey Khlystov 2024

import numpy as np
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import quad
# import matplotlib.pyplot as plt

####### "Classical" approach #######


def coll_int(temper, eps):
    """
    Collision integral for diffusion coefficient calculations
    :param temper: Temperature, K
    :param eps: eps/kb
    :return: the collision integral
    """
    # data from Bird p746
    x = np.arange(0.3, 2., 0.05)
    y = np.array([2.662, 2.476, 2.318, 2.184, 2.066,
                  1.966, 1.877, 1.798, 1.729, 1.667,
                  1.612, 1.562, 1.517, 1.476, 1.439,
                  1.406, 1.375, 1.346, 1.320, 1.296,
                  1.273, 1.253, 1.233, 1.215, 1.198,
                  1.182, 1.167, 1.153, 1.140, 1.128,
                  1.116, 1.105, 1.094, 1.084])
    f_interp = sp.interpolate.interp1d(x, y)  # interpolation function
    return float(f_interp(temper/eps))


def crit_vol(gr):
    """
    Critical volume, Lydersen method
    :param gr: array of the number of the corresponding groups:
    Group contributions:
    0:-CH3,-CH2-            1:-C<           2:=c<,=c=
    3:-CH2-(Ring)           4:>C<(Ring)     5:-F
    6:-Br                   7:-OH           8:-O-
    9:>C=O                  10:HC=O-        11:-COO-
    12:>NH                  13:>N           14:-CN
    15:-SH,-S-              16:=s           17:>CH
    18:=CH2,#ch             19:=c-h,#c-     20:>CH-(Ring)
    21:=ch-,=c<,=c=(ring)   22:-Cl          23:-I
    24:-OH(Aromat)          25:-O-(Ring)    26:>C=O(Ring)
    27:-COOH                28:-NH2         29:>NH(Ring)
    30:>N-(Ring)            31:-NO2         32:-S-(Ring)

    :return:  critical volume of the molecule
    """
    g = np.array([55, 41, 36, 44.5, 31, 18, 70, 18, 20, 60, 73,
                  80, 37, 42, 80,   55, 47, 51, 45, 36, 46, 37,
                  49, 95, 3,  8,    50, 80, 28, 27, 32, 78, 45])  # length 33
    return 40. + np.dot(gr, g)


def diff_coeff(temper, mw, temp_melt, vc, p=1):
    """
    Diffusion coefficient of a substance in air in cm2/s (Bird, p511)

    :param temper: temperature, K
    :param mw: molecular weight of the species, g/mol
    :param temp_melt: melting point temperature of the species, K
    :param vc: critical volume for the species, cm3/mol
    :param p: pressure, atm
    :return:
    """
    mw_air = 28.97  # mol weight of air, g/mol
    eps_air = 97  # K, eps/Kb of air
    eps_i = 1.92 * temp_melt  # Bird, p22
    eps_i_air = np.sqrt(eps_i * eps_air)
    sigma_air = 3.617  # Angstrom
    sigma_i = 0.841 * vc**(1/3)  # Bird, p22
    sigma_i_air = (sigma_i + sigma_air) / 2
    return 0.0018583 * np.sqrt(temper**3 * (1/mw_air + 1/mw))/(p * sigma_i_air**2 * coll_int(temper, eps_i_air))


def mean_free_path(diffcoef, mw, tempr):
    """
    The mean free path of a gas in another according to Fuchs-Sutugin in nm

    :param diffcoef: diff coeff in cm2/s
    :param mw: molecular weight in g/mol
    :param tempr: temperature in K
    :return: the mean free path
    """
    return 3e5 * diffcoef / np.sqrt(8 * 8.31 * tempr / (np.pi * mw * 1e-3))


# adipic acid params
def adipic_params(tempr, press):
    """
    Calculates diffusion coefficient and mfp for adipic acid

    :param tempr: temperature in K
    :param press: pressure in atm
    :return: the diffusion coefficient and the mfp for adipic acid
    """
    gr = np.zeros(33)
    gr[0] = 4  # number of -CH2- and -CH3 groups
    gr[27] = 2  # number of -COOH groups
    temp_melt = 273.15+152.1  # melting point
    mw = 146.14  # g/mol
    diffcoeff = diff_coeff(tempr, mw, temp_melt, crit_vol(gr), press)
    mfp = mean_free_path(diffcoeff, mw, tempr)  # mean free path, should be ~ 82.41
    return diffcoeff, mfp


def char_time(c_sat, d0=100, diff_coef=0.0565, rho=1.3):
    """
    Calculates characteristic time in seconds.

    :param c_sat: Saturation concentration, ug/m3
    :param d0: Initial particle diameter, nm.
    :param diff_coef: Diffusion coefficient, cm2/s
    :param rho: Particle density, g/cm3
    :return: Characteristic time, s
    """
    # g/cm3 1e-14cm2 / (cm2 /s 1e-6 g / (1e6 cm3)) =  1e-14 1e12 s = 1e-2 s
    return 1e-2 * rho * d0**2 / (12 * diff_coef * c_sat)


def kelvin(dp, sigma=0.17, temper=298.15, mw=146.1, rho=1.3):
    """
    The Kelvin effect

    :param dp: particle diameter, nm
    :param sigma: surface tension, J/m2. The default is 0.17.
    :param temper: temperature, K. The default is 298.15.
    :param mw: molecular weight, g/mol. The default is 146.1.
    :param rho: the density, g/cm3. The default is 1.3
    :return: the Kelvin effect
    """
    return np.exp(1e3 * 4 * sigma * mw / (8.31 * rho * temper * dp))


def fuchs(dp, mfp=81.6, alpha=1):
    """
    Fuchs-Sutugin correction factor
    :param dp: particle diameter, nm.
    :param mfp: the mean free path, nm.
    :param alpha: the evaporation coefficient.
    :return: the Fuchs correction
    """
    knudsen = mfp * 2 / dp
    return 1 / (1.333 * knudsen / alpha + (0.3773 * knudsen + 1) / (knudsen + 1))


def diam_from_xi(xi, vbs, d0):
    """
    Diameter of at any given time, provided the original vbs and initial diameter.

    :param xi: the current values of individual mfrs
    :param vbs: the original VBS, i.e., the mass fractions of individual compounds in the original mixture
    :param d0: the initial diameter, nm.
    :return: the diameter for a given xi, nm.
    """
    return d0 * (np.sum(vbs * xi))**(1/3)


def deriv_logxi_trans(logxi, t, vbs, tau, d_0, mfp=81.6, alpha=1, sigma=0.17, temper=298.15, mw=146.1, rho=1.3,
                      d_min=5):
    """
    Change rate of log MFRs of individual compounds in the mixture in the transition regime. This is for now for a
    mixture with the same Mw.

    :param logxi: log of the current values of individual mfrs
    :param t: time, s. Is not used in calculations, but is needed for integration of this function.
    :param vbs: the original VBS, i.e., the mass fractions of individual compounds in the original mixture
    :param tau: characteristic times of individual compounds
    :param d_0: the initial diameter, nm.
    :param mfp: the mean free path of the evaporating material.
    :param alpha: the evaporation coefficient.
    :param sigma: the surface tension, J/m2.
    :param temper: temperature, K.
    :param mw: molecular weight, g/mol.
    :param rho: density, g/cm3
    :param d_min: diameter at which the evaporation will stop, nm. Used to avoid problems with Kelvin at small d.
    :return: the rate of individual mfr change, 1/s.
    """
    mfr_cuberoot = (np.sum(vbs * np.exp(logxi))) ** (1/3)
    dp = d_0 * mfr_cuberoot
    if dp <= d_min:
        return np.zeros_like(logxi)
    else:
        return - fuchs(dp, mfp, alpha) * kelvin(dp, sigma, temper, mw, rho) / (tau * mfr_cuberoot ** 2)


def deriv_logxi_cont(logxi, t, vbs, tau):
    """
    Change rate of MFRs of individual compounds in the mixture in the continuum regime. This is for now for a
    mixture with the same Mw.

    :param logxi: log of the current values of individual mfrs
    :param t: time, s. Is not used in calculations, but is needed for integration of this function.
    :param vbs: the original VBS, i.e., the mass fractions of individual compounds in the original mixture
    :param tau: characteristic times of individual compounds
    :return: the rate of individual mfr change, 1/s.
    """
    return - 1 / (tau * (np.sum(vbs * np.exp(logxi))) ** (2/3))


def calc_xi_trans(vbs, taus, t, d_0, mfp=81.6, alpha=1, sigma=0.17, temper=298.15, mw=146.1, rho=1.3, d_min=5,
                  atol=1e-14):
    """
    Calculates MFR of individual components in a mixture after evaporation for time t in the transition regime.
    Calculations assume that all compounds have the same properties except saturation vapor concentration, the mixture
    is ideal, and that the particle is well-mixed. This function uses log-xi method. This is for now for a
    mixture with the same Mw.

    :param vbs: the original VBS, i.e., the mass fractions of individual compounds in the original mixture
    :param taus: characteristic times of individual compounds
    :param t: time, s.
    :param d_0: the initial diameter, nm.
    :param mfp: the mean free path of the evaporating material.
    :param alpha: the evaporation coefficient.
    :param sigma: the surface tension, J/m2.
    :param temper: temperature, K.
    :param mw: molecular weight, g/mol.
    :param rho: density, g/cm3
    :param d_min: diameter at which the evaporation will stop, nm. Used to avoid problems with Kelvin at small d.
    :param atol: absolute tolerance for the integration.
    :return: the MFR of individual compounds.
    """
    logxi = odeint(deriv_logxi_trans, np.zeros_like(vbs), t,
                   args=(vbs, taus, d_0, mfp, alpha, sigma, temper, mw, rho, d_min),
                   atol=atol)
    return np.exp(logxi)


def calc_xi_cont(vbs, taus, t, atol=1e-14, rtol=1e-12):
    """
    Calculates MFR of individual components in a mixture after evaporation for time t in the continuum regime.
    Calculations assume that all compounds have the same properties except saturation vapor concentration, the mixture
    is ideal, and that the particle is well-mixed. This function uses log-xi method. This is for now for a
    mixture with the same Mw.

    :param vbs: the original VBS, i.e., the mass fractions of individual compounds in the original mixture
    :param taus: characteristic times of individual compounds
    :param t: time, s.
    :param atol: absolute tolerance for the integration.
    :param rtol: relative tolerance for the integration
    :return: the MFR of individual compounds.
    """
    logxi = odeint(deriv_logxi_cont, np.zeros_like(vbs), t, args=(vbs, taus), atol=atol, rtol=rtol)
    return np.exp(logxi)


####### The new approach starts here #######

def f_trans(xi, xi_ind, cs, vbs, mw, d0=300, mfp=81.6, alpha=1, sigma=0.17, temper=298.15, rho=1.3):
    """
    Function for calculating the effective time for the transition regime
    :param xi: the MFR of the bin in question
    :param xi_ind: the index of the bin
    :param cs: the saturation concentrations of the compounds in the mixture (array); does not need to be log spaced
    :param vbs: the mass fraction of compounds in the mixture (array length of cs)
    :param mw: the molecular weight of compounds in the mixture (array length of cs)
    :param d0: the initial diameter, nm.
    :param mfp: the mean free path, nm.
    :param alpha: the evaporation coefficient.
    :param sigma: surface tension, J/m2. The default is 0.17.
    :param temper: temperature, K. The default is 298.15.
    :param rho: the density, g/cm3. The default is 1.3

    :return: the function needed to calculate the effective time.
    """
    m_star = mw/mw[xi_ind]
    a_ji = cs/cs[xi_ind]
    dp = diam_from_xi(xi, vbs, d0)

    return (np.sum(vbs * xi**a_ji))**(5/3) / (fuchs(dp, mfp, alpha) * kelvin(dp, sigma, temper, mw[xi_ind], rho) *
                                              xi * np.sum(m_star * vbs * xi**a_ji))


def calc_t_eff_trans(x_final, xi_ind, cs, vbs, mw, d0=300, mfp=81.6, alpha=1, sigma=0.17, temper=298.15, rho=1.3):
    """
    Function to calculate the effective time needed for a compound to reach a certain MFR in the transition regime using
    numerical integration.
    :param x_final: the MFR to reach
    :param xi_ind: the index of the bin for which the effective time is to be calculated
    :param cs: the saturation concentrations of the compounds in the mixture (array); does not need to be log spaced
    :param vbs: the mass fraction of compounds in the mixture (array length of cs)
    :param mw: the molecular weight of compounds in the mixture (array length of cs)
    :param d0: the initial diameter, nm.
    :param mfp: the mean free path, nm.
    :param alpha: the evaporation coefficient.
    :param sigma: surface tension, J/m2. The default is 0.17.
    :param temper: temperature, K. The default is 298.15.
    :param rho: the density, g/cm3. The default is 1.3
    :return: effective time to reach MFR = x_final
    """
    res = quad(f_trans, x_final, 1, args=(xi_ind, cs, vbs, mw, d0, mfp, alpha, sigma, temper, rho))
    return res[0]


def f_cont(xi, xi_ind, cs, vbs, mw):
    """
    Function under the integral of Eq.11 (the exact solution for t^*_i in the continuum regime)
    :param xi: the MFR of the bin in question
    :param xi_ind: the index of the bin
    :param cs: the saturation concentrations of the compounds in the mixture (array); does not need to be log spaced
    :param vbs: the mass fraction of compounds in the mixture (array length of cs)
    :param mw: the molecular weight of compounds in the mixture (array length of cs)
    :return: the value of the function under the integral of Eq.
    """
    m_star = mw/mw[xi_ind]
    a_ji = cs/cs[xi_ind]

    return (np.sum(vbs * xi**a_ji))**(5/3) / (xi * np.sum(m_star * vbs * xi**a_ji))


def f_cont_cmw(xi, xi_ind, cs, vbs):
    """
    Function under the integral of Eq.11 (the exact solution for t^*_i in the continuum regime) for a mixture with
    the same molecular weight.
    :param xi: the MFR of the bin in question
    :param xi_ind: the index of the bin
    :param cs: the saturation concentrations of the compounds in the mixture (array); does not need to be log spaced
    :param vbs: the mass fraction of compounds in the mixture (array length of cs)
    :return: the value of the function under the integral of Eq.
    """
    a_ji = cs/cs[xi_ind]

    return (np.sum(vbs*xi**a_ji))**(2/3) / xi


def f_cont_approx(xi, bi):
    """
    Function under the integral of Eq.15 (the approximate solution for t^*_i in the continuum regime)
    :param xi: the MFR of the bin in question
    :param bi: the mass fraction of the material in the lower volatility bins divided by the mass fraction of the bin
    :return: the value of the function under the integral of Eq.15
    """
    return (xi + bi)**(2/3) / xi


def calc_t_eff_cont(x_final, xi_ind, cs, vbs, mw):
    """
    Function to calculate the effective time needed to reach a certain MFR in the continuum regime using numerical
    integration.
    :param x_final: the MFR to reach
    :param xi_ind: the index of the bin for which the effective time is to be calculated
    :param cs: the saturation concentrations of the compounds in the mixture (array); does not need to be log spaced
    :param vbs: the mass fraction of compounds in the mixture (array length of cs)
    :param mw: the molecular weight of compounds in the mixture (array length of cs)
    :return: effective time to reach MFR = x_final
    """
    res = quad(f_cont, x_final, 1, args=(xi_ind, cs, vbs, mw))
    return res[0]


def calc_t_eff_cont_cmw(x_final, xi_ind, cs, vbs):
    """
    Function to calculate the effective time needed to reach a certain MFR in the continuum regime using numerical
    integration for a mixture with the same molecular weight.
    :param x_final: the MFR to reach
    :param xi_ind: the index of the bin for which the effective time is to be calculated
    :param cs: the saturation concentrations of the compounds in the mixture (array); does not need to be log spaced
    :param vbs: the mass fraction of compounds in the mixture (array length of cs)
    :return: effective time to reach MFR = x_final
    """
    res = quad(f_cont_cmw, x_final, 1, args=(xi_ind, cs, vbs))
    return res[0]


def calc_t_eff_appr_num(x_final, xi_ind, vbs):
    """
    Function to calculate the effective time needed to reach a certain MFR in the continuum regime using numerical
    integration of the approximate formula (Eq.15).
    :param x_final: the MFR to reach
    :param xi_ind: the index of the bin for which the effective time is to be calculated
    :param vbs: the mass fraction of compounds in the mixture (array length of cs);
    :return: effective time to reach MFR = x_final, approximated using Eq.15.
    """
    if xi_ind == 0:
        bi = 0
    else:
        bi = np.sum(vbs[:xi_ind])/vbs[xi_ind]
    res = quad(f_cont_approx, x_final, 1, args=(bi))
    return res[0] * vbs[xi_ind]**(2/3)


def calc_t_eff_appr(x_final, xi_ind, vbs):
    """
    Function to calculate the effective time needed to reach a certain MFR in the continuum regime using analytical
    solution of the approximate formula (Eq.18).
    :param x_final: the MFR to reach
    :param xi_ind: the index of the bin for which the effective time is to be calculated
    :param vbs: the mass fraction of compounds in the mixture (array length of cs);
    :return: effective time to reach MFR = x_final, approximated using Eq.15.
    """
    if xi_ind == 0:
        res = 1.5 * (1 - x_final**(2/3))
    else:
        bi = np.sum(vbs[:xi_ind])/vbs[xi_ind]
        y1 = (1 + 1/bi)**(1/3)
        y2 = (1 + x_final/bi)**(1/3)
        res = bi**(2/3) * (0.5 * np.log(x_final) + 1.5 * np.log((y1 - 1)/(y2 - 1))
                           + np.sqrt(3) * (np.arctan((2*y1 + 1)/np.sqrt(3)) - np.arctan((2*y2 + 1)/np.sqrt(3))))
        res += 1.5 * ((bi + 1)**(2/3) - (bi + x_final)**(2/3))
    return res * vbs[xi_ind]**(2/3)


def f_fm(xi, xi_ind, cs, vbs, mw):
    """
    Function under the integral of Eq.11 (the exact solution for t^*_i in the free molecular regime)
    :param xi: the MFR of the bin in question
    :param xi_ind: the index of the bin for which the effective time is to be calculated
    :param cs: the saturation concentrations of the compounds in the mixture (array); does not need to be log spaced
    :param vbs: the mass fraction of compounds in the mixture (array length of cs)
    :param mw: the molecular weight of compounds in the mixture (array length of cs)
    :return: the value of the function under the integral of Eq.
    """
    m_star = mw/mw[xi_ind]
    a_ji = cs/cs[xi_ind]
    return (np.sum(vbs * xi**a_ji))**(4/3) / (xi * np.sum(m_star * vbs * xi**a_ji))


def f_fm_approx(xi, bi):
    """
    Function under the integral of Eq.15 (the approximate solution for t^*_i in the continuum regime)
    :param xi: the MFR of the bin in question
    :param bi: the mass fraction of the material in the lower volatility bins divided by the mass fraction of the bin
    :return: the value of the function under the integral of Eq.15
    """
    return (xi + bi)**(1/3) / xi


def calc_t_eff_fm(x_final, xi_ind, cs, vbs, mw):
    """
    Function to calculate the effective time needed to reach a certain MFR in the continuum regime using numerical
    integration.
    :param x_final: the MFR to reach
    :param xi_ind: the index of the bin for which the effective time is to be calculated
    :param cs: the saturation concentrations of the compounds in the mixture (array); does not need to be log spaced
    :param vbs: the mass fraction of compounds in the mixture (array length of cs)
    :param mw: the molecular weight of compounds in the mixture (array length of cs)
    :return: effective time to reach MFR = x_final
    """
    res = quad(f_fm, x_final, 1, args=(xi_ind, cs, vbs, mw))
    return res[0]


def calc_t_eff_fm_appr_num(x_final, xi_ind, vbs):
    """
    Function to calculate the effective time needed to reach a certain MFR in the continuum regime using numerical
    integration of the approximate formula (Eq.15).
    :param x_final: the MFR to reach
    :param xi_ind: the index of the bin for which the effective time is to be calculated
    :param vbs: the mass fraction of compounds in the mixture (array length of cs);
    :return: effective time to reach MFR = x_final, approximated using Eq.15.
    """
    if xi_ind == 0:
        bi = 0
    else:
        bi = np.sum(vbs[:xi_ind])/vbs[xi_ind]
    res = quad(f_fm_approx, x_final, 1, args=(bi))
    return res[0] * vbs[xi_ind]**(1/3)


def calc_t_eff_fm_appr(x_final, xi_ind, vbs):
    """
    Function to calculate the effective time needed to reach a certain MFR in the continuum regime using analytical
    solution of the approximate formula (Eq.18).
    :param x_final: the MFR to reach
    :param xi_ind: the index of the bin for which the effective time is to be calculated
    :param vbs: the mass fraction of compounds in the mixture (array length of cs);
    :return: effective time to reach MFR = x_final, approximated using Eq.15.
    """
    if xi_ind == 0:
        res = 3 * (1 - x_final**(1/3))
    else:
        bi = np.sum(vbs[:xi_ind])/vbs[xi_ind]
        y1 = (1 + 1/bi)**(1/3)
        y2 = (1 + x_final/bi)**(1/3)
        res = bi ** (1/3) * (0.5 * np.log(x_final) + 1.5 * np.log((y1 - 1) / (y2 - 1))
                             + np.sqrt(3) * (np.arctan((2 * y1 + 1) / np.sqrt(3)) -
                                             np.arctan((2 * y2 + 1) / np.sqrt(3))))
        res += 2*1.5 * ((bi + 1) ** (1 / 3) - (bi + x_final) ** (1 / 3))

    return res * vbs[xi_ind]**(1/3)


if __name__ == "__main__":
    t = np.logspace(-3, 8, 100)  # times to calculate xi for using the classical method
    Cs = np.logspace(-3, 3, 7)  # saturation concentrations
    f = np.array([1, 2, 3, 4, 3, 2, 1])  # vbs
    f = f/f.sum()  # normalize the vbs
    d_start = 300  # the initial diameter

    Mw = np.ones_like(Cs)*146.1  # molecular weights. Equal for the comparison, as the classical method is not implemented for vaiable MW yet

    x_final = np.linspace(1e-3, 0.999,  100)  # final xi values for calculations using the new method

    char_t = char_time(Cs, d_start)  # characteristic times

    ref_bin = 3  # the reference bin

    # to compare the classical anf the new approach run the code below, then plot it (will require importing matplotlib
    xi_class_trans = calc_xi_trans(f, char_t, t, d_start)
    t_eff_trans = np.array([calc_t_eff_trans(x, ref_bin, Cs, f, Mw, d_start) for x in x_final])

    # xi_class_cont = calc_xi_cont(f, char_t, t, d_start)
    # t_eff_cont = np.array([calc_t_eff_cont(x, ref_bin, Cs, f, Mw) for x in x_final])

    # this is for plotting a comparison:
    # plt.semilogx(t, xi_class_trans, label='classical')
    # plt.semilogx(t*char_t[ref_bin], x_final, 'x', label='new')
    # plt.legend()
    # plt.show()