from __future__ import division
import numpy as np
import interactionRate
import os
import gitHelp as gh
from calc_all import fields_cmbebl, fields_urb
from units import eV, mass_electron, c_light, sigma_thomson, alpha_finestructure

me2 = (mass_electron*c_light**2.) ** 2  # squared electron mass [J^2/c^4]

def sigmaPP(s):
    """ Pair production cross section (Breit-Wheeler), see Lee 1996 """
    smin = 4 * me2
    if (s < smin):
        return 0.

    b = np.sqrt(1 - smin / s)
    return sigma_thomson * 3 / 16 * (1 - b**2) * ((3 - b**4) * (np.log1p(b) - np.log1p(-b)) - 2 * b * (2 - b**2))


def sigmaDPP(s):
    """ Double-pair production cross section, see R.W. Brown eq. (4.5) with k^2 = q^2 = 0 """
    smin = 16 * me2
    if (s < smin):
        return 0

    return 6.45E-34 * (1 - smin / s)**6


def sigmaICS(s):
    """ Inverse Compton scattering cross sections, see Lee 1996 """
    smin = me2
    if (s < smin):  # numerically unstable close to smin
        return 0

    # note: formula unstable for (s - smin) / smin < 1E-5
    b = (s - smin) / (s + smin)
    A = 2 / b / (1 + b) * (2 + 2 * b - b**2 - 2 * b**3)
    B = (2 - 3 * b**2 - b**3) / b**2 * (np.log1p(b) - np.log1p(-b))
    return sigma_thomson * 3 / 8 * smin / s / b * (A - B)


def sigmaTPP(s):
    """ Triplet-pair production cross section, see Lee 1996 """
    beta = 28 / 9 * np.log(s / me2) - 218 / 27
    if beta < 0:
        return 0
    
    return sigma_thomson * 3 / 8 / np.pi * alpha_finestructure * beta


def getTabulatedXS(sigma, skin):
    """ Get crosssection for tabulated s_kin """
    if sigma in (sigmaPP, sigmaDPP):  # photon interactions
        return np.array([sigma(s) for s in skin])
    if sigma in (sigmaTPP, sigmaICS):  # electron interactions
        return np.array([sigma(s) for s in skin + me2])
    return False


def getSmin(sigma):
    """ Return minimum required s_kin = s - (mc^2)^2 for interaction """
    return {sigmaPP: 4 * me2,
            sigmaDPP: 16 * me2,
            sigmaTPP: np.exp((218 / 27) / (28 / 9)) * me2 - me2,
            sigmaICS: 1e-40 * me2
            }[sigma]


def getEmin(sigma, field):
    """ Return minimum required cosmic ray energy for interaction *sigma* with *field* """
    return getSmin(sigma) / 4 / field.getEmax()


def process(sigma, field, name):
    """ 
        calculate the interaction rates for a given process on a given photon field 

        sigma : crossection (function) of the EM-process
        field : photon field as defined in photonField.py
        name  : name of the process which will be calculated. Necessary for the naming of the data folder
    """

    # output folder
    folder = 'data/' + name
    if not os.path.exists(folder):
        os.makedirs(folder)

    # tabulated energies, limit to energies where the interaction is possible
    Emin = getEmin(sigma, field)
    E = np.logspace(9, 23, 281) * eV
    E = E[E > Emin]
    
    # -------------------------------------------
    # calculate interaction rates
    # -------------------------------------------
    # tabulated values of s_kin = s - mc^2
    # Note: integration method (Romberg) requires 2^n + 1 log-spaced tabulation points
    s_kin = np.logspace(4, 23, 2 ** 18 + 1) * eV**2
    xs = getTabulatedXS(sigma, s_kin)
    rate = interactionRate.calc_rate_s(s_kin, xs, E, field)

    # save
    fname = folder + '/rate_%s.txt' % field.name
    data = np.c_[np.log10(E / eV), rate]
    fmt = '%.2f\t%8.7e'
    try:
        git_hash = gh.get_git_revision_hash()
        header = ("%s interaction rates\nphoton field: %s\n"% (name, field.info)
                  +"Produced with crpropa-data version: "+git_hash+"\n"
                  +"log10(E/eV), 1/lambda [1/Mpc]" )
    except:
        header = ("%s interaction rates\nphoton field: %s\n"% (name, field.info)
                  +"log10(E/eV), 1/lambda [1/Mpc]")
    np.savetxt(fname, data, fmt=fmt, header=header)

    # -------------------------------------------
    # calculate cumulative differential interaction rates for sampling s values
    # -------------------------------------------
    # find minimum value of s_kin
    skin1 = getSmin(sigma)  # s threshold for interaction
    skin2 = 4 * field.getEmin() * E[0]  # minimum achievable s in collision with background photon (at any tabulated E)
    skin_min = max(skin1, skin2)

    # tabulated values of s_kin = s - mc^2, limit to relevant range
    # Note: use higher resolution and then downsample
    skin = np.logspace(4, 23, 380000 + 1) * eV**2
    skin = skin[skin > skin_min]

    xs = getTabulatedXS(sigma, skin)
    rate = interactionRate.calc_rate_s(skin, xs, E, field, cdf=True)

    # downsample
    skin_save = np.logspace(4, 23, 190 + 1) * eV**2
    skin_save = skin_save[skin_save > skin_min]
    rate_save = np.array([np.interp(skin_save, skin, r) for r in rate])

    # save
    data = np.c_[np.log10(E / eV), rate_save]  # prepend log10(E/eV) as first column
    row0 = np.r_[0, np.log10(skin_save / eV**2)][np.newaxis]
    data = np.r_[row0, data]  # prepend log10(s_kin/eV^2) as first row

    fname = folder + '/cdf_%s.txt' % field.name
    fmt = '%.2f' + '\t%6.5e' * np.shape(rate_save)[1]
    try:
        git_hash = gh.get_git_revision_hash()
        header = ("%s cumulative differential rate\nphoton field: %s\n"% (name, field.info)
                  +"Produced with crpropa-data version: "+git_hash+"\n"
                  +"log10(E/eV), d(1/lambda)/ds_kin [1/Mpc/eV^2] for log10(s_kin/eV^2) as given in first row" )
    except:
        header = ("%s cumulative differential rate\nphoton field: %s\n"% (name, field.info)
                  +"log10(E/eV), d(1/lambda)/ds_kin [1/Mpc/eV^2] for log10(s_kin/eV^2) as given in first row")
    np.savetxt(fname, data, fmt=fmt, header=header)

    del data, rate, skin, skin_save, rate_save

if __name__ == "__main__":

    for field in fields_cmbebl+fields_urb:
        print(field.name)
        process(sigmaPP, field, 'EMPairProduction')
        process(sigmaDPP, field, 'EMDoublePairProduction')
        process(sigmaTPP, field, 'EMTripletPairProduction')
        process(sigmaICS, field, 'EMInverseComptonScattering')
