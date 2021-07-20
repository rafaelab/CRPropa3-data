import os
import numpy as np

import photonField
import interactionRate 

try:
    from joblib import Parallel, delayed
    _parallel = True
except:
    _parallel = False



# ____________________________________________________________________________________________
#
# units
eV = 1.60217657e-19

# output folder
folder = 'data/ElasticScattering'
if not os.path.exists(folder):
    os.makedirs(folder)

# ____________________________________________________________________________________________
#
# tabulated UHECR Lorentz-factors
gamma = np.logspace(6, 14, 201)  

# load cross section data from TALYS
ddir = 'tables/PD_Talys1.8_Khan/'
eps = np.genfromtxt(ddir + 'eps_elastic.txt') * eV * 1e6  # nuclear rest frame photon energies [J]
data = np.genfromtxt(ddir + 'xs_elastic.txt', dtype=[('Z', int), ('N', int), ('xs', '%if8' % len(eps))])

# only consider TALYS cross sections for A >= 12
idx = (data['Z'] + data['N']) >= 12
data = data[idx]

# factor out the principal scaling given by the TRK formula: sigma_int ~ Z*N/A
data['xs'] /= (data['Z'] * data['N'] / (data['Z'] + data['N']))[:, np.newaxis]

# pad cross sections to next larger 2^n + 1 tabulation points for Romberg integration
eps = interactionRate.romb_pad_logspaced(eps, 513)
xs = np.array([interactionRate.romb_pad_zero(x, 513) for x in data['xs']]) * 1e-31


# ____________________________________________________________________________________________
#
def compute_interaction_rates(field):
    """
    """
    # calculate the interaction rate, averaged over all isotopes
    rate = np.mean([interactionRate.calc_rate_eps(eps, x, gamma, field) for x in xs], axis=0)
    fname = folder + '/rate_%s.txt' % field.name.split('_')[0]
    header = 'Average interaction rate for elastic scattering of %s photons off nuclei\nScale with Z*N/A for nuclei\n1/lambda [1/Mpc] for log10(gamma) = 6-14 in 201 steps' % field.info
    np.savetxt(fname, rate, fmt='%g', header=header)

    # calculate CDF for background photon energies, averaged over all isotopes
    cdf = np.zeros((len(gamma), len(eps)))
    for x in xs:
        r = interactionRate.calc_rate_eps(eps, x, gamma, field, cdf=True)
        cdf += r / np.max(r, axis=1, keepdims=True)
    cdf /= len(data)
    cdf = np.nan_to_num(cdf)

    fname = folder + '/cdf_%s.txt' % field.name.split('_')[0]
    data1 = np.c_[np.log10(gamma), cdf]
    fmt = '%g' + '\t%g' * len(eps)
    header = '# Average CDF(background photon energy) for elastic scattering with the %s\n# log10(gamma), (1/lambda)_cumulative for eps = log10(2 keV) - log10(263 MeV) in 513 steps' % field.info
    np.savetxt(fname, data1, fmt=fmt, header=header)


# ____________________________________________________________________________________________
#
if __name__ == '__main__':
    
    fields = [
        photonField.CMB(),
        photonField.EBL_Gilmore12(),
        photonField.URB_Nitu21()
    ]

    fields = [photonField.URB_Nitu21()]

    # Run in parallel if joblib is available
    if _parallel:
        ncores = -1 # use all cores
        Parallel(n_jobs = ncores)(delayed(compute_interaction_rates)(field) for field in fields)
    else:
        for field in fields:
            compute_interaction_rates(field)

# ____________________________________________________________________________________________
#