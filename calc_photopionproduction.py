import os
import numpy as np

import interactionRate
import photonField

try:
    from joblib import Parallel, delayed
    _parallel = True
except:
    _parallel = False


# ____________________________________________________________________________________________
#
# units
eV = 1.60217657e-19

# output directory
resDir = 'data/PhotoPionProduction'
if not os.path.exists(resDir):
    os.makedirs(resDir)

# Load proton and neutron cross sections [1/m^2] for tabulated energies [J]
# Truncate to largest length 2^i + 1 for Romberg integration
d = np.loadtxt('tables/PPP/xs_proton.txt')
eps1 = d[:2049, 0] * 1e9 * eV  # [J]
xs1  = d[:2049, 1] * 1e-34  # [m^2]

d = np.loadtxt('tables/PPP/xs_neutron.txt')
eps2 = d[:2049, 0] * 1e9 * eV  # [J]
xs2  = d[:2049, 1] * 1e-34  # [m^2]


# tabulated Lorentz factors
lgamma = np.linspace(6, 16, 251, endpoint = True)  
gamma = 10 ** lgamma

# ____________________________________________________________________________________________
#
def compute_interaction_rates(field):
    """
    This function calculates the interaction rates at z=0.
    When redshift information is available, the redshift-dependent rates are computed.
    """
    redshifts = field.redshift
    if redshifts is not None:   # calculate redshift-dependent interaction rates
        # thin out long redshift lists (e.g., Finke10)
        n_zmax = 100
        if len(redshifts) > n_zmax:
            thin = int(len(redshifts) / n_zmax)
            redshifts = redshifts[::thin]

        data = []
        for i, z in enumerate(redshifts):
            r1 = interactionRate.calc_rate_eps(eps1, xs1, gamma, field, z)
            r2 = interactionRate.calc_rate_eps(eps2, xs2, gamma, field, z)
            data.append(np.c_[[z] * len(lgamma), lgamma, r1, r2]) 
        data = np.nan_to_num(np.concatenate([d for d in data], axis = 0)) 

        fname = '%s/rate_%s.txt' % (resDir, field.name.replace('IRB', 'IRBz'))
        fmt = '%.2f\t%.2f\t%.6e\t%.6e'
        header  = 'Photopion production rate for the %s\n (redshift dependent).' % field.info
        header += 'Format: z\tlog10(gamma)\t1/lambda_proton [1/Mpc]\t1/lambda_neutron [1/Mpc]' 
        np.savetxt(fname, data, fmt = fmt, header = header)

    # calculate interaction rates at z=0, default option
    r1 = interactionRate.calc_rate_eps(eps1, xs1, gamma, field)
    r2 = interactionRate.calc_rate_eps(eps2, xs2, gamma, field)

    fname = '%s/rate_%s.txt' % (resDir, field.name)
    data = np.c_[lgamma, r1, r2]
    fmt = '%.2f\t%.6e\t%.6e'
    header  = 'Photopion production rate at z=0.'
    header += 'Format: log10(gamma)\t1/lambda_proton [1/Mpc]\t1/lambda_neutron [1/Mpc]'
    np.savetxt(fname, data, fmt = fmt, header = header)

# ____________________________________________________________________________________________
#
if __name__ == '__main__':
    
    fields = [
        photonField.CMB(),
        photonField.EBL_Kneiske04(),
        photonField.EBL_Stecker05(),
        photonField.EBL_Franceschini08(),
        photonField.EBL_Finke10(),
        photonField.EBL_Dominguez11(),
        photonField.EBL_Gilmore12(),
        photonField.EBL_Stecker16('upper'),
        photonField.EBL_Stecker16('lower'),
        photonField.URB_Nitu21(),
        photonField.URB_Protheroe96()
    ]

    # Run in parallel if joblib is available
    if _parallel:
        ncores = -1 # use all cores
        Parallel(n_jobs = ncores)(delayed(compute_interaction_rates)(field) for field in fields)
    else:
        for field in fields:
            compute_interaction_rates(field)

# ____________________________________________________________________________________________
#