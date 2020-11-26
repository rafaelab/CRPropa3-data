import numpy as np
import interactionRate
import photonField
import os





eV = 1.60217657e-19
lgamma = np.linspace(6, 16, 251, endpoint = True)  # tabulated Lorentz factors
gamma = 10 ** lgamma
resDir = 'data/PhotoPionProduction'

# ----------------------------------------------------
# Load proton and neutron cross sections [1/m^2] for tabulated energies [J]
# Truncate to largest length 2^i + 1 for Romberg integration
# ----------------------------------------------------
d = np.loadtxt('tables/PPP/xs_proton.txt')
eps1 = d[0, :2049] * 1e9 * eV  # [J]
xs1 = d[1, :2049] * 1e-34  # [m^2]

d = np.loadtxt('tables/PPP/xs_neutron.txt')
eps2 = d[0, :2049] * 1e9 * eV  # [J]
xs2 = d[1, :2049] * 1e-34  # [m^2]


# ----------------------------------------------------
# ----------------------------------------------------
def compute_rates(field):
    """
    This function calculates the interaction rates at z=0.
    When redshift information is available, the redshift-dependent rates are computed.
    """
    # output folder
    if not os.path.exists(resDir):
        os.makedirs(resDir)

    # calculate interaction rates at z=0, default option
    r1 = interactionRate.calc_rate_eps(eps1, xs1, gamma, field)
    r2 = interactionRate.calc_rate_eps(eps2, xs2, gamma, field)

    fname = '%s/rate_%s.txt' % (resDir, field.name)
    data = np.c_[lgamma, r1, r2]
    fmt = '%.2f\t%.6e\t%.6e'
    header  = 'Photopion production rate at z=0.'
    header += 'Format: log10(gamma)\t1/lambda_proton [1/Mpc]\t1/lambda_neutron [1/Mpc]'
    np.savetxt(fname, data, fmt = fmt, header = header)

    # calculate redshift-dependent interaction rates
    redshifts = field.redshift
    if redshifts is not None:
        # thin out long redshift lists (e.g., Finke10)
        n_zmax = 100
        if len(redshifts) > n_zmax:
            thin = int(len(redshifts) / n_zmax)
            redshifts = redshifts[::thin]

        data = []
        for z in redshifts:
            r1 = interactionRate.calc_rate_eps(eps1, xs1, gamma, field, z)
            r2 = interactionRate.calc_rate_eps(eps2, xs2, gamma, field, z)
            data.append(np.c_[[z] * len(lgamma), lgamma, r1, r2])

        data = np.nan_to_num(np.concatenate([d for d in data], axis = 0))
        fname = '%s/rate_%s.txt' % (resDir, field.name.replace('IRB', 'IRBz'))
        fmt = '%.2f\t%.2f\t%.6e\t%.6e'
        header  = 'Photopion production rate for the %s\n (redshift dependent).' % field.info
        header += 'Format: z\tlog10(gamma)\t1/lambda_proton [1/Mpc]\t1/lambda_neutron [1/Mpc]' 
        np.savetxt(fname, data, fmt = fmt, header = header)


# ----------------------------------------------------
# ----------------------------------------------------
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

    for field in fields:
        print(field.name)
        compute_rates(field)
