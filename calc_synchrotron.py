import numpy as np
from scipy.special import kv
from scipy import integrate

def synchrotron_spectrum(xval):
	"""
	Calculate cumulative synchrotron spectrum.
	Follows:
	  J.~D. Jackson, Classical Electrondynamics.
	  Wiley, 3rd ed., p. 681, eq. 14.91

	F(x) = (\int_{x}^{\infinity} K_{5/3}(x') dx') 
	for x = E_gamma / E_critical, where E_critical = hbar * 3/2 * c/rho * (E/(mc**2))**2
	E_gamma : Energy of synchrotron photon
	E       : Energy of particle
	rho     : gyroradius 

	Returns : 
	  The cumulative synchrotron function
	"""
	F = np.zeros(len(xval))
	for i, value in enumerate(xval):
		a = xval[i:]
		F[i] = integrate.trapz(x = a, y = kv(5. / 3., a))
	for i,value in enumerate(xval):
		b = integrate.cumtrapz(x = xval, y = xval * F, initial = 0)
	return b / b[-1]


# ----------------------------------------------------------------
# Cumulative differential synchrotron spectrum (comp. J.D. Jackson(p. 681, formula 14.91))
# ----------------------------------------------------------------

x = np.logspace(-8, 2, 1001)
cdf = synchrotron_spectrum(x)
lx = np.log10(x)
data = np.c_[lx, cdf]
fname  = 'data/Synchrotron/spectrum.txt'
header = 'x\t: photon frequency to critical frequency fraction\nlog10(x)\tCDF\n'
fmt = '%3.2f\t%7.6e'
np.savetxt(fname, data, fmt = fmt, header = header)
