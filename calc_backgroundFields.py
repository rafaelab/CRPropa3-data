import numpy as np

from photonField import *



# ____________________________________________________________________________________________
#
# general settings
resDir = 'data/Scaling/'

# units
cm3 = 1e-6  # [m^3]
eV = 1.602176634e-19 # [J]

# ____________________________________________________________________________________________
#
def generatePhotonFieldFiles(field):
    """
    ABOUT
    -----
    Function to generate the photon field files for CRPropa 3.1.6.

    INPUT
    -----
    . field: background photon field from photonField.py
    """
    if 'IRB' in field.name:
        energy = field.data[0][0]
    else:
        emin = np.log10(field.getEmin())
        emax = np.log10(field.getEmax())
        energy = np.logspace(emin, emax, 101, endpoint = True) 

    fDensity  = resDir + field.name + '_photonDensity.txt'
    fEnergy   = resDir + field.name + '_photonEnergy.txt'
    fRedshift = resDir + field.name + '_redshift.txt'

    np.savetxt(fEnergy, energy, fmt = '%7.6e')

    if field.redshift is not None:
        density = np.zeros(len(energy) * len(field.redshift))
        k = 0
        redshift = field.redshift
        if len(redshift) > 100:
            redshift = redshift[::10]
        for e in energy:
            for z in redshift:
                density[k] = field.getDensity(e, z) * e
                k += 1
        np.savetxt(fDensity, density, fmt = '%7.6e')
        np.savetxt(fRedshift, field.redshift, fmt = '%4.3e')
    else:
        density = field.getDensity(energy) * energy
        np.savetxt(fDensity, density, fmt = '%7.6e')

# ____________________________________________________________________________________________
#
if __name__ == '__main__':
    
    urb1 = URB_Protheroe96()
    urb2 = URB_Fixsen11()
    urb3 = URB_Nitu21()
    urbs = [urb1, urb2, urb3]

    ebl01 = EBL_Gilmore12()
    ebl02 = EBL_Dominguez11()
    ebl03 = EBL_Franceschini08()
    ebl04 = EBL_Finke10()
    ebl05 = EBL_Stecker16('lower')
    ebl06 = EBL_Stecker16('upper')
    ebl07 = EBL_Stecker05()
    ebl08 = EBL_Kneiske04()
    ebls = [ebl01, ebl02, ebl03, ebl04, ebl05, ebl06, ebl07, ebl08]

    for field in urbs:
        generatePhotonFieldFiles(field)

    for field in ebls:
        generatePhotonFieldFiles(field)

# ____________________________________________________________________________________________
#
