import numpy as np
from .constants import bohr2ang,hartree2J,hartree2cm,amu2kg,ang2m,h,hbar,c,cmeter,hz2cm,convert
from .nc_analysis import cartesian_freq,internal_freq

def cubic_from_internals(cubic, hess, m, B1, B2):
    """
    Computes cubic normal coordinate force constants in cm-1 from internal coordinate derivatives.

    Parameters
    ----------
    cubic : 3d array
        Internal coordinate third derivative tensor (analogue of Hessian) Hartree/Angstrom^3
    hess : 2d array
        Internal coordinate Hessian in Hartree/Angstrom^2
    m : 1d array
        Masses of the atoms in amu (length is number of atoms)
    B1 : 2d array
        1st order B tensor which relates internal coordinates to cartesian coordinates
    B2 : 3d array
        2nd order B tensor which relates internal coordinates to cartesian coordinates

    Returns
    -------
    Cubic force constants in cm-1 corresponding to dimensionless normal coordinates.

    Equations from Hoy, Mills, and Strey, 1972.
    """
    harmfreqs, junk, L = internal_freq(hess, B1, m)
    M = np.sqrt(1 / np.repeat(m,3))
    inv_trans_L = np.linalg.inv(L).T
    little_l = np.einsum('a,ia,ir->ar', M, B1, inv_trans_L)
    L1 = np.einsum('ia,a,ar->ir', B1, M, little_l)
    L2 = np.einsum('iab,a,ar,b,bs->irs', B2, M, little_l, M, little_l)
    term1 = np.einsum('ijk,ir,js,kt->rst', cubic, L1, L1, L1)
    term2 = np.einsum('ij, irs, jt->rst', hess, L2, L1)
    term3 = np.einsum('ij, irt, js->rst', hess, L2, L1)
    term4 = np.einsum('ij,ist,jr->rst', hess, L2, L1)
    nc_cubic = term1 + term2 + term3 + term4                              # UNITS: Hartree / Ang^3 amu^3/2
    frac = (hbar / (2*np.pi*cmeter))**(3/2)
    nc_cubic *= (1 / (ang2m**3 * amu2kg**(3/2)))                          # UNITS: Hartree / m^3 kg^3/2
    nc_cubic *= frac                                                      # UNITS: Hartree / m^(3/2) 
    nc_cubic *= (1 / 100**(3/2))                                          # UNITS: Hartree cm-1 ^ (3/2)
    # Multiply each element by appropriate 3 harmonic frequencies
    omega = harmfreqs**(-1/2)
    nc_cubic = np.einsum('ijk,i,j,k->ijk', nc_cubic, omega, omega, omega) # UNITS: Hartree
    nc_cubic *= -hartree2cm                                               # UNITS: cm-1
    return nc_cubic

def cubic_from_cartesians(cubic, hess):
    """ 
    Computes cubic normal coordinate force constants in cm-1 from cartesian coordinate derivatives.

    Parameters
    ----------
    cubic : 3d array
        Cartesian coordinate third derivative tensor in Hartree/Angstrom^3
    hess : 2d array
        Cartesian coordinate Hessian in Hartree/Angstrom^2

    Returns
    -------
    Cubic force constants in cm-1 corresponding to dimensionless normal coordinates.

    Equations from Gaw, Yamaguchi, Schaefer, Handy 1986.
    """
    f, junk, L = cartesian_freq(hess, m)
    nc_cubic = np.einsum('ir,js,kt,ijk->rst', L, L, L, cubic)            # UNITS: Hartree/ Ang^3 amu^(3/2)
    frac = (hbar / (2*np.pi*cmeter))**(3/2)
    nc_cubic *= (1 / (ang2m**3 * amu2kg**(3/2)))                         # UNITS: Hartree / m^3 kg^3/2
    nc_cubic *= frac                                                     # UNITS: Hartree / m^(3/2) 
    nc_cubic *= (1 / 100**(3/2))                                         # UNITS: Hartree cm-1 ^ (3/2)
    # Multiply each element by appropriate 3 harmonic frequencies
    #omega = np.array([1737.31536,3987.9131,4144.72382])**(-1/2)
    omega = harmfreqs**(-1/2)
    nc_cubic = np.einsum('ijk,i,j,k->ijk', nc_cubic, omega, omega, omega)# UNITS: Hartree
    nc_cubic *= hartree2cm               #TODO minus sign ???           # UNITS: cm-1
    return nc_cubic



