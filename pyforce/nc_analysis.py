"""
Perform normal coordinate analysis in cartesian or internal coordinates
"""
import numpy as np
from scipy.linalg import fractional_matrix_power
from .constants import convert

def cartesian_freq(hess, m):
    """
    Do normal coordinate analysis in Cartesian coordinates
    
    Parameters
    ----------
    hess : 2d array
        The Hessian in Cartesian coordinates (not mass weighted)
    m : 1d array
        The masses of the atoms in amu. Has size = natoms.
    Returns
    -------
    freqs : Harmonic frequencies in wavenumbers (cm-1).
    LMW   : Normal coordinate eigenvectors from mass-weighted Hessian
    Lmw   : Normal coordinate eigenvectors with massweighting partially removed: Lmw = m^-1/2 * LMW
    """
    m = np.repeat(m,3)
    M = 1 / np.sqrt(m)
    diagM = np.diag(M)
    Hmw = diagM.dot(hess).dot(diagM)
    lamda, LMW = np.linalg.eig(Hmw)
    idx = lamda.argsort()
    lamda = lamda[idx]
    LMW = LMW[:,idx]
    freqs = np.sqrt(lamda) * convert
    Lmw = np.einsum('i,ir->ir', M,LMW)
    return freqs[6:], LMW[:,6:], Lmw[:,6:]

def internal_freq(hess, B1, m):
    """
    Do normal coordinate analysis with internal coordinates via GF method. 

    Parameters
    ----------
    hess : ndarray 
        NumPy array of Hessian in internal coordinates, Hartrees/Angstrom^2
    B1 : ndarray
        Numpy array 1st order B tensor corresponding to internal coordinate definitions in Hessian 
    m : ndarray
        Numpy array of masses of each atom in amu. Size is number of atoms. 
    Returns 
    -------
    Frequencies in wavenumbers, normalized normal coordinates, and mass-weighted (1/sqrt(amu)) normal coordinates 
    All values sorted to be in order of increasing energy of the frequencies.
    """
    m = np.repeat(m,3)
    M = 1 / m
    G = np.einsum('in,jn,n->ij', B1, B1, M)
    Gt = fractional_matrix_power(G, 0.5)
    Fp = Gt.dot(hess).dot(Gt)
    lamda, L = np.linalg.eig(Fp)
    mwL = Gt.dot(L)
    # Return Frequencies and 'L matrix' (mass weighted) in increasing order
    idx = lamda.argsort()
    lamda = lamda[idx]
    freqs = np.sqrt(lamda) * convert
    return freqs, L[:,idx], mwL[:,idx]


