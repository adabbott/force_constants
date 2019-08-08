import numpy as np

bohr2ang = 0.529177249
hartree2J = 4.3597443e-18
hartree2cm = 219474.63
amu2kg = 1.6605389e-27
ang2m = 1e-10
h = 6.6260701510e-34   # Plancks in J s
hbar = 1.054571817e-34 # Reduced Plancks constant J s
c = 29979245800.0 # speed of light in cm/s
cmeter = 299792458 # speed of light in cm/s
hz2cm = 3.33565e-11

# For converting harmonic frequencies
convert = np.sqrt(hartree2J/(amu2kg*ang2m*ang2m))/(c*2*np.pi)

