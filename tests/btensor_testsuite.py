import psi4
import torch
import numpy as np
from pyforce import btensors
from pyforce import intcos
import optking
import time

torch.set_printoptions(precision=5)
bohr2ang = 0.529177249
psi4.core.be_quiet()

def test_molecule(psi_geom,ad_intcoords, optking_intcoords):
    """ 
    Tests autodiff-OptKing hack versus optking

    Parameters: 
    ---------
    psi_geom : Psi4 Matrix,Molecule.geometry(),
    ad_intcoords : list of new autodiff optking internal coordinates objects STRE,BEND,TORS
    optking_intcoords : list of optking internal coordinates objects Stre,Bend,Tors
    """
    npgeom = np.array(psi_geom) * bohr2ang
    a = time.time()
    optking_coords = optking.intcosMisc.qValues(optking_intcoords,npgeom)
    B1 = optking.intcosMisc.Bmat(optking_intcoords,npgeom)
    b = time.time()
    geom1 = torch.tensor(npgeom,requires_grad=True)
    B1_new = btensors.compute_btensors(ad_intcoords,geom1,order=1)
    c = time.time()
    print('Same B-Matrix...',torch.allclose(B1_new,torch.tensor(B1)), end=' ')
    print("AutoDiff took {} ms".format(round((c-b)*1000,5)), end=' ')
    print("Optking took {} ms".format( round((b-a)*1000,5)))
    return

h2o = psi4.geometry(
'''
H            0.000000000000     0.000000000000     0.950000000000 
H            0.000000000000     0.872305301500    -0.376275777700 
O            0.000000000000     0.000000000000     0.000000000000 
''')
linear_h2o = psi4.geometry(
'''
H            0.000000000000     0.000000000000     1.950000000000 
H            0.000000000000     0.000000000000     0.050000000000 
O            0.000000000000     0.000000000000     1.000000000000 
''')
ammonia = psi4.geometry(
'''
 N  0.000000  0.0       0.0 
 H  1.584222  0.0       1.12022
 H  0.0      -1.58422  -1.12022
 H -1.584222  0.0       1.12022
 H  0.0       1.58422  -1.12022
 unit au
''')
h2co = psi4.geometry(
'''
C            0.000000000000     0.000000000000    -0.607835855018 
O            0.000000000000     0.000000000000     0.608048883261 
H            0.000000000000     0.942350938995    -1.206389817026 
H            0.000000000000    -0.942350938995    -1.206389817026 
''')
bent_h2co = psi4.geometry(# h2co,but 160 degree dihedral
'''
C            0.011014420656    -0.636416764906     0.000000000000 
O            0.011014420656     0.628402205849     0.000000000000 
H           -0.152976834267    -1.197746817763     0.930040622624 
H           -0.152976834267    -1.197746817763    -0.930040622624 
''')
hooh = psi4.geometry(
'''
H
O 1 0.9
O 2 1.4 1 100.0
H 3 0.9 2 100.0 1 114.0
''')
sf4 = psi4.geometry(
'''
 S  0.00000000  -0.00000000  -0.30618267
 F -1.50688420  -0.00000000   0.56381732
 F  0.00000000  -1.74000000  -0.30618267
 F -0.00000000   1.74000000  -0.30618267
 F  1.50688420   0.00000000   0.56381732
''')
# Use intcos_generate_exit autogenerated internal coordinates for a few big molecules
allene = psi4.geometry(
"""
H  0.0  -0.92   -1.8
H  0.0   0.92   -1.8
C  0.0   0.00   -1.3
C  0.0   0.00    0.0
C  0.0   0.00    1.3
H  0.92  0.00    1.8
H -0.92  0.00    1.8
""")
big = psi4.geometry( 
'''
 C  0.00000000 0.00000000 0.00000000
 Cl 0.19771002 -0.99671665 -1.43703398
 C  1.06037767 1.11678073 0.00000000
 C  2.55772698 0.75685710 0.00000000
 H  3.15117939 1.67114056 0.00000000
 H  2.79090687 0.17233980 0.88998127
 H  2.79090687 0.17233980 -0.88998127
 H  0.75109254 2.16198057 0.00000000
 H -0.99541786 0.44412079 0.00000000
 H  0.12244541 -0.61728474 0.88998127
'''
)

#h2o_autodiff = [intcos.STRE(2,1),intcos.STRE(2,0),intcos.BEND(1,2,0)]
#linear_h2o_autodiff = [intcos.STRE(2,1),intcos.STRE(2,0),intcos.BEND(1,2,0,bendType='LINEAR')]
#ammonia_autodiff = [intcos.STRE(0,1),intcos.STRE(0,2),intcos.STRE(0,3),intcos.STRE(0,4),intcos.BEND(1,0,2),intcos.BEND(1,0,3),intcos.BEND(1,0,4),intcos.BEND(2,0,3), intcos.BEND(2,0,4),intcos.BEND(3,0,4)]
#h2co_autodiff = [intcos.STRE(0,1),intcos.STRE(0,2),intcos.BEND(2,0,1),intcos.STRE(0,3),intcos.BEND(3,0,1),intcos.TORS(3,0,1,2)]
#hooh_autodiff = [intcos.STRE(0,1),intcos.STRE(0,2),intcos.BEND(2,1,0),intcos.STRE(3,2),intcos.BEND(3,2,1),intcos.TORS(3,2,1,0)]
#sf4_autodiff = [intcos.TORS(0,1,2,3),intcos.TORS(1,3,4,0),intcos.TORS(0,2,1,4)]
#allene_autodiff = [intcos.STRE(0, 2),intcos.STRE(1, 2),intcos.STRE(2, 3),intcos.STRE(3, 4),intcos.STRE(4, 5),intcos.STRE(4, 6),intcos.BEND(0, 2, 1),intcos.BEND(0, 2, 3),intcos.BEND(1, 2, 3),intcos.BEND(2, 3, 4),intcos.BEND(2, 3, 4),intcos.BEND(3, 4, 5),intcos.BEND(3, 4, 6),intcos.BEND(5, 4, 6),intcos.TORS(0, 2, 4, 5),intcos.TORS(0, 2, 4, 6),intcos.TORS(1, 2, 4, 5),intcos.TORS(1, 2, 4, 6)]
#big_autodiff = [intcos.STRE(0, 1),intcos.STRE(0, 2),intcos.STRE(0, 8),intcos.STRE(0, 9),intcos.STRE(2, 3),intcos.STRE(2, 7),intcos.STRE(3, 4),intcos.STRE(3, 5),intcos.STRE(3, 6),intcos.BEND(0, 2, 3),intcos.BEND(0, 2, 7),intcos.BEND(1, 0, 2),intcos.BEND(1, 0, 8),intcos.BEND(1, 0, 9),intcos.BEND(2, 0, 8),intcos.BEND(2, 0, 9),intcos.BEND(2, 3, 4),intcos.BEND(2, 3, 5),intcos.BEND(2, 3, 6),intcos.BEND(3, 2, 7),intcos.BEND(4, 3, 5),intcos.BEND(4, 3, 6),intcos.BEND(5, 3, 6),intcos.BEND(8, 0, 9),intcos.TORS(0, 2, 3, 4),intcos.TORS(0, 2, 3, 5),intcos.TORS(0, 2, 3, 6),intcos.TORS(1, 0, 2, 3),intcos.TORS(1, 0, 2, 7),intcos.TORS(3, 2, 0, 8),intcos.TORS(3, 2, 0, 9),intcos.TORS(4, 3, 2, 7),intcos.TORS(5, 3, 2, 7),intcos.TORS(6, 3, 2, 7),intcos.TORS(7, 2, 0, 8),intcos.TORS(7, 2, 0, 9)]
#
h2o_autodiff = [intcos.STRE(2,1),intcos.STRE(2,0),intcos.BEND(1,2,0)]
h2o_optking  = [optking.Stre(2,1),optking.Stre(2,0),optking.Bend(1,2,0)]

linear_h2o_autodiff = [intcos.STRE(2,1),intcos.STRE(2,0),intcos.BEND(1,2,0,bendType='LINEAR')]
linear_h2o_optking  = [optking.Stre(2,1),optking.Stre(2,0),optking.Bend(1,2,0,bendType='LINEAR')]

ammonia_autodiff = [intcos.STRE(0,1),intcos.STRE(0,2),intcos.STRE(0,3),intcos.STRE(0,4),intcos.BEND(1,0,2),intcos.BEND(1,0,3),intcos.BEND(1,0,4),intcos.BEND(2,0,3), intcos.BEND(2,0,4),intcos.BEND(3,0,4)]
ammonia_optking = [optking.Stre(0,1),optking.Stre(0,2),optking.Stre(0,3),optking.Stre(0,4),optking.Bend(1,0,2),optking.Bend(1,0,3),optking.Bend(1,0,4),optking.Bend(2,0,3),optking.Bend(2,0,4),optking.Bend(3,0,4)]

h2co_autodiff = [intcos.STRE(0,1),intcos.STRE(0,2),intcos.BEND(2,0,1),intcos.STRE(0,3),intcos.BEND(3,0,1),intcos.TORS(3,0,1,2)]
h2co_optking = [optking.Stre(0,1),optking.Stre(0,2),optking.Bend(2,0,1),optking.Stre(0,3),optking.Bend(3,0,1),optking.Tors(3,0,1,2)]

hooh_autodiff = [intcos.STRE(0,1),intcos.STRE(0,2),intcos.BEND(2,1,0),intcos.STRE(3,2),intcos.BEND(3,2,1),intcos.TORS(3,2,1,0)]
hooh_optking = [optking.Stre(0,1),optking.Stre(0,2),optking.Bend(2,1,0),optking.Stre(3,2),optking.Bend(3,2,1),optking.Tors(3,2,1,0)]

sf4_autodiff = [intcos.TORS(0,1,2,3),intcos.TORS(1,3,4,0),intcos.TORS(0,2,1,4)]
sf4_optking = [optking.Tors(0,1,2,3),optking.Tors(1,3,4,0),optking.Tors(0,2,1,4)]

allene_autodiff = [intcos.STRE(0, 2),intcos.STRE(1, 2),intcos.STRE(2, 3),intcos.STRE(3, 4),intcos.STRE(4, 5),intcos.STRE(4, 6),intcos.BEND(0, 2, 1),intcos.BEND(0, 2, 3),intcos.BEND(1, 2, 3),intcos.BEND(2, 3, 4),intcos.BEND(2, 3, 4),intcos.
BEND(3, 4, 5),intcos.BEND(3, 4, 6),intcos.BEND(5, 4, 6),intcos.TORS(0, 2, 4, 5),intcos.TORS(0, 2, 4, 6),intcos.TORS(1, 2, 4, 5),intcos.TORS(1, 2, 4, 6)]
allene_optking = [optking.Stre(0, 2),optking.Stre(1, 2),optking.Stre(2, 3),optking.Stre(3, 4),optking.Stre(4, 5),optking.Stre(4, 6),optking.Bend(0, 2, 1),optking.Bend(0, 2, 3),optking.Bend(1, 2, 3),optking.Bend(2, 3, 4, bendType="LINEAR"),optking.Bend(2, 3, 4, bendType="LINEAR"),optking.Bend(3, 4, 5),optking.Bend(3, 4, 6),optking.Bend(5, 4, 6),optking.Tors(0, 2, 4, 5),optking.Tors(0, 2, 4, 6),optking.Tors(1, 2, 4, 5),optking.Tors(1, 2, 4, 6)]

big_autodiff = [intcos.STRE(0, 1),intcos.STRE(0, 2),intcos.STRE(0, 8),intcos.STRE(0, 9),intcos.STRE(2, 3),intcos.STRE(2, 7),intcos.STRE(3, 4),intcos.STRE(3, 5),intcos.STRE(3, 6),intcos.BEND(0, 2, 3),intcos.BEND(0, 2, 7),intcos.BEND(1, 0, 2),intcos.BEND(1, 0, 8),intcos.BEND(1, 0, 9),intcos.BEND(2, 0, 8),intcos.BEND(2, 0, 9),intcos.BEND(2, 3, 4),intcos.BEND(2, 3, 5),intcos.BEND(2, 3, 6),intcos.BEND(3, 2, 7),intcos.BEND(4, 3, 5),intcos.BEND(4, 3, 6),intcos.BEND(5, 3, 6),intcos.BEND(8, 0, 9),intcos.TORS(0, 2, 3, 4),intcos.TORS(0, 2, 3, 5),intcos.TORS(0, 2, 3, 6),intcos.TORS(1, 0, 2, 3),intcos.TORS(1, 0, 2, 7),intcos.TORS(3, 2, 0, 8),intcos.TORS(3, 2, 0, 9),intcos.TORS(4, 3, 2, 7),intcos.TORS(5, 3, 2, 7),intcos.TORS(6, 3, 2, 7),intcos.TORS(7, 2, 0, 8),intcos.TORS(7, 2, 0, 9)]
big_optking = [optking.Stre(0, 1),optking.Stre(0, 2),optking.Stre(0, 8),optking.Stre(0, 9),optking.Stre(2, 3),optking.Stre(2, 7),optking.Stre(3, 4),optking.Stre(3, 5),optking.Stre(3, 6),optking.Bend(0, 2, 3),optking.Bend(0, 2, 7),optking.Bend(1, 0, 2),optking.Bend(1, 0, 8),optking.Bend(1, 0, 9),optking.Bend(2, 0, 8),optking.Bend(2, 0, 9),optking.Bend(2, 3, 4),optking.Bend(2, 3, 5),optking.Bend(2, 3, 6),optking.Bend(3, 2, 7),optking.Bend(4, 3, 5),optking.Bend(4, 3, 6),optking.Bend(5, 3, 6),optking.Bend(8, 0, 9),optking.Tors(0, 2, 3, 4),optking.Tors(0, 2, 3, 5),optking.Tors(0, 2, 3, 6),optking.Tors(1, 0, 2, 3),optking.Tors(1, 0, 2, 7),optking.Tors(3, 2, 0, 8),optking.Tors(3, 2, 0, 9),optking.Tors(4, 3, 2, 7),optking.Tors(5, 3, 2, 7),optking.Tors(6, 3, 2, 7),optking.Tors(7, 2, 0, 8),optking.Tors(7, 2, 0, 9)]


print("Testing water...")
test_molecule(h2o.geometry(),h2o_autodiff,  h2o_optking )
print("Testing linear water...")
test_molecule(linear_h2o.geometry(),linear_h2o_autodiff, linear_h2o_optking)
print("Testing ammonia...")
test_molecule(ammonia.geometry(),ammonia_autodiff, ammonia_optking)
print("Testing formaldehyde...")
test_molecule(h2co.geometry(),h2co_autodiff, h2co_optking)
print("Testing bent formaldehyde...")
test_molecule(bent_h2co.geometry(),h2co_autodiff, h2co_optking)
print("Testing hooh...")
test_molecule(hooh.geometry(),hooh_autodiff, hooh_optking)
print("Testing nonsense sf4 ...")
test_molecule(sf4.geometry(),sf4_autodiff, sf4_optking)
print("Testing allene...")
test_molecule(allene.geometry(),allene_autodiff, allene_optking)
print("Testing ch3chch2cl...")
test_molecule(big.geometry(),big_autodiff, big_optking)




