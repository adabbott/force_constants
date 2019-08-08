"""
Contains internal coordinate classes STRE, BEND, and TORS.
Each performs coordinate conversions with Torch functions
in order to track derivatives and enable B tensor computations.
Modified from OPTKING.
"""

import torch
import math
from . import v3d 

def qValues(intcos, geom):
    """Calculates internal coordinates from cartesian geometry
    Parameters
    ---------
    intcos : list
        (nat) list of stretches, bends, etc...
        E.g. [intcos.STRE(0,2), intcos.STRE(1,2), intcos.BEND(0,2,1)]
    geom : torch.tensor (with attribute requires_grad=True if tracking derivatives)
        Shape (nat, 3) or (3*nat,) cartesian geometry in Angstroms

    Returns
    -------
    q : torch.tensor 
        internal coordinate values computed from cartesian coordinates
    """
    if len(list(geom.size())) == 1:
        tmpgeom = torch.reshape(geom, (-1,3))
        q = torch.stack([i.q(tmpgeom) for i in intcos])
    else:
        q = torch.stack([i.q(geom) for i in intcos])
    return q

class STRE(object):
    def __init__(self, a, b):
        if a < b: 
            atoms = (a,b) 
        else: 
            atoms = (b,a)
        self.A = atoms[0]
        self.B = atoms[1]

    def q(self, geom):
        return v3d.dist(geom[self.A], geom[self.B])

class BEND(object):
    def __init__(self, a, b, c, bendType="REGULAR"):
        if a < c:
            atoms = (a, b, c)
        else:
            atoms = (c, b, a)
        self.A = atoms[0]
        self.B = atoms[1]
        self.C = atoms[2]
        self._bendType = bendType 
        self._axes_fixed = False

    def compute_axes(self, geom):
        u = v3d.eAB(geom[self.B], geom[self.A])  # B->A
        v = v3d.eAB(geom[self.B], geom[self.C])  # B->C

        if v3d.are_parallel_or_antiparallel(u,v):
            self._bendType = "LINEAR"

        if self._bendType == "REGULAR":                   # not a linear-bend type
            self._w = v3d.normalize(v3d.cross(u,v)) # cross product and normalize
            self._x = v3d.normalize(u + v)             # angle bisector
            return

        #tv1 = torch.tensor([1,0,0], dtype=torch.float64, requires_grad=True) 
        tv1 = torch.tensor([1,0,0], dtype=torch.float64) 
        #tmp_tv2 = torch.tensor([0,1,1], dtype=torch.float64, requires_grad=True)
        tmp_tv2 = torch.tensor([0,1,1], dtype=torch.float64)
        tv2 = v3d.normalize(tmp_tv2)

        u_tv1 = v3d.are_parallel_or_antiparallel(u, tv1)
        v_tv1 = v3d.are_parallel_or_antiparallel(v, tv1)
        u_tv2 = v3d.are_parallel_or_antiparallel(u, tv2)
        v_tv2 = v3d.are_parallel_or_antiparallel(v, tv2)

        # handle both types of linear bends
        if not v3d.are_parallel_or_antiparallel(u, v):
            self._w = v3d.normalize(v3d.cross(u, v))  # orthogonal vector
            self._x = v3d.normalize(u + v)
        # u || v but not || to tv1.
        elif not u_tv1 and not v_tv1:
            self._w = v3d.normalize(v3d.cross(u, tv1))
            self._x = v3d.normalize(v3d.cross(self._w, u))
        # u || v but not || to tv2.
        elif not u_tv2 and not v_tv2:
            self._w = v3d.normalize(v3d.cross(u,tv2))
            self._x = v3d.normalize(v3d.cross(self._w, u))

        if self._bendType == "COMPLEMENT":
            w2 = torch.copy(self._w)
            self._w = -1.0 * self._x  # -w_normal -> x_complement
            self._x = w2
        return
        
    def q(self, geom):
        if not self._axes_fixed:
            self.compute_axes(geom)
        u = v3d.eAB(geom[self.B], geom[self.A])  # B->A
        v = v3d.eAB(geom[self.B], geom[self.C])  # B->C
        #origin = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        origin = torch.zeros(3, dtype=torch.float64)
        phi1 = v3d.angle(u, origin, self._x) 
        phi2 = v3d.angle(self._x, origin, v)
        phi = phi1 + phi2
        return phi

    def new_compute_axes(self, coord1,coord2,coord3):
        u = v3d.eAB(coord2, coord1)  # B->A
        v = v3d.eAB(coord2, coord3)  # B->C

        if v3d.are_parallel_or_antiparallel(u,v):
            self._bendType = "LINEAR"

        if self._bendType == "REGULAR":                   # not a linear-bend type
            self._w = v3d.normalize(v3d.cross(u,v)) # cross product and normalize
            self._x = v3d.normalize(u + v)             # angle bisector
            return

        tv1 = torch.tensor([1,0,0], dtype=torch.float64) 
        tmp_tv2 = torch.tensor([0,1,1], dtype=torch.float64)
        tv2 = v3d.normalize(tmp_tv2)

        u_tv1 = v3d.are_parallel_or_antiparallel(u, tv1)
        v_tv1 = v3d.are_parallel_or_antiparallel(v, tv1)
        u_tv2 = v3d.are_parallel_or_antiparallel(u, tv2)
        v_tv2 = v3d.are_parallel_or_antiparallel(v, tv2)

        # handle both types of linear bends
        if not v3d.are_parallel_or_antiparallel(u, v):
            self._w = v3d.normalize(v3d.cross(u, v))  # orthogonal vector
            self._x = v3d.normalize(u + v)
        # u || v but not || to tv1.
        elif not u_tv1 and not v_tv1:
            self._w = v3d.normalize(v3d.cross(u, tv1))
            self._x = v3d.normalize(v3d.cross(self._w, u))
        # u || v but not || to tv2.
        elif not u_tv2 and not v_tv2:
            self._w = v3d.normalize(v3d.cross(u,tv2))
            self._x = v3d.normalize(v3d.cross(self._w, u))

        if self._bendType == "COMPLEMENT":
            w2 = torch.copy(self._w)
            self._w = -1.0 * self._x  # -w_normal -> x_complement
            self._x = w2
        return
          
class TORS(object):
    def __init__(self, a, b, c, d):

        if a < d: self.atoms = (a, b, c, d)
        else: self.atoms = (d, c, b, a)
        self.A = self.atoms[0]
        self.B = self.atoms[1]
        self.C = self.atoms[2]
        self.D = self.atoms[3]
        self._near180 = 0

    def q(self, geom):
        try:
            tau = v3d.tors(geom[self.A], geom[self.B], geom[self.C], geom[self.D])
        except: 
            raise Exception("Tors.q: unable to compute torsion value")
        return tau

