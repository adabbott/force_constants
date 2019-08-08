import torch
from . import intcos
from . import v3d
bohr2ang = 0.529177249

def compute_btensors(internals, geom, order=1):
    """
    Computes and returns all B tensors up to order'th order. Max order is 3rd order
    
    Parameters
    ---------
    internals : list
        A list of internal coordinate objects STRE, BEND, TORS
    geom : torch.tensor
        A Torch tensor of dimension (natoms, 3) containing Cartesian coordinates in Angstroms.

    order : int
        Compute up to this order B tensors and return them all. 

    Returns
    -------
    An unpackable tuple of B tensors up through order'th order. Length is order.
    E.g.  B1, B2, B3 = compute_btensors(intcos, geom, order=3) will give 1st, 2nd, 3rd order B tensors.
    E.g.  B1 = compute_btensors(intcos, geom, order=1) will give just 1st order B tensor.
    """
    if order > 3:
        raise Exception("Only up to 3rd order is allowed. Too expensive after that!")
    #TODO remove trivial derivative computations?
    cart_vec = geom.flatten()
    # Generate internal coordinates from cartesians. This constructs a computation graph which can be differentiated.
    intcoords = intcos.qValues(internals, cart_vec)   
    nint = intcoords.shape[0]
    ncart = 3 * geom.shape[0]
    count = 0
    shape = [nint, ncart]
    count2 = 0

    g1, h1, c1 = [], [], []
    for d0 in intcoords:
        g = torch.autograd.grad(d0, cart_vec, create_graph=True)[0]
        g1.append(g)
        if order > 1:
            h2, c2 = [], []
            for d1 in g:
                h = torch.autograd.grad(d1, cart_vec, create_graph=True)[0]
                h2.append(h)
                if order > 2:
                    c3 = []
                    for d2 in h:
                        c = torch.autograd.grad(d2, cart_vec, create_graph=True)[0]
                        c3.append(c)
                    c2.append(torch.stack(c3))
                else:
                    continue
            h1.append(torch.stack(h2))
            if order > 2: c1.append(torch.stack(c2))
        else:
            continue
    B1 = torch.stack(g1)
    if order > 1:
        B2 = torch.stack(h1)
        if order > 2:
            B3 = torch.stack(c1)
            return B1, B2, B3
        else:
            return B1, B2
    else:
        return B1

