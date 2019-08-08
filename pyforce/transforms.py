import torch
import numpy as np
from . import intcos

def get_interatomics(natoms):
    """
    Builds list internal coordinates for unique interatomic distances in row-wise order of the lower triangle.
    """
    indices = np.asarray(np.tril_indices(natoms,-1)).transpose(1,0)
    interatomics = []
    for i in indices:
        idx1, idx2 = i
        interatomics.append(intcos.STRE(idx1, idx2))
    return interatomics

def cart2distances(cart):
    """Transforms cartesian coordinate torch Tensor (requires_grad=True) into interatomic distances"""
    natom = cart.size()[0]
    ndistances = int((natom**2 - natom) / 2)
    distances = torch.zeros((ndistances), requires_grad=True, dtype=torch.float64)
    count = 0
    for i,atom1 in enumerate(cart):
        for j,atom2 in enumerate(cart):
            if j > i:
                distances[count] = torch.norm(cart[i]- cart[j])
                count += 1
    return distances

def cart2internals(cart, internals):
    values = intcos.qValues(internals, cart)   # Generate internal coordinates from cartesians
    return values

def cartcubic2intcubic(cart_cubic, int_hess, B1, B2):
    G = np.dot(B1, B1.T)
    Ginv = np.linalg.inv(G)
    A = np.dot(Ginv, B1)
    tmp1 = np.einsum('ia,jb,kc,abc->ijk', A, A, A, cart_cubic)
    tmp2 = np.einsum('lmn,il,jm,kn->ijk', B2, int_hess, A, A)
    tmp3 = np.einsum('lmn,jl,im,kn->ijk', B2, int_hess, A, A)
    tmp4 = np.einsum('lmn,kl,im,jn->ijk', B2, int_hess, A, A)
    int_cubic = tmp1 - tmp2 - tmp3 - tmp4
    return int_cubic

def intcubic2cartcubic(intcubic, inthess, B1, B2):
    tmp1 = np.einsum('ia,jb,kc,ijk->abc', B1, B1, B1, intcubic)
    tmp2 = np.einsum('iab,jc,ij->abc', B2, B1, inthess)
    tmp3 = np.einsum('ica,jb,ij->abc', B2, B1, inthess)
    tmp4 = np.einsum('ibc,ja,ij->abc', B2, B1, inthess)
    cart_cubic = tmp1 + tmp2 + tmp3 + tmp4
    return cart_cubic

def cartderiv2intderiv(derivative_tensor, B1):
    """
    Converts cartesian derivative tensor (gradient or Hessian) into internal coordinates
    Only valid at stationary points for Hessians and above.

    Parameters
    ----------   
    derivative_tensor : np.ndarray
        Tensor of nth derivative in Cartesian coordinates 
    B1 : np.ndarray
        B-matrix converting Cartesian coordinates to internal coordinates
    """
    G = np.dot(B1, B1.T)
    Ginv = np.linalg.inv(G)
    A = np.dot(Ginv, B1)
    dim = len(derivative_tensor.shape)
    if dim == 1:
        int_tensor = np.einsum('ia,a->i', A, derivative_tensor)
    elif dim == 2:
        int_tensor = np.einsum('ia,jb,ab->ij', A, A, derivative_tensor)
    else:
        raise Exception("Too many dimensions. Add code to function")
    return int_tensor

def intderiv2cartderiv(derivative_tensor, B1):
    """ 
    Converts cartesian derivative tensor (gradient or Hessian) into internal coordinates
    Only valid at stationary points for Hessians and above.

    Parameters
    ----------   
    derivative_tensor : np.ndarray
        Tensor of nth derivative in internal coordinates 
    B1 : np.ndarray
        B-matrix converting internal coordinates to Cartesian coordinates
    """
    dim = len(derivative_tensor.shape)
    if dim == 1:
        cart_tensor = np.einsum('ia,i->a', B1, derivative_tensor)
    elif dim == 2:
        cart_tensor = np.einsum('ia,jb,ij->ab', B1, B1, derivative_tensor)
    else:
        raise Exception("Too many dimensions. Add code to function to compute")
    return cart_tensor

def differentiate_nn(geom, E, order=4):
    """
    Takes a geometry, sends it through the NN with the transform() method.
    Returns derivative tensors of a neural network for a particular geometry.
    If order=4 it will return an unpackable tuple of the gradient, hessian, cubic, and quartic derivatives
    If order=5 it will return the  hessian, cubic, quartic, and quintic derivatives
    If order=6 it will return the hessian, cubic, quartic, quintic, and sextic derivatives.

    Parameters
    ----------
    geometry : list
        A list of geometry parameters
    cartesian : bool
        Are the coordinates cartesian coordinates? Otherwise assumes internal coordinates (interatomic distances, typically for NN models)
    order : int
        Highest order of derivative to compute

    Returns
    -------
    A tuple of derivative tensors up through order'th derivatives
    """
    # Initialize variables, transform geometry, compute energy.
    #tmpgeom = []
    #for i in geometry:
    #    tmpgeom.append(torch.tensor(i, dtype=torch.float64, requires_grad=True))
    #if not cartesian:
    #    geom = torch.stack(tmpgeom)
    ##    E = transform(geom)
    #else:
    #    tmpgeom2 = torch.stack(tmpgeom)#.clone().detach().requires_grad_(True)
    #    geom = cart2distances(tmpgeom2)
    #E = transform(geom)
    # Compute derivatives. Build up higher order tensors one dimension at a time.
    gradient = torch.autograd.grad(E, geom, create_graph=True)[0]
    h1, c1, q1, f1, s1 = [], [], [], [], []
    for d1 in gradient:
        h = torch.autograd.grad(d1, geom, create_graph=True)[0]
        h1.append(h)
        c2, q2, f2, s2 = [], [], [], []
        for d2 in h:
        #for d2 in h[np.tril_indices]: 
            c = torch.autograd.grad(d2, geom, create_graph=True)[0]
            c2.append(c)
            q3, f3, s3 = [], [], []
            for d3 in c:
                q = torch.autograd.grad(d3, geom, create_graph=True)[0]
                q3.append(q)
                if order > 4:
                    f4, s4 = [], []
                    for d4 in q:
                        f = torch.autograd.grad(d4, geom, create_graph=True)[0]
                        f4.append(f)
                        if order > 5:
                            s5 = []
                            for d5 in f:
                                s = torch.autograd.grad(d5, geom, create_graph=True)[0]
                                s5.append(s)
                            s4.append(torch.stack(s5))
                        else:
                            continue
                    f3.append(torch.stack(f4))
                    if order > 5: s3.append(torch.stack(s4))
                else:
                    continue
            q2.append(torch.stack(q3))
            if order > 4: f2.append(torch.stack(f3))
            if order > 5: s2.append(torch.stack(s3))
        c1.append(torch.stack(c2))
        q1.append(torch.stack(q2))
        if order > 4: f1.append(torch.stack(f2))
        if order > 5: s1.append(torch.stack(s2))

    hessian = torch.stack(h1)
    cubic = torch.stack(c1)
    quartic = torch.stack(q1)
    if order == 4:
        return hessian, cubic, quartic
    elif order == 5:
        quintic = torch.stack(f1)
        return hessian, cubic, quartic, quintic
    elif order == 6:
        quintic = torch.stack(f1)
        sextic = torch.stack(s1)
        return hessian, cubic, quartic, quintic, sextic


