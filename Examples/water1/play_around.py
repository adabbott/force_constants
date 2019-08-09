from peslearn.ml import NeuralNetwork
from peslearn import InputProcessor
import numpy as np
import torch
from compute_energy import pes
import pyforce

np.set_printoptions(threshold=5000, linewidth=200, precision=5, suppress=True)
torch.set_printoptions(threshold=5000, linewidth=200, precision=12)

# Load NN model
nn = NeuralNetwork('model_data/PES.dat', InputProcessor(''), molecule_type='A2B')
params = {'layers': (64, 64), 'morse_transform': {'morse': True, 'morse_alpha': 1.2000000000000002}, 'pip': {'degree_reduction': False, 'pip': True}, 'scale_X': {'activation': 'tanh', 'scale_X': 'mm11'}, 'scale_y': 'std', 'lr': 0.8}
X, y, Xscaler, yscaler =  nn.preprocess(params, nn.raw_X, nn.raw_y)
model = torch.load('model_data/model.pt')

# Construct computation graph for sending raw coordinates through the NN model
# so that derivatives d^nE/d(coord)^n can be found.
def transform(interatomics):
    """ Takes Torch Tensor (requires_grad=True) of interatomic distances, manually transforms geometry to track gradients, computes energy
        Hard-coded based on hyperparameters above. Returns: energy in units the NN model was trained on"""
    inp2 = -interatomics / 1.2
    inp3 = torch.exp(inp2)
    inp4 = torch.stack((inp3[0], inp3[1] + inp3[2], torch.sum(torch.pow(inp3[1:],2))), dim=0) # Careful! Degree reduce?
    inp5 = (inp4 * torch.tensor(Xscaler.scale_, dtype=torch.float64)) + torch.tensor(Xscaler.min_, dtype=torch.float64)
    out1 = model(inp5)
    energy = (out1 * torch.tensor(yscaler.scale_, dtype=torch.float64)) + torch.tensor(yscaler.mean_, dtype=torch.float64)
    return energy

# Compute force constants with interatomic distances
# Define equilbrium geometry with interatomic distances, cartesians, and define internal coordinate objects
m = np.array([1.007825032230, 1.007825032230, 15.994914619570])
cartesians = torch.tensor([[ 0.0000000000,0.0000000000,0.9496765298],
                           [ 0.0000000000,0.8834024755,-0.3485478124],
                           [ 0.0000000000,0.0000000000,0.0000000000]], dtype=torch.float64, requires_grad=True)
interatomics = pyforce.transforms.get_interatomics(3)

# Construct B tensors (interatomic distances, idm), remove from torch computation graphs and convert to numpy arrays
B1_idm, B2_idm, B3_idm = pyforce.compute_btensors(interatomics, cartesians, order=3)
B1_idm, B2_idm, B3_idm = B1_idm.detach().numpy(), B2_idm.detach().numpy(), B3_idm.detach().numpy()

# Compute derivatives of PES at equilbrium geometry
#eq_geom = [1.570282260121,0.949676529800,0.949676529800]
eq_geom = [1.570282260121,0.949676529800,0.969676529800]
tmp = []
for i in eq_geom:
    tmp.append(torch.tensor(i, dtype=torch.float64, requires_grad=True))
geom = torch.stack(tmp)
E = transform(geom)

interatomic_hess, interatomic_cubic = pyforce.differentiate_nn(E, geom, order=3)

hess, cubic = pyforce.transforms.new_differentiate_nn(E, tmp, order=3)

print(torch.allclose(hess,interatomic_hess))
print(torch.allclose(cubic,interatomic_cubic))

#
