import torch
import numpy as np
import torch.nn as nn

import time

#https://github.com/scipy/scipy/blob/main/scipy/spatial/transform/_rotation.pyx
def rmsdalign(a, b, weights=None): # alignes B to A  # [*, N, 3]
    B = a.shape[:-2]
    N = a.shape[-2]
    if weights == None:
        weights = a.new_ones(*B, N)
    weights = weights.unsqueeze(-1)
    a_mean = (a * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    a = a - a_mean
    b_mean = (b * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    b = b - b_mean
    B = torch.einsum('...ji,...jk->...ik', weights * a, b)
    u, s, vh = torch.linalg.svd(B)

    # Correct improper rotation if necessary (as in Kabsch algorithm)
    '''
    if torch.linalg.det(u @ vh) < 0:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]
    '''
    sgn = torch.sign(torch.linalg.det(u @ vh))
    s[...,-1] *= sgn
    u[...,:,-1] *= sgn.unsqueeze(-1)
    C = u @ vh # c rotates B to A
    return b @ C.mT + a_mean
    
def kabsch_rmsd(a, b, weights=None):
    B = a.shape[:-2]
    N = a.shape[-2]
    if weights == None:
        weights = a.new_ones(*B, N)
    b_aligned = rmsdalign(a, b, weights)
    out = torch.square(b_aligned - a).sum(-1)
    out = (out * weights).sum(-1) / weights.sum(-1)
    return torch.sqrt(out)

class SAXS_to_Contact_Map(nn.Module):
    '''
    This module takes a SAXS curve and returns a contact map.
    
    The SAXS curve is a 1D tensor of shape (batch_size, N) where N is the number of points in the SAXS curve.
    Currently, the SAXS curve has a size of 512 points.
    The contact map is a 3D tensor of shape (batch_size, N, N) where N is the number of points in the output.
    Currently, the N is set to 256 which is the crop size of the AlphaFold model.

    Then we use a attention model to convert the 1D tensor to a 3D tensor.
    The output matrix should be seminegative definite.
    '''
    def __init__(self, input_shape, hidden_features, output_dim):
        super().__init__()
        self.channels=hidden_features
        self.layer_norm = nn.LayerNorm(input_shape)
        self.q = nn.Linear(in_features=input_shape, out_features=hidden_features * output_dim)
        self.k = nn.Linear(in_features=input_shape, out_features=hidden_features * output_dim)
        self.v = nn.Linear(in_features=input_shape, out_features=hidden_features * output_dim)
        self.out_layer = nn.Linear(hidden_features, output_dim)
        self.relu_layer = nn.ReLU()
    def forward(self, x):
        h_=self.layer_norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        q = q.view(h_.shape[0],-1,self.channels)
        k = k.view(h_.shape[0],-1,self.channels)
        v = v.view(h_.shape[0],-1,self.channels)
        w_ = torch.bmm(q,k.permute(0,2,1))
        w_ = w_ * (self.channels**(-0.5))
        w_ = torch.nn.functional.softmax(w_,dim=2)
        h_ = torch.bmm(w_,v) 
        h_ = self.out_layer(h_)
        h_ = self.relu_layer(h_)
        return -1 * h_

class HarmonicPrior(nn.Module):
    '''
    This module is the Harmonic Prior module.
    It use the SAXS_to_Contact_Map module to convert the SAXS curve to a contact map.
    And then change the diagonal of the contact map to make the sum of each row and column to be 0.
    This will make the map become semipositve definite by converting to the form of (x-y)^2.
    Then we use the eigenvalue decomposition to get the eigenvalues and eigenvectors.
    And sample 256 points as the noise coordinates for the beta-carbon atoms.
    For sequence length shorter than 256, we will get the first N coordinates in the actual model.
    '''
    def __init__(self, input_shape, hidden_features, output_dim):
        super().__init__()
        self.a =3/(3.8**2)
        self.input_shape=input_shape
        self.channels=hidden_features
        self.output_dim=output_dim
        self.pre_contact_map=SAXS_to_Contact_Map(input_shape,hidden_features, output_dim)
        
    def energy_reshape(self, contact):
        diag_mask = torch.ones_like(contact[0])
        diag_mask = diag_mask.fill_diagonal_(0)
        masked_contact = contact*diag_mask
        column_sums = -1 * torch.sum(masked_contact,dim=2)
        result = torch.diagonal_scatter(masked_contact, column_sums, dim1=1, dim2=2)
        return result
        
    def forward(self, x):
        start_time=time.time()
        pre_contact_map=self.pre_contact_map(x)
        # A*AT whether all positive value 
        contact_map_herm=pre_contact_map + torch.transpose(pre_contact_map,1,2)
        contact_map_energy = self.energy_reshape(contact_map_herm)
        step1_time=time.time()
        lambda_value, nu_vector = torch.linalg.eigh(contact_map_energy)
        step2_time = time.time()
        batch_dims=x.size(0)
        self.lambda_value = torch.clamp(lambda_value, min=0.0001)
        lambda_value_inverse = torch.sqrt(1/self.lambda_value)
        step3_time = time.time()
        rand=torch.randn(batch_dims, self.output_dim, 3, device=x.device)
        step4_time = time.time()
        dot_product = torch.einsum('ij,ijk->ijk', lambda_value_inverse ,rand )
        return_value=torch.bmm(nu_vector, dot_product)
        step5_time = time.time()
        #print('step 1:', step1_time-start_time)
        #print('step 2:', step2_time-step1_time)
        #print('step 3:', step3_time-step2_time)
        #print('step 4:', step4_time-step3_time)
        #print('step 5:', step5_time-step4_time)
        #print(return_value.shape)
        #print(torch.sum(contact_map_energy,dim=(2,1)))
        return return_value, contact_map_energy
    
class PriorLoss(nn.Module):
    '''
    This module is the loss function for the Harmonic Prior module.
    It make the superdiagonal and subdiagonal of the contact map to be a.
    This will constraint the distance of the nearest neighbors atoms.
    '''
    def __init__(self, N=256, a =3/(3.8**2)):
        super().__init__()
        self.a = a
        self.N = N
        self.background = self.fixed_background()
        self.mask_matrix = self.mask()
        self.loss_fn=nn.MSELoss(reduction='sum')

    def fixed_background(self):
        N = self.N
        J = torch.zeros(N, N)
        for i, j in zip(np.arange(N-1), np.arange(1, N)):
            #J[i,i] += self.a
            #J[j,j] += self.a
            J[i,j] = J[j,i] = - self.a
        return J
    # I should remove the diag_mask
    def mask(self):
        diag_mask = torch.eye(self.N, dtype=torch.float32)
        superdiagonal_mask = torch.roll(diag_mask, shifts=1, dims=1)
        superdiagonal_mask[:, 0] = 0
        subdiagonal_mask = torch.roll(diag_mask, shifts=-1, dims=1)
        subdiagonal_mask[:, -1] = 0
        return superdiagonal_mask+subdiagonal_mask
    
    def forward(self, x):
        mask_matrix = self.mask_matrix.to(x.device)  # Ensure mask is on the same device as x
        background = self.background.unsqueeze(0).repeat(x.size(0), 1, 1).to(x.device)  # Ensure background is on the same device as x
        masked_x = x * mask_matrix.float()  # Apply mask
        return self.loss_fn(masked_x, background) # Compute and return the loss

class old_HarmonicPrior:
    def __init__(self, N = 256, a =3/(3.8**2)):
        J = torch.zeros(N, N)
        for i, j in zip(np.arange(N-1), np.arange(1, N)):
            J[i,i] += a
            J[j,j] += a
            J[i,j] = J[j,i] = -a
        D, P = torch.linalg.eigh(J)
        D_inv = 1/D
        D_inv[0] = 0
        self.P, self.D_inv = P, D_inv
        self.N = N

    def to(self, device):
        self.P = self.P.to(device)
        self.D_inv = self.D_inv.to(device)
        
    def sample(self, batch_dims=()):
        return self.P @ (torch.sqrt(self.D_inv)[:,None] * torch.randn(*batch_dims, self.N, 3, device=self.P.device))