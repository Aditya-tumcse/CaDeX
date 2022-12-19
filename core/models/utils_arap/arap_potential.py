# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional
from core.models.utils_arap.base_tools import *
import time

torch.backends.cuda.preferred_linalg_library("cusolver") 
def arap_exact(vert_diff_cdc, vert_diff_query, neigh, n_vert):
    S_neigh = torch.bmm(vert_diff_cdc.unsqueeze(2),vert_diff_query.unsqueeze(1))
    
    S = my_zeros([n_vert, 3, 3])

    S = torch.index_add(S, 0, neigh[:, 0], S_neigh)
    S = torch.index_add(S, 0, neigh[:, 1], S_neigh)
    
    # Kabsch algorithm
    U, _, V = torch.svd(S, compute_uv=True)
    
    R = torch.bmm(U, V.transpose(1, 2))
    
    Sigma = my_ones((R.shape[0], 1, 3))
    Sigma[:, :, 2] = torch.det(R).unsqueeze(1)

    R = torch.bmm(U * Sigma, V.transpose(1, 2))
    
    return R


def arap_energy_exact(vert_cdc, vert_query, neigh, lambda_reg_len=1e-6):
    n_vert = vert_cdc.shape[0]
    
    vert_diff_cdc = torch.sub(vert_cdc[neigh[:,0]],vert_cdc[neigh[:,1]])
    vert_diff_query = torch.sub(vert_query[neigh[:,0]],vert_query[neigh[:,1]])

    # Beginning of exact minimization scheme
    # Assuming deformed coordinates (cdc coords) are correct, find value of rotation matrix that ARAP
    R_t = arap_exact(vert_diff_cdc, vert_diff_query, neigh, n_vert)
    
    R_neigh_t = 0.5 * (
        torch.index_select(R_t, 0, neigh[:, 0])
        + torch.index_select(R_t, 0, neigh[:, 1])
    )

    # Assuming R_neigh_t is correct, obtain the deformed coordinates such that they are ARAP deformed
    vert_diff_query_rot = torch.bmm(R_neigh_t, vert_diff_query.unsqueeze(2)).squeeze() # obtain the new coordinates after deforming the shaper as rigid as possible 
    acc_t_neigh = vert_diff_cdc - vert_diff_query_rot # Minimize the difference between deformed coords and original coords

    E_arap = acc_t_neigh.norm() ** 2 + lambda_reg_len * (vert_cdc - vert_query).norm() ** 2
    E_arap = 0.01 * E_arap
  
    return E_arap


if __name__ == "__main__":
    print("main of arap_potential.py")
