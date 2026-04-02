from itertools import combinations, permutations

import torch


def A_direct_mat(
    n_atoms: int,
    Omega_t: torch.Tensor,
    Mu_t: torch.Tensor,
    Nu_t: torch.Tensor,
    U_t: torch.Tensor,
):
    """Directly build the A matrix from CD linear problem: Ax=b  from
    the sequence, upto 2 body terms."""
    n_single = 3 * n_atoms
    n_sym = 3 * len(list(combinations(range(n_atoms), 2)))  # 3 * C(n,2)
    n_asym = 3 * len(list(permutations(range(n_atoms), 2)))  # 3 * n*(n-1)
    n_total = n_single + n_sym + n_asym

    A_matrix = torch.zeros(n_total, n_total, dtype=torch.float64)
    # Omega goes colum 1 + 3*i, row 2+3*i
    # Mu goes colum 0 + 3*i, row 1+3*i
    # Nu goes colum 0 + 3*i, row 1+3*i
    #

    for i in range(n_atoms):
        A_matrix[1 + 3 * i, 2 + 3 * i] = -Omega_t[i].real.detach()
        A_matrix[0 + 3 * i, 2 + 3 * i] = Mu_t[i].real.detach()
        A_matrix[0 + 3 * i, 1 + 3 * i] = -Nu_t[i].real.detach()

    # single-body terms, always go in the first n_single columns and n_single rows

    # interaction terms in the 2 body terms
    sing_plus_sym = n_single + n_sym

    z_0 = 2
    for i, j in combinations(range(n_atoms), 2):

        A_matrix[i * 3, sing_plus_sym + z_0] = -U_t[i, j].real.detach()
        A_matrix[i * 3 + 1, sing_plus_sym + z_0 - 1] = U_t[i, j].real.detach()
        A_matrix[j * 3, sing_plus_sym + z_0 + 3] = -U_t[i, j].real.detach()
        A_matrix[j * 3 + 1, sing_plus_sym + z_0 + 2] = U_t[i, j].real.detach()
        z_0 += 6

    z_0 = 0
    l_0 = 0
    for i, j in combinations(range(n_atoms), 2):
        A_matrix[n_single + l_0, sing_plus_sym + z_0] = -Nu_t[j].real.detach()
        A_matrix[n_single + l_0, sing_plus_sym + z_0 + 3] = -Nu_t[i].real.detach()

        A_matrix[n_single + l_0, sing_plus_sym + z_0 + 1] = Mu_t[j].real.detach()
        A_matrix[n_single + l_0, sing_plus_sym + z_0 + 1 + 3] = Mu_t[i].real.detach()

        A_matrix[n_single + l_0 + 1, sing_plus_sym + z_0] = Nu_t[i].real.detach()
        A_matrix[n_single + l_0 + 1, sing_plus_sym + z_0 + 3] = Nu_t[j].real.detach()

        A_matrix[n_single + l_0 + 1, sing_plus_sym + z_0 + 2] = -Omega_t[
            j
        ].real.detach()
        A_matrix[n_single + l_0 + 1, sing_plus_sym + z_0 + 2 + 3] = -Omega_t[
            i
        ].real.detach()

        A_matrix[n_single + l_0 + 2, sing_plus_sym + z_0 + 1] = -Mu_t[i].real.detach()
        A_matrix[n_single + l_0 + 2, sing_plus_sym + z_0 + 1 + 3] = -Mu_t[
            j
        ].real.detach()

        A_matrix[n_single + l_0 + 2, sing_plus_sym + z_0 + 2] = Omega_t[i].real.detach()
        A_matrix[n_single + l_0 + 2, sing_plus_sym + z_0 + 2 + 3] = Omega_t[
            j
        ].real.detach()

        z_0 += 6
        l_0 += 3

    z_0 = 1
    l_0 = 0
    for i, j in combinations(range(n_atoms), 2):
        A_matrix[sing_plus_sym + l_0, sing_plus_sym + z_0] = -Omega_t[j].real.detach()
        A_matrix[sing_plus_sym + l_0 + 3, sing_plus_sym + z_0 + 3] = -Omega_t[
            i
        ].real.detach()

        A_matrix[sing_plus_sym + l_0 + 1, sing_plus_sym + z_0 + 1] = -Nu_t[
            i
        ].real.detach()
        A_matrix[sing_plus_sym + l_0 + 1 + 3, sing_plus_sym + z_0 + 1 + 3] = -Nu_t[
            j
        ].real.detach()

        A_matrix[sing_plus_sym + l_0 + 2, sing_plus_sym + z_0 + 2] = -Mu_t[
            j
        ].real.detach()
        A_matrix[sing_plus_sym + l_0, sing_plus_sym + z_0 + 4] = Mu_t[i].real.detach()

        l_0 += 6
        z_0 += 6

    A_matrix = A_matrix - A_matrix.T

    return A_matrix


# def b_direct_vec(
#     n_atoms: int,
#     dOmega_t: torch.Tensor,
#     dMu_t: torch.Tensor,
#     dNu_t: torch.Tensor,
# ):
#     """Directly build the b vector from the sequence.
#     b vector is the derivatives of local fields X1, Y1, Z1, X2, Y2, Z2, X3, Y3,
#     Z3, ..
#  """
#     n_single = 3 * n_atoms
#     n_sym = 3 * len(list(combinations(range(n_atoms), 2)))  # 3 * C(n,2)
#     n_asym = 3 * len(list(permutations(range(n_atoms), 2)))  # 3 * n*(n-1)
#     n_total = n_single + n_sym + n_asym

#     b_vector = torch.zeros(n_total, dtype=torch.float64)
#     for i in range(n_atoms):
#         b_vector[0 + 3 * i] = -dOmega_t[i] / 2
#         b_vector[1 + 3 * i] = -dMu_t[i] / 2
#         b_vector[2 + 3 * i] = -dNu_t[i] / 2

#     return b_vector


def b_direct_vec(n_atoms, dOmega_t, dMu_t, dNu_t):
    n_sym = 3 * len(list(combinations(range(n_atoms), 2)))
    n_asym = 3 * len(list(permutations(range(n_atoms), 2)))

    singles = torch.stack(
        [
            v
            for i in range(n_atoms)
            for v in (-dOmega_t[i] / 2, -dMu_t[i] / 2, -dNu_t[i] / 2)
        ]
    )
    rest = torch.zeros(n_sym + n_asym, dtype=torch.float64)
    return torch.cat([singles, rest])


def solve_cd_torch(M, b):
    """
    Minimum-norm least-squares solution via pseudo-inverse (differentiable).
    """
    return torch.linalg.pinv(M) @ b
