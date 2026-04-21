import torch


def Hamiltonian_to_pulse(Omega_H, Mu_H, Nu_H, a_corr, b_corr, c_corr):
    """from Hamiltonian coefficents to pulse parameters, using the
    counter-diabatic corrections.
    """
    X_total = Omega_H + a_corr
    Y_total = Mu_H + b_corr

    delta_cd = -(Nu_H + c_corr)  #
    omega_cd = torch.sqrt(X_total**2 + Y_total**2)

    phi_cd = torch.atan2(Y_total, X_total)

    return omega_cd, delta_cd, phi_cd


def pulse_to_Hamiltonian(
    omegas: torch.Tensor, deltas: torch.Tensor, phis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # """Assemble full pulse [start, interior..., end] — BCs are fixed."""
    # create the hamiltonian at each time step using the sequence data
    omega_H = omegas.real * torch.cos(phis)
    mu_H = omegas.real * torch.sin(phis)
    nu_H = -deltas.real

    # Omega_H = (omega[:, None] / 2) * torch.cos(phi[:, None]).expand(-1, n_atoms)
    # Mu_H    = (omega[:, None] / 2) * torch.sin(phi[:, None]).expand(-1, n_atoms)
    # Nu_H    = (delta[:, None] / 2).expand(-1, n_atoms)

    # omega = torch.cat([bc_omega[:1], omega_free, bc_omega[1:]])  # shape (N+1,)
    # delta = torch.cat([bc_delta[:1], delta_free, bc_delta[1:]])
    # phi = torch.cat([bc_phi[:1], phi_free, bc_phi[1:]])

    return omega_H, mu_H, nu_H


def compute_derivatives_analytical(
    omegas: torch.Tensor, deltas: torch.Tensor, phis: torch.Tensor, dt: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # finite diff the raw pulse params with 2nd-order stencils
    def diff2(x):
        d0 = (-3 * x[0:1] + 4 * x[1:2] - x[2:3]) / (2 * dt)
        di = (x[2:] - x[:-2]) / (2 * dt)
        dn = (3 * x[-1:] - 4 * x[-2:-1] + x[-3:-2]) / (2 * dt)
        return torch.cat([d0, di, dn], dim=0)

    dOmega = diff2(omegas)
    dDelta = diff2(deltas)
    dPhi = diff2(phis)
    # exact, check the 0.5 in front of omega
    dOmega_H = dOmega * torch.cos(phis) - omegas * torch.sin(phis) * dPhi
    dMu_H = dOmega * torch.sin(phis) + omegas * torch.cos(phis) * dPhi
    dNu_H = -dDelta
    return dOmega_H, dMu_H, dNu_H
