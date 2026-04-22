from itertools import combinations, permutations
from sympy import Function, I, diff, symbols
from sympy import symbols
from sympy.physics.quantum import Commutator
from counter_diabatic.utils_ import (
    single_body_op,
    s1,
    s2,
    s3,
    two_body_op,
)

# ── helpers ────────


def build_cd(n_qubits):
    X = [single_body_op(s1, i, n_qubits) for i in range(n_qubits)]
    Y = [single_body_op(s2, i, n_qubits) for i in range(n_qubits)]
    Z = [single_body_op(s3, i, n_qubits) for i in range(n_qubits)]

    Omega = [symbols(f"Omega_{i+1}", real=True, positive=True) for i in range(n_qubits)]
    Mu = [symbols(f"mu_{i+1}", real=True) for i in range(n_qubits)]
    Nu = [symbols(f"nu_{i+1}", real=True) for i in range(n_qubits)]
    U = {
        (i, j): symbols(f"U{i+1}{j+1}", real=True, positive=True)
        for i, j in combinations(range(n_qubits), 2)
    }
    lam = symbols("lambda", real=True)

    a = {i: symbols(f"a_{i+1}", real=True) for i in range(n_qubits)}
    b = {i: symbols(f"b_{i+1}", real=True) for i in range(n_qubits)}
    c = {i: symbols(f"c_{i+1}", real=True) for i in range(n_qubits)}
    d_xx = {
        (i, j): symbols(f"delta_xx_{i+1}{j+1}", real=True)
        for i, j in combinations(range(n_qubits), 2)
    }
    d_yy = {
        (i, j): symbols(f"delta_yy_{i+1}{j+1}", real=True)
        for i, j in combinations(range(n_qubits), 2)
    }
    d_zz = {
        (i, j): symbols(f"delta_zz_{i+1}{j+1}", real=True)
        for i, j in combinations(range(n_qubits), 2)
    }
    d_xy = {
        (i, j): symbols(f"delta_xy_{i+1}{j+1}", real=True)
        for i, j in permutations(range(n_qubits), 2)
    }
    d_xz = {
        (i, j): symbols(f"delta_xz_{i+1}{j+1}", real=True)
        for i, j in permutations(range(n_qubits), 2)
    }
    d_yz = {
        (i, j): symbols(f"delta_yz_{i+1}{j+1}", real=True)
        for i, j in permutations(range(n_qubits), 2)
    }

    Omega_l = [Function(f"Omega_{i+1}")(lam) for i in range(n_qubits)]
    mu_l = [Function(f"mu_{i+1}")(lam) for i in range(n_qubits)]
    nu_l = [Function(f"nu_{i+1}")(lam) for i in range(n_qubits)]

    A = sum(a[i] * X[i] + b[i] * Y[i] + c[i] * Z[i] for i in range(n_qubits))
    for i, j in combinations(range(n_qubits), 2):
        A += d_xx[(i, j)] * two_body_op(s1, s1, i, j, n_qubits)
        A += d_yy[(i, j)] * two_body_op(s2, s2, i, j, n_qubits)
        A += d_zz[(i, j)] * two_body_op(s3, s3, i, j, n_qubits)
    for i, j in permutations(range(n_qubits), 2):
        A += d_xy[(i, j)] * two_body_op(s1, s2, i, j, n_qubits)
        A += d_xz[(i, j)] * two_body_op(s1, s3, i, j, n_qubits)
        A += d_yz[(i, j)] * two_body_op(s2, s3, i, j, n_qubits)

    H = sum(
        Omega[i] * X[i] + Mu[i] * Y[i] + Nu[i] * Z[i] for i in range(n_qubits)
    ) + sum(
        U[(i, j)] * two_body_op(s3, s3, i, j, n_qubits)
        for i, j in combinations(range(n_qubits), 2)
    )
    H_lam = sum(
        Omega_l[i] * X[i] + mu_l[i] * Y[i] + nu_l[i] * Z[i] for i in range(n_qubits)
    ) + sum(
        U[(i, j)] * two_body_op(s3, s3, i, j, n_qubits)
        for i, j in combinations(range(n_qubits), 2)
    )
    dH_dlam = diff(H_lam, lam)
    icomm = (
        (I * Commutator(A, H).expand(commutator=True)).expand(commutator=True).doit()
    )
    CD = dH_dlam + icomm

    return (
        CD,
        X,
        Y,
        Z,
        Omega,
        Mu,
        Nu,
        U,
        a,
        b,
        c,
        d_xx,
        d_yy,
        d_zz,
        d_xy,
        d_xz,
        d_yz,
        Omega_l,
        mu_l,
        nu_l,
        lam,
    )
