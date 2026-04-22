from sympy import I, diff, simplify, symbols
from counter_diabatic.utils_ import tp_compose, tp_trace
from sympy.physics.quantum import Commutator
from counter_diabatic.utils_ import single_body_op, two_body_op, s1, s2, s3, nop
from utils_test import build_cd


def test_tp_compose_scalar_times_add_of_tp():
    """scalar*(tp1+tp2) — TP buried inside Mul(scalar, Add(tp1,tp2)).

    Arises from non-commuting Paulis in nop-containing commutators,
    e.g. [X, nop] produces terms with sigma1*(1-sigma3).
    expand() distributes the scalar into the Add so each summand is scalar*TP.
    """
    n = 2
    H_nop = single_body_op(s1, 0, n) + two_body_op(nop, nop, 0, 1, n)
    A_simple = symbols("a", real=True) * single_body_op(s1, 0, n)
    icomm = (
        (I * Commutator(A_simple, H_nop).expand(commutator=True))
        .expand(commutator=True)
        .doit()
    )
    Y = [single_body_op(s2, i, n) for i in range(n)]
    result = tp_trace(tp_compose(Y[0], icomm), n)
    assert result is not None


def test_tp_compose_cancelling_nop_term():
    """[sigma3, nop] = 0 — both TPs expand to sigma3-sigma3^2 and cancel.

    expand() returns S.Zero; tp_compose must return 0*operator = 0.
    """
    n = 2
    Z0 = single_body_op(s3, 0, n)
    NOP0 = single_body_op(nop, 0, n)
    comm = (
        (I * Commutator(Z0, NOP0).expand(commutator=True))
        .expand(commutator=True)
        .doit()
    )
    Y = [single_body_op(s2, i, n) for i in range(n)]
    result = tp_trace(tp_compose(Y[0], comm), n)
    assert result == 0


def test_cd_assertions_n2():
    """Three known projections of CD onto Pauli basis for n=2 (no nop in H)."""
    n = 2
    (
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
    ) = build_cd(n)

    norm = 2 ** (n + 1)

    expected = (
        -Omega[0] * d_yz[(1, 0)]
        - Omega[1] * d_yz[(0, 1)]
        + d_xy[(0, 1)] * Nu[0]
        + d_xy[(1, 0)] * Nu[1]
    )
    assert simplify(tp_trace(tp_compose(Y[0] * Y[1], CD), n) / norm - expected) == 0

    expected2 = (
        -b[0] * Nu[0]
        + c[0] * Mu[0]
        - d_yz[(0, 1)] * U[(0, 1)]
        + diff(Omega_l[0], lam) / 2
    )
    assert (tp_trace(tp_compose(X[0], CD), n) / norm).equals(expected2)

    expected3 = (
        a[1] * Nu[1]
        - c[1] * Omega[1]
        + d_xz[(1, 0)] * U[(0, 1)]
        + diff(mu_l[1], lam) / 2
    )
    assert (tp_trace(tp_compose(Y[1], CD), n) / norm).equals(expected3)


def test_cd_assertions_n3():
    """Three known projections of CD onto Pauli basis for n=3 (no nop in H)."""
    n = 3
    (
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
    ) = build_cd(n)

    norm = 2 ** (n + 1)

    expected = (
        -Omega[0] * d_yz[(1, 0)]
        - Omega[1] * d_yz[(0, 1)]
        + d_xy[(0, 1)] * Nu[0]
        + d_xy[(1, 0)] * Nu[1]
    )
    assert simplify(tp_trace(tp_compose(Y[0] * Y[1], CD), n) / norm - expected) == 0

    expected = (
        -Omega[1] * d_yz[(2, 1)]
        - Omega[2] * d_yz[(1, 2)]
        + d_xy[(1, 2)] * Nu[1]
        + d_xy[(2, 1)] * Nu[2]
    )
    assert simplify(tp_trace(tp_compose(Y[1] * Y[2], CD), n) / norm - expected) == 0

    # 3 qubit terms
    expected = -U[(0, 2)] * d_yy[(0, 1)] + U[(1, 2)] * d_xx[(0, 1)]
    assert (
        simplify(tp_trace(tp_compose(X[0] * Y[1] * Z[2], CD), n) / norm - expected) == 0
    )
    # 3 qubit terms
    expected = -U[(0, 1)] * d_yz[(0, 2)] - U[(0, 2)] * d_yz[(0, 1)]
    assert (
        simplify(tp_trace(tp_compose(X[0] * Z[1] * Z[2], CD), n) / norm - expected) == 0
    )
