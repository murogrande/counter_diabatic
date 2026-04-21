from itertools import combinations, permutations

import pytest
from sympy import Function, I, S, diff, simplify, symbols
from sympy.physics.paulialgebra import Pauli
from sympy.physics.quantum import Commutator, TensorProduct
from counter_diabatic.utils_test import _extract_tp, _pauli_trace, tp_compose, tp_trace

# ── helpers ───────────────────────────────────────────────────────────────────

s1, s2, s3 = Pauli(1), Pauli(2), Pauli(3)
I2 = s1 * s1
nop = (I2 - s3) / 2


def single_body_op(p, site, n):
    f = [I2] * n
    f[site] = p
    return TensorProduct(*f)


def two_body_op(p1, p2, site1, site2, n):
    f = [I2] * n
    f[site1] = p1
    f[site2] = p2
    return TensorProduct(*f)


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

    H = sum(Omega[i] * X[i] + Mu[i] * Y[i] + Nu[i] * Z[i] for i in range(n_qubits)) + sum(
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
    icomm = (I * Commutator(A, H).expand(commutator=True)).expand(commutator=True).doit()
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


# ── _extract_tp unit tests ────────────────────────────────────────────────────


def test_extract_tp_bare():
    tp = TensorProduct(s1, s2)
    scalar, out = _extract_tp(tp)
    assert out is tp
    assert scalar == S.One


def test_extract_tp_with_scalar():
    tp = TensorProduct(s1, s2)
    expr = 3 * I * tp
    scalar, out = _extract_tp(expr)
    assert out is tp
    assert scalar == 3 * I


def test_extract_tp_pure_scalar_returns_none():
    scalar, tp = _extract_tp(S(5))
    assert scalar is None and tp is None


def test_extract_tp_zero_returns_none():
    scalar, tp = _extract_tp(S.Zero)
    assert scalar is None and tp is None


def test_extract_tp_product_of_two_tps_merges():
    # TP(s1,s2)*TP(s2,s3) — sympy evaluates Pauli products element-wise:
    # s1*s2=I*s3, s2*s3=I*s1 => result is -1*TP(s3,s1), a single TP.
    tp1, tp2 = TensorProduct(s1, s2), TensorProduct(s2, s3)
    scalar, tp = _extract_tp(tp1 * tp2)
    assert isinstance(tp, TensorProduct)
    assert scalar is not None


# ── _pauli_trace unit tests ───────────────────────────────────────────────────


def test_pauli_trace_single_pauli_is_zero():
    for p in (s1, s2, s3):
        assert _pauli_trace(p) == 0


def test_pauli_trace_scalar():
    assert _pauli_trace(S(3)) == 6  # Tr(3*I_2) = 6
    assert _pauli_trace(S.One) == 2  # Tr(I_2)   = 2


def test_pauli_trace_scalar_times_pauli_is_zero():
    assert _pauli_trace(5 * s1) == 0


def test_pauli_trace_pauli_product_same_index():
    # sigma_i^2 = I  =>  Tr(sigma_i^2) = 2
    assert _pauli_trace(s1 * s1) == 2
    assert _pauli_trace(s3 * s3) == 2


def test_pauli_trace_pauli_product_diff_index_zero():
    # Tr(sigma_i * sigma_j) = 0  for i != j (result is traceless Pauli)
    assert _pauli_trace(s1 * s2) == 0
    assert _pauli_trace(s2 * s3) == 0


def test_pauli_trace_triple_product():
    # sigma2*sigma1*sigma3 = (-i*sigma3)*sigma3 = -i  =>  Tr = -2i
    assert _pauli_trace(s2 * s1 * s3) == -2 * I


def test_pauli_trace_nop():
    # nop = (I2-s3)/2  =>  Tr(nop) = (Tr(I)-Tr(s3))/2 = (2-0)/2 = 1
    assert _pauli_trace(nop) == 1


def test_pauli_trace_pauli_times_nop():
    # Tr(si*(I-s3)/2) = (Tr(si) - Tr(si*s3)) / 2
    # s1*s3 = -I*s2 (traceless)  => Tr(s1*nop) = (0-0)/2 = 0
    # s2*s3 =  I*s1 (traceless)  => Tr(s2*nop) = (0-0)/2 = 0
    # s3*s3 = I (identity)       => Tr(s3*nop) = (0-2)/2 = -1
    assert _pauli_trace(s1 * nop) == 0
    assert _pauli_trace(s2 * nop) == 0
    assert _pauli_trace(s3 * nop) == -1


def test_pauli_trace_unevaluated_product_with_add():
    # sigma2 * sigma1*(1-sigma3) expanded = -i*sigma3 + i  =>  Tr = 2i
    expr = s2 * s1 * (1 - s3)
    assert _pauli_trace(expr) == 2 * I


def test_pauli_trace_cancelling_sigma3_product():
    # sigma3*(1-sigma3) vs (1-sigma3)*sigma3 both = sigma3 - sigma3^2 = sigma3-1
    # Their difference is 0  =>  Tr = 0
    assert _pauli_trace(s3 * (1 - s3) - (1 - s3) * s3) == 0


# ── tp_compose / tp_trace unit tests ─────────────────────────────────────────


def test_tp_compose_simple():
    # TP(s1,s2) · TP(s3,s1) = TP(s1*s3, s2*s1)
    A = TensorProduct(s1, s2)
    B = TensorProduct(s3, s1)
    result = tp_compose(A, B)
    expected = TensorProduct(s1 * s3, s2 * s1)
    assert result == expected


def test_tp_compose_with_scalar():
    A = TensorProduct(s1, s2)
    B = 2 * TensorProduct(s3, s1)
    result = tp_compose(A, B)
    assert result == 2 * TensorProduct(s1 * s3, s2 * s1)


def test_tp_compose_distributive_over_add():
    A = TensorProduct(s1, s2)
    B = TensorProduct(s3, s1) + TensorProduct(s2, s3)
    result = tp_compose(A, B)
    assert result == TensorProduct(s1 * s3, s2 * s1) + TensorProduct(s1 * s2, s2 * s3)


def test_tp_compose_arity_mismatch_raises():
    A = TensorProduct(s1, s2)
    B = TensorProduct(s1, s2, s3)
    with pytest.raises(ValueError, match="arity"):
        tp_compose(A, B)


def test_tp_compose_scalar_times_add_of_tp():
    # scalar*(tp1+tp2): no TP at top level of Mul — expand() needed
    tp1 = TensorProduct(s1, s2)
    tp2 = TensorProduct(s2, s1)
    B = I * (tp1 + tp2)  # stored as Mul(I, Add(tp1, tp2))
    A = TensorProduct(s3, s3)
    result = tp_compose(A, B)
    expected = tp_compose(A, I * tp1) + tp_compose(A, I * tp2)
    assert result == expected


def test_tp_compose_zero_after_pauli_cancel():
    # [sigma3, nop]: sigma3*(1-sigma3) = (1-sigma3)*sigma3 => terms cancel => 0
    n = 2
    Z0 = single_body_op(s3, 0, n)
    NOP0 = single_body_op(nop, 0, n)
    comm = (
        (I * Commutator(Z0, NOP0).expand(commutator=True)).expand(commutator=True).doit()
    )
    result = tp_compose(single_body_op(s1, 0, n), comm)
    assert result == 0


def test_tp_trace_traceless():
    # Tr(sigma_i x sigma_j) = Tr(sigma_i)*Tr(sigma_j) = 0
    assert tp_trace(TensorProduct(s1, s2), 2) == 0


def test_tp_trace_identity():
    # Tr(I_2 x I_2) = 2*2 = 4  (I2=s1*s1=1, so TP(1,1) is scalar 1 times identity)
    assert tp_trace(TensorProduct(I2, I2), 2) == 4


def test_tp_trace_mixed():
    # Tr(s1 x I) = Tr(s1)*Tr(I) = 0*2 = 0
    assert tp_trace(TensorProduct(s1, I2), 2) == 0


def test_tp_trace_scalar_term():
    # Tr over a pure scalar (no TP) = 2^n * scalar
    assert tp_trace(S(3), 2) == 12


# ── integration tests ─────────────────────────────────────────────────────────


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

    expand() returns S.Zero; tp_compose must return 0*operator = 0
    rather than raising ValueError.
    """
    n = 2
    Z0 = single_body_op(s3, 0, n)
    NOP0 = single_body_op(nop, 0, n)
    comm = (
        (I * Commutator(Z0, NOP0).expand(commutator=True)).expand(commutator=True).doit()
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
        a[1] * Nu[1] - c[1] * Omega[1] + d_xz[(1, 0)] * U[(0, 1)] + diff(mu_l[1], lam) / 2
    )
    assert (tp_trace(tp_compose(Y[1], CD), n) / norm).equals(expected3)
