import pytest
from sympy import  I, S
from sympy.physics.quantum import Commutator, TensorProduct
from counter_diabatic.utils_ import (
    _extract_tp,
    _pauli_trace,
    tp_compose,
    tp_trace,
    single_body_op,
    two_body_op,
    s1,
    s2,
    s3,
    nop,
    I2,
)

# ── _extract_tp unit tests ─────


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


# ── _pauli_trace unit tests ────────


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


# ── tp_compose / tp_trace unit tests ────────


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
        (I * Commutator(Z0, NOP0).expand(commutator=True))
        .expand(commutator=True)
        .doit()
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
