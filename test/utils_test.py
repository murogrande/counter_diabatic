from sympy import Add
from sympy import Mul as SMul
from sympy.physics.paulialgebra import Pauli as PauliSym
from sympy.physics.paulialgebra import evaluate_pauli_product
from sympy.physics.quantum import TensorProduct


def _extract_tp(expr):
    """Split a single term into its scalar prefactor and TensorProduct factor.

    Handles three forms:
      - TensorProduct(...)           -> (1, tp)
      - scalar * TensorProduct(...)  -> (scalar, tp)
      - anything else                -> (None, None)  [pure scalar]

    Raises ValueError if the Mul contains more than one TensorProduct
    (unexpanded operator product — use tp_compose first).
    """
    if isinstance(expr, TensorProduct):
        return SMul.identity, expr

    if isinstance(expr, SMul):
        tps = [a for a in expr.args if isinstance(a, TensorProduct)]
        if len(tps) == 1:
            scalars = [a for a in expr.args if not isinstance(a, TensorProduct)]
            return (SMul(*scalars) if scalars else SMul.identity), tps[0]
        if len(tps) > 1:
            raise ValueError(
                f"_extract_tp: {len(tps)} TensorProducts in one Mul — "
                "call tp_compose before tp_trace.\n"
                f"  expr = {expr}"
            )

    return None, None  # pure scalar or unrecognised form


def tp_compose(A, B):
    """Compose two operators written as sums of scalar * TensorProduct: A · B.

    Distributes linearly over Add, then multiplies factor-wise inside each
    TensorProduct:

        (sA * TP(a1, a2)) · (sB * TP(b1, b2)) = sA*sB * TP(a1*b1, a2*b2)

    Raises ValueError if a term cannot be decomposed or if the two
    TensorProducts have different numbers of factors (arity mismatch).
    """
    if isinstance(A, Add):
        return sum(tp_compose(t, B) for t in A.args)
    if isinstance(B, Add):
        return sum(tp_compose(A, t) for t in B.args)

    sA, tpA = _extract_tp(A)
    sB, tpB = _extract_tp(B)

    # Track expanded forms as fallback values for the scalar case below.
    A_exp, B_exp = A, B

    # Case 1: scalar*(tp1+tp2) — no TP at top level of Mul because it's
    # wrapped in an Add.  expand() distributes the scalar in.
    # Case 2: sigma3*(1-sigma3) vs (1-sigma3)*sigma3 — both equal sigma3-sigma3²
    # so the two TPs cancel and expand() returns S.Zero (a scalar, not an Add).
    # In that case tpB stays None; we fall through to the scalar handler below.
    if tpA is None:
        A_exp = A.expand()
        if isinstance(A_exp, Add):
            return sum(tp_compose(t, B) for t in A_exp.args)
        sA, tpA = _extract_tp(A_exp)

    if tpB is None:
        B_exp = B.expand()
        if isinstance(B_exp, Add):
            return sum(tp_compose(A, t) for t in B_exp.args)
        sB, tpB = _extract_tp(B_exp)

    if tpA is None or tpB is None:
        # One side is a pure scalar after all expansion (e.g. 0 when a CD term
        # collapses via Pauli algebra: [sigma3, nop] = 0).
        # scalar * operator  or  operator * scalar  (scalars commute).
        lhs = A_exp if tpA is None else sA * tpA
        rhs = B_exp if tpB is None else sB * tpB
        return lhs * rhs

    if len(tpA.args) != len(tpB.args):
        raise ValueError(
            f"tp_compose: arity mismatch — "
            f"|tpA|={len(tpA.args)}, |tpB|={len(tpB.args)}"
        )

    return sA * sB * TensorProduct(*[a * b for a, b in zip(tpA.args, tpB.args)])


def _pauli_trace(p):
    """Compute Tr of a single-qubit expression in the Pauli algebra.

    Identities used:
      Tr(sigma_k) = 0   for k = 1, 2, 3  (Paulis are traceless)
      Tr(s * I_2) = 2*s for any scalar s
      Tr is linear: Tr(A + B) = Tr(A) + Tr(B)

    Note: SymPy's Pauli(n) only exists for n in {1,2,3}; the identity
    is a plain scalar, handled by the fallback case.
    """
    # Distribute Mul over Add (e.g. sigma1*(1-sigma3) -> sigma1 - sigma1*sigma3).
    # expand() uses Mul._from_args internally so Pauli products stay as Mul objects;
    # evaluate_pauli_product is then needed to reduce them to scalar*(Pauli or 1).
    p = p.expand()

    if isinstance(p, Add):
        return sum(_pauli_trace(t) for t in p.args)

    if isinstance(p, PauliSym):
        return 0  # sigma_1/2/3 are all traceless

    if isinstance(p, SMul):
        # Reduce all Pauli products: sigma_i*sigma_j -> scalar*sigma_k (or scalar).
        # e.g. Mul(-1,s2,s1,s3) -> evaluate -> I (imaginary unit, a scalar)
        p = evaluate_pauli_product(p)
        if isinstance(p, Add):
            return sum(_pauli_trace(t) for t in p.args)
        if isinstance(p, PauliSym):
            return 0
        if isinstance(p, SMul):
            if any(isinstance(a, PauliSym) for a in p.args):
                return 0
            return 2 * p  # pure scalar: Tr(s*I_2) = 2*s
        return 2 * p  # scalar after full reduction

    return 2 * p  # pure scalar s: Tr(s*I_2) = 2*s


def tp_trace(expr, n_qubits):
    """Compute Tr of a two-qubit operator written as sums of scalar * TensorProduct.

    Uses Tr(A⊗B) = Tr(A)*Tr(B) and linearity.  Each TensorProduct factor is
    traced with _pauli_trace.

    Raises ValueError for terms that are not scalar * TensorProduct — typically
    this means an operator product was not expanded with tp_compose first.
    """
    if isinstance(expr, Add):
        return sum(tp_trace(t, n_qubits) for t in expr.args)

    scalar, tp = _extract_tp(expr)

    if tp is None:
        if scalar is None:
            return 2**n_qubits * expr  # Tr(s * I_{2^n}) = 2^n * s
        raise ValueError(
            f"tp_trace: unrecognised term (not scalar*TensorProduct): {expr!r}"
        )

    # Tr(scalar * p1⊗p2⊗…) = scalar * Tr(p1) * Tr(p2) * …
    result = scalar
    for factor in tp.args:
        result *= _pauli_trace(factor)
    return result
