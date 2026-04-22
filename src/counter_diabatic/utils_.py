import torch
from sympy import Add
from sympy import Mul as SMul
from sympy.physics.paulialgebra import Pauli
from sympy.physics.paulialgebra import evaluate_pauli_product
from sympy.physics.quantum import TensorProduct
from itertools import combinations
from sympy.physics.quantum import Commutator, TensorProduct
from sympy import I

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

    if isinstance(p, Pauli):
        return 0  # sigma_1/2/3 are all traceless

    if isinstance(p, SMul):
        # Reduce all Pauli products: sigma_i*sigma_j -> scalar*sigma_k (or scalar).
        # e.g. Mul(-1,s2,s1,s3) -> evaluate -> I (imaginary unit, a scalar)
        p = evaluate_pauli_product(p)
        if isinstance(p, Add):
            return sum(_pauli_trace(t) for t in p.args)
        if isinstance(p, Pauli):
            return 0
        if isinstance(p, SMul):
            if any(isinstance(a, Pauli) for a in p.args):
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


# ── Pauli basis (same ordering as ansatz A) ──────────────────────────────
def make_basis_ops(n):
    """Return (ops, eq_labels, x_labels): rows=equations, cols=unknowns."""
    ops, eq_labels, x_labels = [], [], []

    def paired_permutations(n):
        return [
            (i, j) for i, j in combinations(range(n), 2) for i, j in [(i, j), (j, i)]
        ]

    for i in range(n):
        ops.append(single_body_op(s1, i, n))
        eq_labels.append(f"X{i+1}")
        x_labels.append(f"a_{i+1}")
        ops.append(single_body_op(s2, i, n))
        eq_labels.append(f"Y{i+1}")
        x_labels.append(f"b_{i+1}")
        ops.append(single_body_op(s3, i, n))
        eq_labels.append(f"Z{i+1}")
        x_labels.append(f"c_{i+1}")
    for i, j in combinations(range(n), 2):
        ops.append(two_body_op(s1, s1, i, j, n)); eq_labels.append(f'X{i+1}X{j+1}'); x_labels.append(f'dxx_{i+1}{j+1}')
        ops.append(two_body_op(s2, s2, i, j, n)); eq_labels.append(f'Y{i+1}Y{j+1}'); x_labels.append(f'dyy_{i+1}{j+1}')
        ops.append(two_body_op(s3, s3, i, j, n)); eq_labels.append(f'Z{i+1}Z{j+1}'); x_labels.append(f'dzz_{i+1}{j+1}')
    for i, j in paired_permutations(n):
        ops.append(two_body_op(s1, s2, i, j, n)); eq_labels.append(f'X{i+1}Y{j+1}'); x_labels.append(f'dxy_{i+1}{j+1}')
        ops.append(two_body_op(s1, s3, i, j, n)); eq_labels.append(f'X{i+1}Z{j+1}'); x_labels.append(f'dxz_{i+1}{j+1}')
        ops.append(two_body_op(s2, s3, i, j, n)); eq_labels.append(f'Y{i+1}Z{j+1}'); x_labels.append(f'dyz_{i+1}{j+1}')
    return ops, eq_labels, x_labels


# ── Build constant coefficient tensors (symbolic, done once) ─────────────


def build_coefficient_tensors(n:int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exploits linearity of H to decompose M and b into parameter-free matrices:

        M(Ω,μ,ν,U) = Σ_k [ Ω_k·M_Om[k] + μ_k·M_Mu[k] + ν_k·M_Nu[k] ]
                     + Σ_{k<l} U_kl · M_U[(k,l)]

        b(Ω̇,μ̇,ν̇)  = Σ_k [ Ω̇_k·b_Om[k] + μ̇_k·b_Mu[k] + ν̇_k·b_Nu[k] ]

    where M[i,j] = Tr(P_i · i[P_j, H_unit]) / 2^(n+1)  (one H term at a time)
          b[i]   = -Tr(P_i · dH_unit)       / 2^(n+1)

    Each returned object is a constant float64 torch tensor (no grad).
    """
    basis_ops, _, _ = make_basis_ops(n)
    norm = 2 ** (n +1)
    N = len(basis_ops)

    def _M_for(H_unit):
        mat = torch.zeros((N, N))
        for i, P_i in enumerate(basis_ops):
            for j, P_j in enumerate(basis_ops):
                comm = (
                    (I * Commutator(P_j, H_unit).expand(commutator=True))
                    .expand(commutator=True)
                    .doit()
                )
                if comm == 0:
                    continue
                mat[i, j] = float(tp_trace(tp_compose(P_i, comm), n) / norm)
        return torch.tensor(mat, dtype=torch.float64)

    def _b_for(dH_unit):
        vec = torch.zeros(N)
        for i, P_i in enumerate(basis_ops):
            entry = -tp_trace(tp_compose(P_i, dH_unit), n) / norm
            if entry == 0:
                continue
            vec[i] = float(entry)
        return torch.tensor(vec, dtype=torch.float64)

    print(f"Building coefficient tensors  n={n}  N={N} ...")
    M_Om = [_M_for(single_body_op(s1, k, n)) for k in range(n)]
    M_Mu = [_M_for(single_body_op(s2, k, n)) for k in range(n)]
    M_Nu = [_M_for(single_body_op(s3, k, n)) for k in range(n)]
    M_U = {
        (i, j): _M_for(two_body_op(s3, s3, i, j, n))
        for i, j in combinations(range(n), 2)
    }
    b_Om = [_b_for(single_body_op(s1, k, n)) for k in range(n)]
    b_Mu = [_b_for(single_body_op(s2, k, n)) for k in range(n)]
    b_Nu = [_b_for(single_body_op(s3, k, n)) for k in range(n)]
    return M_Om, M_Mu, M_Nu, M_U, b_Om, b_Mu, b_Nu


def make_M_torch(M_Om, M_Mu, M_Nu, M_U, Omega, Mu, Nu, U) -> torch.Tensor:
    """Assemble M — gradients flow through Omega/Mu/Nu/U."""
    # change this sum to stack().sum(dim=0)
    M = sum(
        Omega[k] * M_Om[k] + Mu[k] * M_Mu[k] + Nu[k] * M_Nu[k]
        for k in range(len(Omega))
    )
    for (i, j), M_ij in M_U.items():
        M = M + U[(i, j)] * M_ij
    #  = torch.stack(
    # [
    # Omega[k] * M_Om[k]
    # + Mu[k] * M_Mu[k]
    # + Nu[k] * M_Nu[k]
    # for k in range(len(Omega))
    # ]
    # ).sum(dim=0) , then add the interaction terms
    return M


def make_b_torch(b_Om, b_Mu, b_Nu, dOmega, dMu, dNu) -> torch.Tensor:
    """Assemble b — gradients flow through dOmega/dMu/dNu."""
    return sum(
        dOmega[k] * b_Om[k] + dMu[k] * b_Mu[k] + dNu[k] * b_Nu[k]
        for k in range(len(dOmega))
    )