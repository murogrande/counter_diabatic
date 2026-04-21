# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Counter-Diabatic (CD) gauge potential calculations for neutral-atom quantum systems. Computes optimal CD driving terms to suppress diabatic transitions during quantum evolution, bridging symbolic derivation with Pasqal hardware emulation.

## Dependencies

```bash
pip install sympy numpy torch pulser emu-base emu-mps jupyter
```

## Running

```bash
jupyter notebook conmu.ipynb   # main CD coefficient derivation
jupyter notebook essay.ipynb   # Pulser integration example
```

## Architecture

**conmu.ipynb** — symbolic + numeric CD derivation:
- Builds n-qubit Pauli operators via tensor products
- Hamiltonian: H = Σ(Ω·X + μ·Y + ν·Z) + Σ(U·ZZ)
- CD ansatz with 1-body coefficients (a, b, c) and 2-body (d_xx, d_yy, d_yz, d_xy)
- Solves the CD condition `[H, A_CD] + ∂H/∂λ = 0` via pseudo-inverse of overdetermined linear system
- Uses `torch` tensors with `requires_grad=True` for autodiff on coefficient optimization

**essay.ipynb** — hardware integration:
- Constructs Pulser sequences (rise/sweep/fall pulse shapes) for 2-atom Rydberg systems
- Runs via `MPSBackend` from `emu_mps` with `MPSConfig`
- Connects symbolic CD coefficients to physical pulse parameters
