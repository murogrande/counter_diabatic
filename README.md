# counter_diabatic

Counter-diabatic (CD) gauge potential calculations for neutral-atom quantum systems.

Computes optimal CD driving terms to suppress diabatic transitions during quantum evolution, bridging symbolic derivation (SymPy) with Pasqal hardware emulation (Pulser / emu-mps).

## Install

```bash
pip install -e ".[dev]"
pre-commit install
```

## Layout

```
src/counter_diabatic/
    pulse_hamil.py          # Hamiltonian ↔ pulse parameter conversions
    sequence_2_matrix.py    # CD linear system (A matrix, b vector, solvers)
test/
    test_sequence_2_matrix.py
    utils_test.py
```

## Notebooks

| Notebook | Description |
|---|---|
| `conmu.ipynb` | Symbolic + numeric CD coefficient derivation |
| `essay.ipynb` | Pulser hardware integration example |

## Tests

```bash
pytest test/ -v
```

## License

MIT
