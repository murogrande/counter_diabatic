# counter_diabatic

Counter-diabatic (CD) gauge potential calculations for neutral-atom quantum systems.

Computes optimal CD driving terms to suppress diabatic transitions during quantum evolution, bridging symbolic derivation (SymPy) with Pasqal hardware emulation (Pulser / emu-mps).

## Install

```bash
pip install -e ".[dev]"
pre-commit install
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
