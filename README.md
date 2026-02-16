# SeqTR
Materials for "[**Worst-Case Error-Controlled Borrowing from Historical Data via Distributionally Robust Optimization**](https://arxiv.org/abs/2602.NNNNN)".

## What is this repo?

### Requirements and Setup
```
# clone the repository
git clone git@github.com:shutech2001/borrowing-based-on-dro.git

# build the environment with poetry
poetry install

# activate virtual environment
eval $(poetry env activate)

# [Option] to activate the interpreter, select the following output as the interpreter.
poetry env info --path
```

## Simulation quickstart
```bash
# run a regular scenario
python scripts/dro_borrowing_simulation.py
```

Outputs:
- `outputs/dro_simulation`: 

## Citation
```text
@article{kimura2026worst,
    author={Kimura, Yui and Tamano, Shu},
    journal={arXiv preprint arXiv:2602.NNNNN},
    title={Worst-Case Error-Controlled Borrowing from Historical Data via Distributionally Robust Optimization},
    year={2026},
}
```

## Contact

If you have any question, please feel free to contact: tamano-shu212@g.ecc.u-tokyo.ac.jp
