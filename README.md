# Incorrect Data Processing Problems — practice

Practical homework for the course **Incorrect Data Processing Problems**.

The thread running through every practice is that a problem can be *mathematically*
solvable and still be useless in floating point: the answer may not be unique, or it
may swing wildly under a perturbation of the input far smaller than your measurement
error. You will use NumPy, SciPy, scikit-learn and SymPy — not to reimplement them,
but to see where they stop protecting you and what you do about it.

## Setup

Everything is pinned in `pyproject.toml`, so every student, and the grader, runs the
same versions. Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync                 # creates .venv with the pinned versions
uv run pytest           # should report failures - that is the starting point
uv run jupyter lab      # for the notebook practices
```

Prefer plain pip? `python -m venv .venv && source .venv/bin/activate && pip install -e .`

Check it worked: `uv run python -c "import numpy, scipy, sklearn, sympy; print('ok')"`

> **Before practice 7**, run the first two cells of `tutorial_pca.ipynb` once while you
> have a decent connection. It downloads the ~200 MB LFW face dataset into
> `~/scikit_learn_data` and caches it. Doing this during the lab wastes the lab.

## Practices

Part 1 — foundations, one file each. Fill in every `raise NotImplementedError`.

| # | File | Topic |
|---|---|---|
| 1 | `vectors.py` | Vectors, norms, angles, linear systems |
| 2 | `matrices.py` | Matrix arithmetic, rank and basis, matrix norms |
| 3 | `matrix_mapping.py` | Linear and affine mappings |
| 4 | `matrix_decomposition.py` | LU, QR, eigendecomposition, SVD |
| 5 | `regularization.py` | Ridge, Lasso and penalised logistic regression |

Part 2 — notebooks. Fill in every `TODO`, then answer the discussion questions in a
markdown cell in your own words.

| # | File | Topic |
|---|---|---|
| 6 | `tutorial_gmm.ipynb` | Gaussian mixtures and the EM algorithm, from scratch |
| 7 | `tutorial_pca.ipynb` | PCA and Eigenfaces, from scratch |
| 8 | `tutorial_svm.ipynb` | SVM dual and the kernel trick, via `scipy.optimize` |

Practices 1–4 are library-call exercises: the point is to know which tool exists and
what it returns. Practices 5–8 are where you build something. Do them in order —
6–8 reuse the eigendecomposition and norms from 1–4.

## Checking your own work

`tests/` runs against practices 1–5:

```bash
uv run pytest             # everything
uv run pytest tests/test_vectors.py -v     # one practice
uv run pytest -k basis    # one function
```

These are the same *kind* of tests used for grading, but not all of them — the grading
run adds hidden cases covering edge inputs and common shortcuts. Passing `tests/` means
you are on track, not that you are finished. Read what each test asserts: several check
a *property* (a norm really depends on its `order`, L1 really produces zero
coefficients) rather than a fixed number, because that is what the task is about.

The notebooks have no pytest suite. Each one has a `SELF-CHECK` cell instead: run it
after the section above it and it will tell you whether your implementation matches a
reference before you build the next section on top of it.

## What to hand in

- All five `.py` files, with no `NotImplementedError` left.
- All three notebooks, **saved with their outputs** — plots included. A notebook whose
  cells were never run cannot be marked.
- For practice 5, a short written comparison (in `CONCLUSIONS.md` or a markdown cell):
  what happened to the coefficients under L1 vs L2, which alpha and C were chosen, and
  why the penalised model can beat the unpenalised one on data it has never seen.
- For practices 6–8, the discussion answers, in your own words.

Before you submit:

```bash
uv run pytest              # all visible tests green
uv run jupyter nbconvert --execute --inplace tutorial_*.ipynb   # notebooks run top to bottom
```

A notebook that only runs when its cells are executed out of order does not count as
running. Restart the kernel and run all.

## How it is marked

The course total is **100 marks**, split across the eight practices. Part 2 carries
more because you build the algorithms rather than call them.

| # | Practice | Marks |
|---|---|---:|
| 1 | `vectors.py` | 8 |
| 2 | `matrices.py` | 8 |
| 3 | `matrix_mapping.py` | 7 |
| 4 | `matrix_decomposition.py` | 7 |
| 5 | `regularization.py` | 16 |
| 6 | `tutorial_gmm.ipynb` | 18 |
| 7 | `tutorial_pca.ipynb` | 18 |
| 8 | `tutorial_svm.ipynb` | 18 |
| | **Total** | **100** |

Do not change the given function signatures, and do not edit `preprocess()` in
`regularization.py` — the grading tests rely on both. If you think something in a
signature is wrong, say so in your write-up; that argument earns marks.

## Useful links

* [Introduction to Linear Algebra for Applied Machine Learning with Python](https://pabloinsente.github.io/intro-linear-algebra)
* [NumPy linalg reference](https://numpy.org/doc/stable/reference/routines.linalg.html)
* [SciPy linalg reference](https://docs.scipy.org/doc/scipy/reference/linalg.html)
* [scikit-learn: regularisation of linear models](https://scikit-learn.org/stable/modules/linear_model.html)
* [Regularization 1](https://github.com/ethen8181/machine-learning/blob/master/regularization/regularization.ipynb)
* [Regularization 2](https://nbviewer.org/github/justmarkham/DAT8/blob/master/notebooks/20_regularization.ipynb)
