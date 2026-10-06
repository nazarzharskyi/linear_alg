# Practice 5 — Ridge, Lasso and penalised logistic regression

Written comparison required by the README. All numbers come from `regularization.py`
on the pinned environment (numpy 2.5.3 / scipy 1.18.1 / scikit-learn 1.9.1), with the
single `train_test_split(..., test_size=0.2, random_state=42)` that `preprocess()`
performs, features standardised on training statistics only.

## 1. Regression — `load_diabetes`, 353 train / 89 test, 10 features

| model | chosen `alpha` | train R² | test R² | test RMSE | non-zero coefs | ‖w‖₂ | ‖w‖₁ |
|---|---|---|---|---|---|---|---|
| OLS (baseline) | — | 0.5279 | 0.4526 | 53.85 | 10 / 10 | 71.63 | 183.12 |
| Ridge (L2) | **39.80** | 0.5211 | 0.4605 | 53.46 | 10 / 10 | 38.38 | 99.80 |
| Lasso (L1) | **1.6732** | 0.5189 | **0.4714** | **52.92** | **7 / 10** | 40.05 | 88.65 |

Both `alpha` values were selected by `GridSearchCV` over
`ALPHA_GRID = np.linspace(1e-3, 1e2, 300)` with 5-fold CV on the training set only.

Coefficients side by side:

| feature | OLS | Ridge | Lasso |
|---|---:|---:|---:|
| age | 1.75 | 2.01 | **0.00** |
| sex | −11.51 | −9.91 | −7.95 |
| bmi | 25.61 | 24.13 | 26.18 |
| bp | 16.83 | 15.38 | 15.03 |
| s1 (total cholesterol) | **−44.45** | −5.43 | −4.77 |
| s2 (LDL) | **+24.64** | −4.04 | **−0.00** |
| s3 (HDL) | 7.68 | −8.95 | −11.06 |
| s4 (TCH) | 13.14 | 7.24 | **0.00** |
| s5 (LTG) | 35.16 | 18.95 | 21.94 |
| s6 (glucose) | 2.35 | 3.75 | 1.72 |

### What happened to the coefficients

The `s1…s4` block is the interesting part, and it is the reason this dataset is in the
exercise. Those four serum measurements are strongly correlated with each other, so the
least-squares problem is close to degenerate along that subspace: many very different
coefficient vectors fit the training data almost equally well. OLS picks one of them,
and it picks a wild one — `s1 = −44.45` against `s2 = +24.64`, two large coefficients of
opposite sign that very nearly cancel. That pair is not a finding about cholesterol; it
is the solver resolving an ambiguity it has no information to resolve.

- **L2 (Ridge) shrinks, but keeps everything.** The penalty `alpha·‖w‖₂²` is smooth and
  its gradient `2·alpha·w` vanishes as `w → 0`, so it never has a reason to push a
  coefficient the last step onto exactly zero. All ten coefficients survive, and ‖w‖₂
  falls from 71.63 to 38.38 — nearly a factor of two. The cancelling `s1`/`s2` pair
  collapses from (−44.45, +24.64) to (−5.43, −4.04): ridge splits the shared signal
  between correlated features instead of letting them fight.
- **L1 (Lasso) selects.** The `|w|` penalty has a constant-magnitude subgradient right
  up to the origin, so there is a finite threshold below which a coefficient's
  contribution to the fit cannot pay for its penalty, and it is set to **exactly** zero.
  Lasso dropped `age`, `s2` and `s4` — three of the ten — leaving a 7-feature model that
  is easier to report and to defend. Note *which* ones: it kept one representative of the
  correlated serum block (`s1`, `s3`, `s5`) and discarded the redundant partners.
- **The quantity each one actually minimises shows up in the norms.** Ridge gives the
  smaller ‖w‖₂ (38.38 vs Lasso's 40.05), Lasso the smaller ‖w‖₁ (88.65 vs Ridge's
  99.80). Each method wins on its own penalty, as it must.

**Train R² falls while test R² rises** — 0.5279 → 0.5211 → 0.5189 on train, 0.4526 →
0.4605 → 0.4714 on test. This is the whole bargain in two columns: the penalty *buys*
generalisation by *spending* training fit. Anyone reporting only training R² would
conclude the baseline is best, and would be exactly wrong.

## 2. Classification — `load_breast_cancer`, 455 train / 114 test, 30 features

| model | chosen `C` | train acc | test acc | non-zero coefs | ‖w‖₂ |
|---|---|---|---|---|---|
| Logistic, unpenalised (`C=np.inf`) | ∞ | **1.0000** | 0.9386 | 30 / 30 | **743.17** |
| Logistic L2 (`l1_ratio=0`) | **1.6609** | 0.9868 | **0.9737** | 30 / 30 | 4.39 |
| Logistic L1 (`l1_ratio=1`, liblinear) | **0.15973** | 0.9802 | 0.9649 | **8 / 30** | 3.01 |

`C` was selected by `GridSearchCV` over `C_GRID = np.logspace(-3, 2, 60)`, 5-fold CV,
scoring accuracy. Recall that `C` is an *inverse* penalty: `C = 0.16` is a stronger
penalty than `C = 1.66`.

The eight features L1 kept: `mean concave points`, `radius error`, `worst radius`,
`worst texture`, `worst smoothness`, `worst concavity`, `worst concave points`,
`worst symmetry`. It discarded nine of the ten `mean …` features and kept six of the ten
`worst …` features — a sensible read, since the extreme value across a nucleus sample
carries diagnostic signal that its average washes out.

### Why the penalised model beats the unpenalised one on unseen data

The headline is `train acc = 1.0000` next to `test acc = 0.9386`. That gap is the usual
overfitting story, but on this dataset something sharper is going on, and it is the
point of the course.

**The unpenalised problem has no solution at all.** I checked that the training set is
linearly separable (a hard-margin `LinearSVC` reaches training accuracy 1.0). For
separable data the logistic likelihood has **no maximum**: once a `w` classifies every
training point correctly, scaling it to `2w`, `10w`, `100w` makes every predicted
probability more confident and the likelihood strictly larger. The supremum sits at
‖w‖ = ∞ and is never attained. The maximum-likelihood estimate does not exist.

So what is the 743.17 that scikit-learn reported? It is **where the optimiser gave up**,
nothing more. Varying only `tol` and leaving the data untouched:

| `tol` | ‖w‖₂ | test accuracy |
|---|---:|---:|
| 1e−2 | 2.4 | **0.9912** |
| 1e−4 (default) | 743.2 | 0.9386 |
| 1e−6 | 1405.7 | 0.9386 |
| 1e−8 | 2012.7 | 0.9386 |
| 1e−10 | 2796.7 | 0.9386 |
| 1e−12 | 3448.2 | 0.9386 |

The coefficients grow without bound as the convergence threshold tightens. The
"unregularised model" is not a property of the data — it is a property of the stopping
rule. This is textbook Hadamard ill-posedness: the solution is not unique, so no amount
of numerical care produces a stable answer. (Raising `max_iter` from 100 to 400 000
changes nothing, which is worth stating, because it shows the solver is halting on its
gradient tolerance rather than running out of iterations.)

The `tol=1e-2` row is the joke at the end: stopping early is itself a form of
regularisation, and it scores best of all — 0.9912 — for no principled reason. It is not
a method, it is a lucky accident, and it would be indefensible to report it as a result.

**Adding the penalty makes the problem well-posed.** The objective
`−loglik(w) + ‖w‖²/(2C)` is strictly convex and coercive: it grows without bound as
‖w‖ → ∞, so a unique finite minimiser exists for any finite `C`. ‖w‖₂ drops from 743 to
4.39, and test accuracy rises from 0.9386 to 0.9737 — the penalised model gets **four
more of the 114 test patients right**.

**Stability, measured.** Adding Gaussian noise of scale 1e−3 to the standardised
training matrix and refitting, averaged over 5 draws, the relative change ‖Δw‖/‖w‖ is:

- unpenalised: **8.07e−02**
- L2-penalised (`C = 1.6609`): **6.71e−03**

A perturbation of the inputs 1000× smaller than a standard deviation moves the
unpenalised coefficients by 8%, and the penalised ones by 0.7% — twelve times less.
That ratio, not the accuracy column, is the real argument for the penalty: it is the
difference between a model that reports what the data says and one that reports what the
last digit of the data happened to say.

Note also that the design matrix itself is *not* the problem here: `cond(X_train)` is
only 327.5 for the cancer data and 20.7 for the diabetes data, both unremarkable. The
ill-posedness comes from the geometry of the *likelihood* on separable data, not from a
badly conditioned matrix. Checking the condition number alone would have missed it
entirely.

### Which to pick

- **Ridge/L2 when every feature is plausibly real and correlated.** It is the better pure
  predictor here (test R² 0.4605 vs OLS 0.4526; test accuracy 0.9737, the best of the
  three classifiers), it has a closed form, so it is cheap, and it has no convergence
  story to worry about.
- **Lasso/L1 when you have to hand someone a short list.** It cost 0.0088 of test
  accuracy on the cancer data (0.9649 vs 0.9737) and bought a model with 8 features
  instead of 30; on the diabetes data it cost nothing at all and actually scored best
  (test R² 0.4714). That is usually a good trade, with one caveat: among correlated
  features, *which* representative L1 keeps is itself unstable, so the selected set
  should be treated as one defensible shortlist, not as the identified causes.

## 3. Notes on the repository

The README invites this ("If you think something in a signature is wrong, say so in your
write-up"). Three things I worked around rather than changed, since the grading tests
depend on the current state:

1. **The README's `pytest` instructions cannot work on a student clone.** `.gitignore`
   excludes `tests/` under the heading *"Instructor material must never land in the
   student repo"*, alongside `*_solution.ipynb`, and `git log` confirms the directory
   was never tracked. So the exclusion is deliberate, not an oversight — but the README
   was not updated to match it. It still says "`tests/` runs against practices 1–5" and
   tells you `uv run pytest` "should report failures - that is the starting point",
   while `pyproject.toml` sets `testpaths = ["tests"]`. On any clone a student can
   actually obtain, pytest collects zero tests and exits 5. Either the tests should ship
   (minus the hidden grading cases) or the README should stop promising them. I verified
   practices 1–5 against my own property checks instead, covering what the README says
   the tests look for (a norm that really depends on its `order`, an L1 fit that really
   produces exact zeros).
2. **`pip install -e .` fails** on this layout: setuptools refuses a flat layout holding
   five top-level modules (`Multiple top-level modules discovered in a flat-layout`).
   The README offers that command as the non-`uv` path, so it is broken as documented.
   It needs an explicit
   `[tool.setuptools] py_modules = ["vectors", "matrices", "matrix_mapping", "matrix_decomposition", "regularization"]`.
   Nothing actually requires the install — `pythonpath = ["."]` already makes the
   imports work — so I installed only the pinned dependencies.
3. **Two misleading names**, kept as given because the signatures are frozen:
   - `vectors.cos_between_vectors` returns the angle in degrees, not a cosine. Its own
     docstring says so, so the name is simply wrong; `angle_between_vectors` would be
     right.
   - `matrix_decomposition.svd` is documented as returning "U, S and V", but the third
     factor satisfying `x == U @ diag(S) @ V₃` is Vᵀ (numpy's `Vh`). I return `Vh`,
     because reconstruction is what a caller will test, and I flagged it in the
     docstring. `S` is likewise the 1-D vector of singular values, not a diagonal matrix.
