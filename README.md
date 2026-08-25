# Frank-Wolfe Optimization for the Minimum Enclosing Ball Problem

Implementation and empirical comparison of three Frank-Wolfe (conditional
gradient) variants — **Standard**, **Pairwise**, and **Fully-Corrective** —
applied to the dual formulation of the **Minimum Enclosing Ball (MEB)**
problem, evaluated on MNIST, Fashion-MNIST and CIFAR-10, with a downstream
application to geometric anomaly detection.

> Originally developed as a university optimization project; reorganized
> here into a modular, reproducible codebase. Full mathematical derivation
> and discussion in [`report.pdf`](./report.pdf).

## Why Frank-Wolfe for MEB

The MEB problem — find the smallest ball containing a set of points — has a
dual formulation that reduces to minimizing a convex quadratic over the
probability simplex. This is a textbook fit for Frank-Wolfe methods: the
Linear Minimization Oracle over the simplex is a trivial `argmin`, the exact
line search has a closed form (quadratic objective), and the algorithm's
sparse-iterate property yields **core-sets** — the ball is defined by only a
handful of boundary points, out of thousands.

## What's compared

| Solver | Idea | Convergence |
|---|---|---|
| **Standard FW** | Moves toward a new atom, shrinks all others | Sublinear O(1/t), suffers from zig-zagging near the boundary |
| **Pairwise FW** | Swaps mass between a "forward" and an "away" atom | Linear rate for strongly convex objectives |
| **Fully-Corrective FW** | Re-optimizes exactly over the active set every iteration | Fewest outer iterations, higher per-iteration cost |

Results (30 independent MEBs — 3 datasets × 10 classes): the Standard method
fails to converge within a 15s budget on every MNIST class, while Pairwise
and Fully-Corrective converge on all 30 cases in well under a second,
reaching duality gaps of 10⁻¹¹–10⁻¹².

## Repository structure

```
├── src/
│   ├── solvers.py      # the three Frank-Wolfe variants + smart initialization
│   ├── data.py          # dataset loading and MEB problem construction
│   ├── experiment.py     # per-class experiment runner + summary table
│   ├── anomaly.py        # geometric anomaly scoring
│   └── plotting.py       # convergence plots, heatmap, anomaly image grids
├── notebooks/
│   └── experiments.ipynb # end-to-end pipeline, calls into src/
├── report.pdf            # full write-up: theory, derivations, results
└── requirements.txt
```

## Anomaly detection

Each trained MEB is treated as a compact model of "what a normal member of
that class looks like". A test point's anomaly score is its distance from
the ball's center, normalized by the ball's radius:

```
ratio(z) = ||z - center|| / radius
```

`ratio > 1` → outside the ball (outlier) · `ratio ≈ 1` → borderline ·
`ratio << 1` → typical example. On MNIST, the highest-ratio digits are
genuinely atypical handwriting; on CIFAR-10, a single visually noisy image
per class can inflate the radius for the whole class — a nice illustration
of the MEB's sensitivity to extreme points.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/experiments.ipynb
```

MNIST and Fashion-MNIST load via `tensorflow.keras.datasets`; CIFAR-10 is
fetched from an OSF mirror and cached locally on first run.

## Notes on LLM usage

Core algorithmic logic and theoretical adaptations were developed
independently; an LLM was used as a debugging/optimization assistant
(vectorization, numerical stability, dataset loading) and for generating
plots and LaTeX formatting — see `report.pdf`, Section 6, for the original
disclosure.
