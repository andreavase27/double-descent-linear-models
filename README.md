# Double Descent in Linear Models

Experimental project for the course *Statistical Methods for Machine Learning*
(Università degli Studi di Milano, A.Y. 2025/26).

## What is double descent

Classical statistical learning theory describes generalization through the
bias–variance trade-off: as a model becomes more complex, the test error
first decreases and then increases, following a U-shaped curve, so a model
that perfectly fits (interpolates) the training data should generalize
poorly. Modern overparameterized models contradict this picture.
**Double descent** reconciles the two: the test error peaks at the
*interpolation threshold*, where the model has just enough parameters to fit
the training data exactly, and then decreases again as the number of
parameters keeps growing. In linear regression the threshold is reached when
the number of features d equals the number of training points n.

## This project

The project reproduces the **double descent** curve described by
Belkin et al. (PNAS 2019) in linear regression. The number of training points
is fixed (n = 100) and the number of features d is varied, comparing
minimum-norm least squares with ridge regression on:

- a **synthetic dataset** with noisy linear labels (two scenarios: equally
  important features, and a few important features);
- the [**California Housing**](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
  dataset, using random Fourier features to make
  d grow beyond n.

Two extensions compare **gradient descent** with the closed-form solution and
study the **effect of the label noise**.

The full report is in [`report/report.pdf`](report/report.pdf).

## Repository structure

```
├── src/
│   ├── synthetic.py       synthetic data generator
│   ├── real.py            California Housing loader, standardization, random Fourier features
│   ├── models.py          least squares (min-norm), ridge, ridge with cross-validation, gradient descent
│   ├── experiment.py      experimental protocol, summary statistics, theoretical error
│   ├── run_synthetic.py   synthetic data: least squares and ridge
│   ├── run_real.py        California Housing: least squares, ridge, bandwidth
│   ├── run_gd.py          extension: gradient descent vs closed form
│   └── run_noise.py       extension: effect of the label noise
├── figures/               figures produced by the scripts (PDF)
├── report/                LaTeX source of the report and final PDF
└── requirements.txt       exact library versions
```

All algorithms are implemented from scratch; NumPy is used only for basic
linear algebra and scikit-learn only to download the California Housing data.

## Setup

The code was run with Python 3.9.

```bash
git clone https://github.com/andreavase27/double-descent-linear-models.git
cd double-descent-linear-models
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the experiments

Run each script from the root of the repository. Every script prints a short
summary and saves its figures in `figures/`.

| Command | What it does | Figures |
|---|---|---|
| `python src/run_synthetic.py` | Least squares and ridge on synthetic data | `synthetic_ls_*.pdf`, `synthetic_ridge_*.pdf` |
| `python src/run_real.py` | Least squares and ridge on California Housing | `real_*.pdf` |
| `python src/run_gd.py` | Gradient descent vs closed form | `synthetic_gd.pdf` |
| `python src/run_noise.py` | Effect of the label noise | `synthetic_noise.pdf` |

California Housing is downloaded by scikit-learn the first time
`run_real.py` is executed and then read from the local cache.

The files `synthetic.py` and `models.py` also contain quick sanity checks,
run with `python src/synthetic.py` and `python src/models.py`.

## Reproducibility

Every random draw goes through a NumPy generator with a fixed seed
(repetition r uses seed r), so running the scripts again produces exactly the
same numbers and figures.

## Author

Andrea Vaselli — M.Sc. student in Data Science for Economics and Health, Università degli Studi di Milano.