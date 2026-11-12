# Reproduction

Use a separate copy of the package for new numerical runs, because the scripts write results. The analytical checks need the packages in `requirements.txt`. The neural calculations also need `requirements-neural.txt`. The file `requirements.lock.txt` records the package versions of the numerical calculations. The scalar experiment and the evaluation-set study were run with NumPy 2.5.4, SciPy 1.18.1, and PyTorch 2.15.0 on Python 3.14. Floating-point results can differ between platforms.

## 1. Independent checks of the analytical claims

```sh
python3 scripts/verify_claims.py
```

This script checks the example of the introduction, the sharp scalar constant, the exact examples, and each value of the population table for the three-dimensional mixture. It writes `results/verification/claims.json` and stops with an error if a check fails.

## 2. Numerical tests

```sh
python3 -m unittest discover -s tests -v
python3 scripts/run_population_check.py
```

## 3. Learned scalar fields with population errors

Evaluate the 30 saved fields and check the derivatives of their Euler updates:

```sh
python3 scripts/run_scalar_population.py --eval-only
python3 scripts/check_scalar_monotonicity.py .
```

Train the fields again and evaluate them (approximately 30 minutes on six processor cores):

```sh
python3 scripts/run_scalar_population.py
```

## 4. Sample-based comparisons in two and eight dimensions

Evaluate the saved fields on the stated evaluation sets:

```sh
python3 scripts/run_sample_only_transfer.py --eval-only
python3 scripts/run_large_bank.py
python3 scripts/run_holdout_diagnostic.py
python3 scripts/summarize_large_bank.py
```

Measure the variability between 230 new evaluation sets (approximately 20 minutes on five processor cores that are not busy):

```sh
python3 scripts/run_evaluation_set_variability.py
```

Train the fields again, in a separate copy only:

```sh
python3 scripts/run_sample_only_transfer.py
python3 scripts/run_extended_ablations.py
```

## 5. Numbers, figures, and documents

```sh
python3 scripts/make_article_numbers.py
python3 scripts/plot_article_figures.py
python3 -m pip install -r requirements-paper.txt
python3 scripts/build_ieee.py
```

The first script writes each quoted number of the new experiments from the saved results. The second script draws the figures of the article. The build needs no new training.
