# A two-grid characterization of Euler sampling accuracy in flow matching

This package contains the article, the supplementary material, the code, the settings, the saved results, and the coefficients of the fitted networks.

## Read the article

The main article and the supplementary material are separate documents. Each document starts at page one and has its own references. Supplementary section, equation, theorem, and table numbers have the prefix S.

The article asks what accuracy at two Euler step counts shows about a velocity field. The theorems use population Wasserstein errors against a prescribed distribution path, with stated regularity assumptions. Three groups of experiments support the theorems:

- exact population computations for Gaussian and mixture paths,
- learned scalar fields, for which the population errors are computed by quadrature,
- sample-based comparisons in two and eight dimensions, with a study of the variability between evaluation sets.

## Build

Install Tectonic and the Python package in `requirements-paper.txt`. Then run:

```sh
python3 -m pip install -r requirements-paper.txt
python3 scripts/build_ieee.py
```

The build resolves the references between the two documents and writes the article and supplement as PDF files. The official IEEE TAI template is used.

## Reproduce

Read `REPRODUCE.md`. The numerical tests check finite calculations. The analytical results depend on the proofs. All tested cases are supplied, including the cases with unfavorable results.

## License

The manuscript uses CC BY 4.0. The original code has no separate software license. Third-party style files keep their terms. See `LICENSE.md`.
