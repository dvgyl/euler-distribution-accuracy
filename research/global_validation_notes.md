# Endpoint cancellation and mesh-dependence validation

## Reproduction

Run from the repository root:

```bash
PYTHONPATH=src python scripts/run_global_validation.py
PYTHONPATH=src python -m unittest discover -s tests -p test_global_validation.py -v
```

The declared dependencies in `requirements.txt` are required. In the present workspace, `mpmath` is installed in `build/deps`; prefix these commands with `PYTHONPATH=build/deps:src` to use that local installation.

## Analytic experiment

On `0<=t<=1`, use

\[
u_t(x)=(aI+b(t)J)x,
\quad a=0.4,
\quad b(t)^2=a^2-\varepsilon\cos(2\pi t),
\quad\varepsilon=0.08,
\]

with `J` the standard planar rotation generator and standard Gaussian initial data. Every field generates `p_t=N(0,e^{2at}I_2)`. Its projected material acceleration is `epsilon*cos(2*pi*t)*x`, so it fails acceleration preservation on an open dense set of times.

Compare uniform nodes against the inverse CDF of

\[
\rho_\delta(t)=1+\delta\cos(2\pi t),\qquad\delta=0.2.
\]

The AP control sets `epsilon=0`, hence `b=a`, while retaining exactly the same target density path. Step counts are all powers of two from 16 through 4096.

## The metric is full Wasserstein distance

The Euler covariance remains scalar at every node. A step multiplies its standard deviation by

\[
\sqrt{(1+ah_k)^2+b(t_k)^2h_k^2}.
\]

The reported error is therefore the exact Gaussian formula

\[
W_2(\widehat p_k,p_{t_k})
=\sqrt2\,|\widehat g_k-e^{at_k}|.
\]

This is neither an empirical OT estimate nor a common-label coupling upper bound. It is evaluated with a compensated log-scale recurrence using `log1p(x)-x` to retain the fine-grid cancellation. The maximum error is taken over all grid nodes. For the quantile mesh, the interior sample uses the node closest to `t=1/4` and records its actual time in the CSV.

## Asymptotic predictions and measured rates

At the finest two meshes, the observed orders are:

| Field and mesh | Endpoint | Near `t=1/4` | Maximum over nodes |
|---|---:|---:|---:|
| Cosine, uniform | 1.999795 | 1.001410 | 0.996874 |
| Cosine, quantile | 0.995523 | 1.001357 | 0.997689 |
| AP control, uniform | 1.999789 | 1.999789 | 1.999789 |
| AP control, quantile | 1.999775 | 1.999811 | 1.999775 |

The predicted coefficients, independently checked by numerical quadrature where applicable, are:

- Uniform cosine endpoint: `N^2 W2(1) -> 0.09001637205706355`.
- Uniform cosine interior: `N W2(1/4) -> 0.00995003409685834`.
- Uniform cosine maximum: `N max_k W2(t_k) -> 0.012177634542334925`, attained asymptotically near `t=0.7601184635770072`.
- Quantile cosine endpoint: `N W2(1) -> 0.008700951365974504`.
- AP control, uniform endpoint: `N^2 W2(1) -> 0.09001637205706355`.
- AP control, quantile endpoint: `N^2 W2(1) -> 0.09570059897410219`.

At `N=4096`, uniform cosine endpoint error is `5.364632539867104e-9`, but maximum-node error is `2.9666322809382333e-6`. Changing only the schedule to the stated quantile mesh raises the cosine endpoint error to `2.117696113770032e-6`. The AP control remains second order on both schedules.

These findings demonstrate a concrete logical distinction: second-order error at one endpoint on one schedule does not imply pointwise acceleration preservation or second-order accuracy throughout the path. The observed order gain can arise from cancellation of a nonzero transported defect. The experiment does not claim that all schedules without acceleration preservation have only first-order endpoint error.

## Independent numerical verification

- The scalar recurrence was checked against a separately coded full `2x2` covariance update for all four field/schedule combinations at `N=16,64,256`. Maximum covariance-entry discrepancy was `1.0436096431476471e-13`; maximum Wasserstein discrepancy was `4.905918925191455e-14`.
- An independent 100-decimal-digit product formula, with quantile nodes independently solved at that precision, was checked for all four combinations at `N=16,256,4096`. Maximum relative Wasserstein discrepancy was `1.4438489865565421e-12`.
- Eight dedicated tests passed. They check the acceleration projection, exact probability mesh, covariance recurrence, weighted asymptotic integrals, distinct endpoint/interior rates, schedule dependence, the signed third-order uniform expansion, and high-precision agreement.
- The complete reproduction, including the twelve high-precision calculations and the figure, took approximately 5.58 seconds in this workspace. This is a reproducibility receipt, not a solver performance comparison.
- `results/global_validation/endpoint_cancellation.pdf` was rendered and visually inspected at its intended 7.2-inch width. The source figure uses embedded TrueType fonts and three compact panels.

The high-precision comparisons are arithmetic cross-checks, not rigorous interval enclosures. This experiment uses an analytic two-dimensional Gaussian field and has no learned-model or image-generation component.

## Output files

- `convergence.csv`: all 36 field/mesh/step-count combinations and measured orders.
- `profiles.csv`: every node on the four finest meshes, with the predicted profiles.
- `covariance_checks.csv`: independent matrix checks.
- `high_precision_checks.csv`: 100-digit values and agreement measurements.
- `summary.json`, `validation.json`: machine-readable constants, findings, and verification receipts.
- `tests.log`: receipt for the eight new tests.
- `endpoint_cancellation.pdf`: the paper figure.

