# Reproduce the paper and results

This guide explains how to regenerate the numerical results and build the complete paper. Run the commands from the package directory with Python 3.14.6 and the dependency versions in the lockfile. No GPU, training data, or model checkpoint is needed.

## Install the dependencies

Create a Python environment with the pinned dependency versions.

```sh
python3.14 -m venv .reproduction-venv
.reproduction-venv/bin/python -m pip install -r requirements.lock.txt
```

## Regenerate the numerical results

The following command runs the complete numerical reproduction.

```sh
make reproduce PYTHON=.reproduction-venv/bin/python
```

The run checks 63 tests before generating numerical summaries, CSV files, raw arrays, tables, and eight PDF figures in `build/reproduction/`. If two calculations differ by more than the specified arithmetic tolerance, the run stops with a nonzero exit status. The command also verifies that the numerical source and configuration remain unchanged during the run.

To select another output directory, call the script directly.

```sh
.reproduction-venv/bin/python -B scripts/reproduce.py --output build/reproduction-new
```

Use a separate directory for each configuration because the script replaces files with the same names in the selected directory. The default is `build/reproduction/`; replacing the included results requires both `--output results` and `--replace-results`.

The settings are in [experiments/config.json](experiments/config.json) for the radial and Gaussian calculations and [experiments/nonlinear_config.json](experiments/nonlinear_config.json) for the nonlinear and solver calculations. The appendix defines the calculations and their error quantities.

Each run creates `raw/`, `nodes.csv`, and `nonlinear_nodes.csv` in its output directory. The raw files contain every primary nonlinear trajectory, while quadrature refinements retain terminal particle arrays and errors at each time node by default. To retain full refinement trajectories, set `retain_refinement_trajectories` to `true` in a copy of `experiments/nonlinear_config.json`. Supply that copy with `--nonlinear-config` and select a separate output directory.

## Understand the calculations and errors

The radial and Gaussian distribution formulas use mpmath with 80-digit arithmetic, alongside separate recursions that compute each solver step from the preceding numerical state. For the linear examples, NumPy updates particle maps, means, and covariances in binary64, the standard double-precision format.

The nonlinear calculation uses deterministic quadrature to approximate integrals over the starting distribution by weighted sums. Calculations with more quadrature nodes check the accuracy of these sums. Integration along the exact flow gives a separate check of continuous kinetic action. Agreement between these calculations supports the implementation but does not replace the mathematical proofs.

The output gives both distribution error and particle error. The quantity `w2` is the Wasserstein distance of order two, defined as the smallest root mean squared separation over all couplings between the numerical and target distributions. A coupling describes a pairing through a joint distribution. The quantity `w2_squared` is the square of this distance. The quantity `particle_rms` is the root mean squared error between numerical and exact particles from the same starting point. This pairing need not give the smallest possible distance between the distributions. Exact zero and floating-point residuals are distinguished in the data.

Small numerical differences can arise between linear-algebra libraries, while font-processing versions can change the bytes of figure PDFs. Compare the numerical values within the stated tolerances and inspect the resulting figures.

## Build the complete manuscript

Install Tectonic 0.17.0, whose first build can require network access to obtain TeX resources. The [paper/](paper/) directory contains the TMLR template and license notices.

```sh
make paper PYTHON=.reproduction-venv/bin/python
```

This command builds `manuscript.pdf` from the included sources, numerical macros, tables, and PDF figures, with the PDF date fixed at midnight UTC. Compiler output is written under `build/`, and the `TECTONIC` variable selects a different compiler executable.

The build uses the figures and `.tex` files in `results/`, rather than the regenerated outputs in `build/reproduction/`. To use regenerated results, copy the required PDF figures and `.tex` files to their corresponding paths in `results/`, then run `make paper`.

The two explanatory figures can also be regenerated separately.

```sh
.reproduction-venv/bin/python -B scripts/render_mechanism.py
```

This command saves `particle_distribution.pdf` and `acceleration_coupling.pdf` in `build/illustrations/`. The contour drawing uses exact Gaussian formulas with identity source covariance and expansion coefficient 0.6, with rotation coefficient 0.6 in the rotating panel. It compares one step of length one from the marked starting point `(1, 0)`. The schematic shows the pairing between numerical and exact particles used in the proof.

## Numerical comparison criteria

The numerical checks use arithmetic tolerances to compare separate calculations. These tolerances are neither theorem bounds nor proved quadrature bounds.

At node k, the Gaussian covariance tolerance is 200 times binary64 machine precision times (k+1) times the larger of 1 and the reference covariance Frobenius norm. Comparisons of distances use the corresponding distribution scale.

The nonlinear reference uses relative and absolute tolerance pairs of (2e-11, 2e-12), (2e-13, 2e-14), and (2.5e-14, 3e-15). Quadrature and reference refinements change one setting at a time, with a quadrature relative-change tolerance of 0.002. Reference discrepancies must remain below 0.001 times the measured error. Scaled coefficient comparisons use a relative-difference tolerance of 0.005, while convergence exponents use an absolute-difference tolerance of 0.01.

The [Gaussian sensitivity refinement file](results/gaussian_sensitivity_refinement.csv) compares the scaled error with the coefficient predicted for small steps. It makes this comparison at every saved uniform step count. The coefficient describes the leading error as the step length tends to zero. The `allowance` column stores the numerical tolerance of 0.005.

For each row, set `h = horizon / steps` and `p = expected_order`. The columns are defined as follows.

| Column | Meaning |
| --- | --- |
| `scaled_error` | `terminal_w2 / h**p` |
| `predicted_coefficient` | Leading asymptotic coefficient from `gaussian_leading_coefficient` in [solver_dependence.py](src/euler_path/solver_dependence.py) |
| `relative_difference` | `abs(scaled_error / predicted_coefficient - 1)` |
| `passed` | `True` exactly when `relative_difference < allowance` |

The comparison uses a strict inequality, with expected order two for `rotation_ratio=1` and order one for the other saved ratios.

All 66 rows are retained, including 42 `False` values and 24 `True` values. The `False` values are evaluated comparisons on coarser grids, where the scaled error lies outside the strict coefficient tolerance. Coefficient agreement is required only at each ratio's finest grid, and all six rows at 8192 steps pass.

A separate test estimates the order from the final grids at 4096 and 8192 steps through `log2(error_4096 / error_8192)`. This estimate must differ from the expected order by less than 0.01, with results saved in [gaussian_sensitivity_orders.json](results/gaussian_sensitivity_orders.json).

Agreement with the coefficient shows how closely the scaled error follows its limit for small steps. Separate matrix comparisons check the Gaussian formulas at every saved time node. The coefficient test does not measure the accuracy of the numerical updates.

## Compare methods at equal NFE

NFE is the number of function evaluations, each of which computes the velocity field. The [experiments/equal_evaluation_config.json](experiments/equal_evaluation_config.json) settings use powers of two from 2 through 4096 evaluations and quadrature orders 64, 128, and 256. Displayed values use order 128. With N evaluations, each Euler method takes N steps, whereas Heun with the base field takes N/2 steps because each step requires two evaluations.

The complete run saves terminal arrays for every NFE value and quadrature order, together with independent Cartesian comparisons, quadrature refinements, and an 80-digit Gaussian identity check. The reference checks compare the implicit radius equation with an ordinary differential equation (ODE). Equal NFE does not mean equal runtime or show that one method is faster.

The comparison produces `equal_evaluation.csv`, `equal_evaluation_table.tex`, `equal_evaluation_summary.json`, and the equal-NFE figure, with unrounded errors used to calculate the table ratio. The nonlinear results apply to the stated source mixture and coefficients. The approximate crossover marks in the Gaussian sensitivity figure show where the leading error terms have similar size when the correction is close to its matched value.

## Compare Gaussian Heun and RK3

The [experiments/gaussian_higher_order_config.json](experiments/gaussian_higher_order_config.json) settings compare methods at 12, 48, 192, and 384 evaluations. Heun uses the corrected field with rotation coefficient `a/sqrt(3)` and two evaluations per step, while Kutta's third-order Runge-Kutta method, RK3, uses the base field and three. This comparison is part of the complete reproduction and can also be run separately.

```sh
PYTHONPATH=src .reproduction-venv/bin/python -B -m euler_path.gaussian_higher_order --config experiments/gaussian_higher_order_config.json --output build/gaussian-higher-order
```

The outputs are `gaussian_higher_order.csv`, `gaussian_higher_order_raw.csv`, `gaussian_higher_order_checks.csv`, `gaussian_higher_order_table.tex`, and `gaussian_higher_order_summary.json`. The raw CSV contains both precision levels and independent matrix checks of the solver stages.

Complex polynomial calculations use 50 and 80 decimal digits, while matrix calculations use 20 extra digits to resolve covariance distances obtained by subtracting nearly equal traces. The relative comparison tolerance is `1e-35`, and each matrix calculation counts its actual field evaluations.
