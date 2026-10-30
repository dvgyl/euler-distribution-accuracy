# Second-order distribution accuracy of explicit Euler on a fixed probability path

Dev D. Goyal. Preprint, version 1.0.0, 30 October 2026. This version has not been certified by peer review.

Read the [complete preprint](manuscript.pdf), including the proofs and appendix. The repository contains the source, code, and numerical results needed to reproduce its examples. See [REPRODUCE.md](REPRODUCE.md) for the commands.

## Citation and license

Use [CITATION.cff](CITATION.cff) or [CITATION.bib](CITATION.bib) to cite this version. The paper is available under [CC BY 4.0](LICENSE.md). The original code has no separate software license. The third-party template files retain their own licenses.


This package includes the complete paper, its editable sources, and the numerical calculations. The [manuscript](manuscript.pdf) contains the references and appendix. [REPRODUCE.md](REPRODUCE.md) explains how to regenerate the results and build the paper.

The paper studies when changing the velocity field improves Euler's approximation of a distribution. The change must keep the same probability path, which means that the exact distribution is unchanged at every time. Different fields can give these distributions through different particle trajectories.

Under the stated assumptions, Euler has a second-order bound on distribution error. The nonlinear example achieves this rate, although its particle error remains first order. Particle error compares numerical and exact particles from the same starting point. Distribution error allows other pairings, so its rate can be different.

The calculations use exact formulas for the distributions or deterministic numerical quadrature, which approximates an integral by a weighted sum. No learned model or random sample is needed.

## Find a result

Numerical summaries, tables, and the paper's PDF figures are in [results/](results/), with their calculation settings in [experiments/](experiments/).

| Question | Result |
| --- | --- |
| Do the nonlinear error rates match the proof? | [nonlinear_orders.json](results/nonlinear_orders.json) |
| What are the nonlinear errors and kinetic actions? | [nonlinear_runs.csv](results/nonlinear_runs.csv) |
| How much do finer quadrature rules change the results? | [nonlinear_quadrature_refinement.csv](results/nonlinear_quadrature_refinement.csv) |
| What happens when the correction or solver changes? | [solver_summary.json](results/solver_summary.json) |
| At which step counts does Gaussian sensitivity agree with its asymptotic coefficient? | [gaussian_sensitivity_refinement.csv](results/gaussian_sensitivity_refinement.csv) |
| How do the methods compare at equal NFE? | [equal_evaluation.csv](results/equal_evaluation.csv) |
| How does Heun with the corrected field compare with RK3 with the base field? | [gaussian_higher_order.csv](results/gaussian_higher_order.csv) |

NFE is the number of function evaluations, where each evaluation computes the velocity field. In the data, `base` denotes the field without rotation and `corrected` denotes the field with rotation. The `heun_cubic_cancellation` setting uses a separate rotation coefficient calibrated for Heun.

The Gaussian sensitivity refinement file contains 66 coefficient comparisons, including 42 rows with `passed=False`. These rows fall outside the coefficient tolerance at the stated step counts. The reproduction check requires agreement only on the finest grid for each rotation ratio, and all six of these rows pass. The [numerical comparison criteria](REPRODUCE.md#numerical-comparison-criteria) define this test and the separate convergence-order test.

## Regenerate the figures and tables

Install the dependencies with [REPRODUCE.md](REPRODUCE.md), then run the commands from the package directory. To select that environment, add `PYTHON=.reproduction-venv/bin/python` to each `make` command.

One `make reproduce` run generates all the numerical figures and tables below in `build/reproduction/`. The included files in `results/` stay unchanged. The run also generates the large raw arrays and both node CSV files, which are not included in the package. The notation table belongs to the manuscript source and is built with `make paper`.

| Figure or table | Command | Output |
| --- | --- | --- |
| Figure 1, particle and contour comparison | `make reproduce` | `build/reproduction/figures/particle_distribution.pdf` |
| Figure 2, Gaussian error rates | `make reproduce` | `build/reproduction/gaussian_convergence.pdf` |
| Figure 3, proof pairing | `make reproduce` | `build/reproduction/figures/acceleration_coupling.pdf` |
| Figure 4, nonlinear shape change | `make reproduce` | `build/reproduction/figures/nonlinear_shape.pdf` |
| Figure 5, nonlinear error rates | `make reproduce` | `build/reproduction/figures/nonlinear_convergence.pdf` |
| Figure 6, equal NFE | `make reproduce` | `build/reproduction/figures/equal_evaluation.pdf` |
| Appendix Figure 7, coefficient sensitivity | `make reproduce` | `build/reproduction/figures/gaussian_sensitivity.pdf` |
| Appendix Figure 8, solver dependence | `make reproduce` | `build/reproduction/figures/solver_dependence.pdf` |
| Table 1, low NFE | `make reproduce` | `build/reproduction/equal_evaluation_table.tex` |
| Appendix Table 2, notation | `make paper` | `manuscript.pdf`, from `paper/appendix.tex` |
| Appendix Table 3, Gaussian higher-order solvers | `make reproduce` | `build/reproduction/gaussian_higher_order_table.tex` |
| Appendix Table 4, nonlinear rates | `make reproduce` | `build/reproduction/numerical_macros.tex`, used by the table in `paper/appendix.tex` |

One measured reproduction run took 53.4 s on an Apple M1 CPU with 16 GB of RAM and Python 3.14.6. This includes 63 tests, all numerical calculations, eight figure PDFs, and the generated tables and macros. The `make reproduce` entries share this single runtime; individual figures and numerical tables were not timed separately.

The total of 53.4 s excludes dependency installation, Python imports before timing starts, and the manuscript build. A separate manuscript build took 1.90 s with cached Tectonic resources. These times depend on hardware and system load and do not measure a sampler speedup.

## Find the source and tests

| Location | Contents |
| --- | --- |
| `manuscript.pdf` | Complete paper |
| `paper/` | Editable sources, template files, and license notices |
| `results/` | Included numerical summaries, tables, and PDF figures |
| `src/` | Distribution formulas and numerical methods |
| `experiments/` | Calculation settings and data generation |
| `tests/` | Numerical tests, including deliberately incorrect inputs |
| `scripts/` | Reproduction, figure drawing, and manuscript build commands |
| `requirements.lock.txt` | Exact Python dependency versions |

The tests compare separate calculations, including Cartesian and polar updates and Gaussian matrix and scalar formulas. They check the changes caused by finer quadrature and more accurate reference calculations, along with error orders and field evaluation counts. Deliberately incorrect updates and solver stages check whether the tests detect errors. These numerical checks test the implementation. The paper gives the mathematical proofs.
