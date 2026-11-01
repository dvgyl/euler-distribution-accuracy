# Revision report: constructive and certified Euler distribution accuracy

**Date:** 1 November 2026  
**Author:** Dev D. Goyal  
**Revised title:** *Constructing and Certifying Second-Order Euler Distribution Accuracy*  
**Starting point:** repository commit `07749ae0b4c37a8ac0a29dad6f3c5dbd803052da` and the supplied version 1.0.0 ZIP/PDF. All 95 archived files were checked against the live repository's Git blob hashes and matched.  
**Status:** research revision draft with analytical proofs, reproducible experiments, and a targeted literature audit. No journal submission, publication, or release is represented by this revision.

## Outcome

The revision adds a concrete nonradial construction and a provable way to recover its correction, rather than leaving the paper at a conditional acceleration-preservation theorem. It also makes the acceleration residual operational and proves that this residual is exactly the leading one-step Wasserstein error coefficient. Stronger convex-observable obstructions expose infeasible paths that pass every second-moment/covariance test.

These are meaningful extensions of the supplied manuscript. They do **not** establish a universal correction procedure for arbitrary learned paths, a practical sampling speedup, or absolute priority over all literature. The contribution uses established gauge freedom, Hilbert projection, transport calculus, and Wasserstein tangent geometry; those ingredients are attributed rather than rebranded as new discoveries.

## 1. New mathematical contributions

### 1.1 A full-support nonradial, nonlinear construction

Identify the plane with the complex numbers. For a standard planar Gaussian reference variable Z, let

`F_t(z) = exp((1+i) a t) z + epsilon exp((1+i) a sqrt(2) t) z^2 exp(-|z|^2)`.

The target path is the pushforward of the Gaussian reference by F_t. If

`(4/e) epsilon exp(a (sqrt(2)-1) T) < 1`,

F_t is a globally smooth bi-Lipschitz diffeomorphism. Its inverse is computed by a uniformly contracting fixed-point iteration. The generator `u(t,F_t z)=partial_t F_t(z)` has the exact identity

`partial_tt F_t(z) = 2 a^2 D F_t(z) J z`.

The reference rotation Jz preserves the Gaussian, so the material acceleration preserves the target marginal at every physical time. The global Lipschitz, smoothness, and moment assumptions of the original Euler theorem are all proved. Thus the fixed constructed field has O(H^2) full-Wasserstein error on arbitrary grids.

The family is genuinely outside the previous Gaussian/radial/affine/product examples. Isotropic covariance and two incompatible complex third moments prove that the deformed law is neither an affine image of a radial law nor an invertible affine image of a product of two independent real variables. This is not an inference from a plot.

A finite collection of angular Fourier modes with arbitrary smooth rapidly decaying radial envelopes gives a broader family under an explicit small-perturbation bound. The term “angular harmonic” refers to the Fourier mode, not to spatial harmonicity of a map.

**Gaussian-start variant.** Replacing the second coefficient by `epsilon sinh((1+i) a sqrt(2) t)` keeps the acceleration identity and makes F_0 the identity. Hence the target can start from standard Gaussian noise and become nonradial. Its regularity and nonproduct properties are proved. The numerical experiments use the exponential variant; the Gaussian-start variant is not presented as separately tested.

**Files:** `paper/revision_construction.tex`, `paper/revision_construction_proofs.tex`.

### 1.2 Globally recoverable correction in a stated family

Every field

`u_omega(t,F_t z) = partial_t F_t(z) + (omega-a) D F_t(z) J z`

generates the same exact path. The reference rotation is a path-preserving gauge. An explicit solenoidal acceleration candidate `c D F_t Jz` gives an exact continuum L2 upper certificate R_t(omega,c) for the projected acceleration residual.

Angular orthogonality reduces R_t squared to two weighted squares with known Gaussian moments. Eliminate c analytically; the resulting objective is a coercive quartic in omega. Enumerating every real root of its derivative, and interval endpoints if constrained, finds the global minimum. The sum-of-squares identity proves that its unique zero is `(omega,c)=(a,2a^2)`.

The experiment discovers this minimum from the polynomial coefficients rather than receiving the correction as its initialization. It also records the nonzero local minimum that traps an ordinary local optimizer. This is a complete finite-dimensional guarantee, not a convergence theorem for an arbitrary neural optimizer.

The explicit phase coupling proves an O(h^2) full-Wasserstein upper bound and an attained first-order common-start particle limit. It does not assert that this upper coupling is optimal.

### 1.3 Exact local Wasserstein error coefficient

Let Pi_p project onto the closure of smooth compactly supported gradients in L2(p). The paper now proves

`lim_{h downarrow 0} W2((Id+h u_t)#p_t, p_{t+h}) / h^2 = ||Pi_{p_t} a_t|| / 2`.

This strengthens the old local necessity result to a sharp quantitative characterization. The proof needs no density, no classical flow for the projected solenoidal component, and no invertibility of Euler's map. The upper bound uses the new frozen-flow estimate; the lower bound uses Taylor expansion under an optimal coupling and the dual gradient norm.

The residual is canonical **for the selected generator**. It is not the intrinsic covariant acceleration of the marginal curve. Changing the generator on the same path can change the residual, as the expanding Gaussian example demonstrates. Recent Neural Wasserstein Splines work is cited to distinguish these notions.

### 1.4 Residual, approximation, and learned-path guarantees

The global projected-residual theorem gives

`max_k W2 <= exp(LT) [D_H sum_j h_j^3 + (B_H/2) sum_j h_j^2 epsilon_{t_j}]`,

with explicit D_H and B_H, where `epsilon_t=||Pi_{p_t} a_t||`. The original acceleration retains its global Lipschitz assumption, but the projected solenoidal component no longer needs a regular flow. This removes the main gap between an abstract weighted projection and the old residual proposition.

Additional results provide:

- upper and lower primal/dual certificates with an exact squared certificate gap;
- weighted Poisson existence under a stated Poincare assumption;
- an upper bound using a Galerkin strong residual and the Poincare constant;
- explicit effects of quadrature and acceleration-approximation errors;
- compactly supported stream potentials with uniform global regularity;
- existence and a finite-net convergence guarantee for compact finite-dimensional gauge search;
- a continuous-time envelope using a regular explicit upper-certificate field;
- a target-path error bound separating initialization, cubic integration remainder, acceleration residual, and projected velocity/path discrepancy;
- a population excess flow-matching regression-loss implication;
- a density/score mismatch bound under explicit norms and Poincare conditions.

A finite Galerkin projection norm is a **lower** bound, so minimizing it alone is not a certificate. The paper explicitly prevents that interpretation. The learned-path theorem requires a mixed jerk bound along the integrated field's flow from the target measure; that condition is not implied by training loss.

**Files:** `paper/revision_certification.tex`, `paper/revision_certification_proofs.tex`.

### 1.5 Stronger feasibility tests and an actual local error floor

Every convex spatial observable must have a convex expectation over time under acceleration preservation. A Hessian-weighted minimum-energy problem gives quantitative residual lower bounds. The paper also derives a covariance semidefinite necessary condition and convexity of every directional standard deviation.

The strongest separating example is a smooth radial Gaussian pushforward on `[0,1/2]`:

`G_t(z) = z + (t^2/2) [(-1+5 exp(-|z|^2/2))/4] z`.

It passes the original minimum-energy second-moment test and the full covariance semidefinite test throughout that interval. Nonetheless, its fourth moment is strictly concave, so no admissible generator can preserve the path under its material acceleration. Every such generator has

`epsilon_0 >= sqrt(3)/16`,

and the sharp local theorem yields the actual one-step error floor

`lim_{h downarrow 0} W2(Euler_step#p_0,p_h)/h^2 >= sqrt(3)/32`.

The displayed minimum-action base field has the exactly computable residual `sqrt(23)/12` at every time. This is distinguished from the smaller universal floor for arbitrary alternative generators. The convex chain rule itself is standard; the concrete strict separation and quantitative obstruction are the added content.

**Files:** `paper/revision_feasibility.tex`, `paper/revision_feasibility_proofs.tex`.

## 2. Experimental evidence and its limits

The principal new example uses a=0.4, epsilon=0.18, T=1. Its global contraction bound is approximately 0.312604.

| Quantity at 512 Euler steps | Evaluated value | Observed order from 256 to 512 |
| --- | ---: | ---: |
| Corrected full-W2 lower bound | 3.438502e-7 | 1.99831 |
| Corrected full-W2 coupling upper bound | 4.880645e-7 | 1.99915 |
| Corrected common-start particle RMS | 6.608556e-4 | 0.99925 |
| Base Euler full-W2 upper bound | 3.303088e-4 | 0.99914 |

These are deterministic quadrature evaluations of continuum bounds. They are not exact nonradial optimal-transport distances, formal interval enclosures, or finite-sample uncertainty intervals. Refinement from 64 radial/64 angular nodes to 96/128 changes the corrected upper estimate by less than 8e-17. Independent DOP853 reference trajectories agree with the analytic trajectories within 6.5e-12 at the tested intermediate times.

The global correction search recovers omega=0.3999999999999999 and c=0.31999999999999995. Its residual formula evaluates to approximately 2.07e-16. The exact zero is certified by the analytical identity, not by interpreting machine precision as proof. A bad local minimum at omega approximately -0.392394 has residual upper bound 0.0724104.

Fixed +/-2% rotation perturbations restore an approximately first-order trend. A separate infeasible anisotropic Gaussian path supplies an all-field residual lower bound and a restricted affine optimum. A noisy, one-coefficient supervised velocity model over 512 observations and three seeds supplies an analytically tractable learned-path floor. This is a real but deliberately limited parametric fit, not a neural mixture-transport experiment.

### The unfavorable comparison remains visible

At equal 384 field evaluations:

| Method | Evaluated W2 lower bound | Evaluated W2 upper bound | Particle RMS |
| --- | ---: | ---: | ---: |
| Corrected Euler | 6.110505e-7 | 8.675000e-7 | 8.809872e-4 |
| Base Heun | 6.072879e-7 | 6.085053e-7 | 6.085053e-7 |
| Base RK3 | 1.066869e-9 | 1.070191e-9 | 1.070191e-9 |

Heun's upper estimate is slightly below corrected Euler's lower estimate, and the ordering survives an independent refinement at this exact budget. This is numerical evidence favoring Heun on this example; no interval-certified ordering is asserted. RK3 is substantially more accurate. All stages include inversion cost. Runtimes are comparable, and the experiments establish no speedup. Correction search time, inverse iteration counts, derivative-free analytical evaluation, hardware/thread settings, and the scope of tracked memory measurements are documented in the supplementary methods and CSV files.

## 3. Verification performed

- 72/72 tests passed: 63 established tests and 9 new checks.
- Complete established reproduction passed: 81 baseline runs; 16,983 baseline node comparisons; 33 nonlinear runs; 24,600 nonlinear node comparisons; 72 sensitivity runs; 66 solver runs; 22 flow-matching runs; 36 equal-evaluation rows; 4 Gaussian higher-order rows; 8 original figures.
- The original rounded manuscript numerical macros reproduced byte-for-byte. Remaining unrounded differences were within the established tolerances.
- Independent checks compared trajectory finite differences with acceleration, inverse reconstruction, exact Fourier residuals with quadrature, global vs local optimization, moment formulas, phase couplings, and actual solver stage counts.
- New constructions and proofs were reviewed through separate model derivations. One time-envelope definition gap was found and fixed; the valid statement requires an explicit regular certificate field rather than differentiability of an abstract projection.
- The complete manuscript compiles with resolved references and no overfull boxes or duplicated PDF destinations. The main article and representative proof/figure/table pages were rendered and visually inspected.

This is not Lean/Coq formalization, a formal floating-point proof, human peer review, or an exhaustive verification of the entire mathematical literature. The revision does not claim otherwise.

## 4. Response to all 20 requested changes

| Plan item | Implementation and status |
| --- | --- |
| 1. Potential-based corrections | Compact-support stream/tensor gauges, explicit regularity, finite-family scope; reference-coordinate construction provides a stronger exact example. |
| 2. Computable weighted residual | Canonical projection, exact local coefficient, Poisson and primal/dual certificates; exact analytic upper certificate in the new example. |
| 3. Optimize the relevant component | Compact finite-net guarantee and globally solved quartic family. Local failure documented. No general neural convergence claim. |
| 4. Broader existence class | Full-support nonradial, nonlinear laws with proved exclusion of affine-radial/product structure; finite angular modes and Gaussian-start variant. |
| 5. Stronger obstructions | Convex-observable hierarchy, covariance semidefinite condition, strict fourth-moment example, universal residual and local-error floors. |
| 6. Operational residual theorem | Projection-based global bound with explicit constants and O(H) node-tolerance requirement; fixed-field vs H-dependent-family distinction. |
| 7. Learned-path error | Combined target-measure theorem with projected velocity defect, mixed jerk condition, and population regression-loss consequence. |
| 8. Construction errors | Explicit quadrature/acceleration/score error terms and assumptions. Numerical experiments use refinement, not formal quadrature intervals. |
| 9. Discover a correction | Global quartic root enumeration from the objective, exact algebraic validation, and local-optimizer trap. |
| 10. Residual plots | Analytic continuum upper certificates versus time/optimization/perturbation; not mislabeled as exact projection norms away from the zero. |
| 11. Intermediate path preservation | Exact trajectory identity plus independent intermediate-time high-accuracy ODE checks. |
| 12. Learned-field evidence | Restricted one-coefficient noisy fit with an analytic learned path and model-error floor. No neural mixture, moderate-dimensional, or image benchmark is claimed. |
| 13. Actual cost | Equal NFE, batch wall time, inversion work, correction search and documented memory scope; unfavorable Heun/RK3 results retained. |
| 14. Introduction | Rewritten around feasibility, construction, certification, and limits. |
| 15. Organization | Compact IEEE main article with all complete new proofs and retained established results in the supplement. |
| 16. Related work | Hurault v2 inspected; gauge/divergence/Wasserstein-solver comparisons updated; recent Neural Wasserstein Splines distinction added. Targeted rather than exhaustive novelty audit. |
| 17. Four regimes | Exact, certified approximate, proven infeasible, and unknown feasibility distinguished. |
| 18. Redundancy | Main exposition condensed; original results and inconvenient solver/particle limitations retained in the supplement. |
| 19. Title/abstract | Changed to reflect the construction and certification actually proved, with explicit scope. |
| 20. IEEE and audit | Standard IEEEtran layout, impact statement, truthful AI disclosure, complete proof and numerical audit. Current TAI-specific limits could not be fully verified; see below. |

## 5. Literature and submission scope

`research/literature_audit.md` records exact sources, versions, and which portions were read. The audit found no prior result that clearly subsumes the new construction in the inspected material, but negative search results do not prove novelty.

The closest checked sources include Hurault–Moreau–Peyre arXiv:2605.08392v2 (7 October 2026), Horvat–Pfister (ICLR 2024), Huang et al. (ICML 2025), Beyler–Bach arXiv:2508.03210v2, Parker's convexity criteria, and Banerjee–Lee–Moosmuller arXiv:2610.19878v1 (14 October 2026). Standard transport and numerical-analysis references remain.

The revision is formatted with the standard IEEEtran class and includes an impact statement and disclosure identifying AI-assisted portions and roles. The official current IEEE general AI-content policy was verified. The TAI author-information page repeatedly returned HTTP 418; current supplement limits, exact impact-statement constraints, and submission-file rules therefore remain unverified. Historical 2023 page/fee instructions are not presented as current rules. The main article is nine pages, but formatting alone does not certify submission readiness or acceptance.

The original released citation files and archival metadata continue to identify version 1.0.0. This work is an unreleased revision, with no newly invented DOI, publication date, or claim of author validation. Before any journal submission the author must verify the new results and current journal requirements. No approval or submission action was undertaken on the author's behalf.
