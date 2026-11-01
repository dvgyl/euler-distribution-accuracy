# Strengthened revision: global criteria and nonlinear obstructions

Date: 1 November 2026. This report documents the second research revision of Dev D. Goyal's *When Euler Is Second-Order Accurate in Distribution: Global Criteria and Nonlinear Obstructions*. It supersedes the first revision's contribution claims and verification counts. The implementation map for the original 20 Markdown requests and the detailed nonradial experiments remain in [the first revision report](research/first_revision_report.md).

## 1. What changed most

The paper now has two central results: an exact global error characterization for recursive Euler, and an obstruction covering all globally regular nonlinear generators of a broad class of odd-dimensional density paths. The earlier construction, certification, feasibility and learned-path results are retained with complete proofs. The main article gives compact summaries; the supplement carries the complete mathematical development.

These are analytical results with separate proof cross-checks and reproducible numerical examples. They are not proof-assistant certificates, independent human peer review, or an exhaustive claim of priority. The author has not been represented as having already approved or independently verified this new material.

### A material correction to the literature assessment

The follow-up search found two very close papers omitted by the first search:

- Obzherin and Khrulkov, [*Density-Level Backward Error for Low-NFE Generative Flows*, arXiv:2610.29475v1](https://arxiv.org/abs/2610.29475v1), dated 21 October 2026 in the primary PDF.
- Obzherin and Khrulkov, [*Finite-Step Geometry of Marginal-Preserving Velocity Fields in Generative Flows*, arXiv:2610.37936v1](https://arxiv.org/abs/2610.37936v1), dated 27 October 2026 in the primary PDF.

The local projected-acceleration coefficient, generic gauge objective, propagated-defect mechanism, and sufficient order-elevation conditions must not be presented as newly discovered here. The revised introduction, certification summary, related work, bibliography and audit make that attribution explicit. The second paper also already applies the classical hairy-ball theorem to tangential fields with deterministic shell radius and a positive shell gap. The topological theorem and that restricted application are credited.

The new nonlinear theorem removes the tangency and deterministic-shell restrictions, allows redistribution across density level sets, and covers arbitrary finite nondegenerate Gaussian mixtures. The new global theorem gives an exact full-Wasserstein coefficient under the stated Euclidean second-moment hypotheses and proves converses. [The follow-up audit](research/followup_literature_audit.md) records the primary versions and the specific theorem sections inspected. A negative search is not a proof of novelty.

## 2. Exact global error and a necessary-and-sufficient criterion

**Sources:** `paper/revision_global.tex`, `paper/revision_global_proofs.tex`; development and assumption record in `research/global_asymptotic_candidates.md`.

Write `a = partial_t u + (Du)u`, let `F_{s,t}` be the exact flow, and let `Pi_p` project onto the closure of smooth compactly supported gradients in `L2(p)`. Transport acceleration to the observation time:

`A_{s,t} = (DF_{s,t} a_s) composed with F_{t,s}`.

For refining grids with normalized step profiles `sigma_n(s)=h_{n,k}/eta_n` converging weak-star to `sigma`, with maximum step `H_n=O(eta_n)`, the actual recursively computed Euler laws satisfy, uniformly in physical time,

`W2(p_hat_t^n,p_t)/eta_n -> (1/2) ||Pi_{p_t} integral_0^t sigma(s) A_{s,t} ds||`.

This is a signed vector integral followed by projection and a norm. Summing local residual norms gives an upper bound but loses the cancellation responsible for the exact global coefficient.

The assumptions are those of the original finite-time theorem with the acceleration-preservation condition removed: a `C2` field, uniform global spatial Lipschitz bounds for velocity and material acceleration, bounded velocity at the origin, a finite-second-moment source, and uniform `L2` acceleration and material-jerk bounds. The proof adds no positive-density, fourth-moment, Poincare, or regular projected-flow assumption.

For one fixed field, the following are equivalent:

1. Material acceleration preserves the instantaneous law at every time.
2. The maximum-node distribution error is `O(H^2)` on every finite grid.
3. Along even one refining sequence of uniform grids, the maximum-node error is `o(h)`.
4. Terminal error is `o(1/N)` for every positive smooth allocation density defining a quantile grid.

An arbitrarily small `C1` neighborhood of uniform allocation suffices in item 4. If acceleration preservation fails, the maximum-node error is exactly `c_u h + o(h)` for a strictly positive field-dependent constant. This establishes a sharp first-order alternative, not merely failure of a second-order upper bound. Constants need not be uniform over fields that change with grid resolution or training.

## 3. Nonlinear rigidity in odd dimension

**Sources:** `paper/revision_parity.tex`, `paper/revision_parity_proofs.tex`; independent derivation in `research/nonlinear_parity_audit.md`.

Let the dimension be odd. Consider a prescribed location-scale path

`p_t = Law(m(t) + g(t) Z)`, with `g>0`,

where the centered source has a positive smooth density `rho proportional to exp(-V)`. Outside a ball, suppose

`z dot grad V(z) >= c |z|^2`, and `D2 V(z) <= M I`.

Only an upper Hessian bound is needed; global log-concavity is not required. Under the smoothness conditions in the theorem, there is an acceleration-preserving generator in the original globally regular class **if and only if both the location and scale are affine in time**. In that case the qualifying generator is unique:

`u(t,x) = m'(t) + (g'(t)/g(t)) (x-m(t))`.

It has zero material acceleration and Euler transports its trajectories exactly on every grid.

### Why this includes arbitrary finite Gaussian mixtures

For posterior mixture weights `w_j`, precision matrices `Q_j`, and component means `mu_j`,

`grad V = sum_j w_j Q_j (z-mu_j)`,

`D2 V = sum_j w_j Q_j - Cov_w(Q_j(z-mu_j)) <= Lambda I`.

Also `z dot grad V >= lambda |z|^2 - B |z|`, with `lambda>0`. Thus every finite mixture with positive weights and positive-definite component covariances meets the tail assumptions after centering. Components may have unequal means and covariances, and the mixture may be multimodal and nonradial.

### The mathematical mechanism

After removing the necessarily affine mean, set `r=u-(g'/g)x` and `kappa=g''/g`. Weighted divergence and the acceleration condition give the pointwise stress identity

`tr((Dr)^2) + r^T D2 V_t r = kappa (x dot grad V_t - d)`.

Here the trace is that of the matrix square, not the squared Frobenius norm. Large density level sets are smooth radial graphs over an even-dimensional sphere. The hairy-ball theorem gives a point where the tangential projection of `r` vanishes. At that point weighted divergence bounds the remaining normal field by `d Lip(r)/|grad V_t|`. The left side stays bounded while the right side diverges if `kappa>0`. The second-moment energy identity excludes `kappa<0` and forces `r=0` when `kappa=0`.

Combining this result with the global characterization gives a numerical-order consequence: on a nonaffine odd-dimensional mixture location-scale path, every fixed admissible generator has first-order maximum-node distribution error. If both location and scale are affine, the unique acceleration-preserving generator is Euler-exact; every other fixed admissible generator has first-order maximum-node error. A specially chosen endpoint may still exhibit cancellation.

## 4. Stronger consequences and their scope

### All convex-observable tests can be insufficient

For every centrally symmetric source with the required moments, positive convex scalar dilation passes every convex-observable and Hessian-weighted minimum-energy test in the paper, as well as the covariance semidefinite test. Nevertheless, the odd-dimensional Gaussian path `N(0,exp(2t) I)` is infeasible by the nonlinear theorem.

This separates feasibility from the entire specified test hierarchy. It is stronger than the earlier fourth-moment counterexample, which separated feasibility only from second-moment and covariance tests. It does not claim to rule out every conceivable convex formulation involving other variables or constraints.

### A residual floor at fixed regularity budget

For the isotropic Gaussian at a time with `g''>0`, write `kappa=g''/g`, `L=Lip(r)`, and `A_d=d+sqrt(d^2+d-2)`. The canonical projected-acceleration residual obeys the explicit conservative bound

`epsilon_t >= g''/(2 sqrt(2 pi)) exp(-[1+max(1,2 A_d L/sqrt(kappa))]^2/2) > 0`.

Half this quantity lower-bounds the exact local `h^2` Wasserstein coefficient. A fixed finite Lipschitz budget therefore cannot meet a vanishing `O(H)` residual tolerance. Approaching zero residual requires a growing regularity budget; the explicit lower bound implies a necessary growth of order at least `sqrt(log(1/epsilon))`.

For the broader tail class, a compactness and lower-semicontinuity argument gives a positive, nonexplicit residual gap at each fixed finite Lipschitz budget. It uses a weak-Lipschitz extension of the stress obstruction. No bound uniform over unbounded Lipschitz budgets is claimed, and a local residual floor is not identified with a terminal global error coefficient.

### What an escape from global Lipschitzness would require

In the odd-dimensional Gaussian case, any hypothetical smooth exact correction outside the globally Lipschitz class must have a Jacobian growing at least linearly on remote spheres, with an explicit bound in the supplement. Every polynomial exact correction has degree at most one, because the highest-degree sum of squares in `|r|^2` cannot be canceled by the lower-degree derivative term. It then reduces to a skew linear field satisfying `K^2=-kappa I`, which is impossible in odd dimension for `kappa>0`.

These are necessary restrictions, not existence claims for nonpolynomial fields with growing derivatives.

## 5. Why finitely many endpoint tests can fail

For any prescribed finite collection of positive smooth allocation densities, finitely many integral constraints leave a nonzero smooth function `f` with small amplitude. The two-dimensional field

`u_t(x) = (a I + sqrt(a^2-f(t)) J) x`

generates the same expanding Gaussian path as the acceleration-preserving control, while its projected acceleration is `f(t)x`. The constraints make terminal error `O(N^-2)` on every selected schedule despite first-order maximum-node error. The construction can be arbitrarily close to the valid field in any prescribed finite smoothness norm. Finitely many observation times can be hidden by adding finitely many constraints.

For the explicit choice `f(t)=epsilon cos(2 pi t)`, uniform-grid endpoint error even has the **same second-order leading coefficient** as the valid control:

`N^2 W2(p_hat_1,p_1) -> 2 sqrt(2) exp(a) a^3 / 3`.

At time one quarter, error is first order. Changing allocation to `rho_delta(t)=1+delta cos(2 pi t)` exposes a nonzero first-order endpoint coefficient for every nonzero admissible `delta`. This proves a limitation of a fixed finite family of asymptotic-order observations; it is not a statement that ordinary finite data can establish an exact asymptotic theorem.

## 6. Numerical validation

**Files:** `src/euler_path/global_validation.py`, `scripts/run_global_validation.py`, `tests/test_global_validation.py`, `results/global_validation/`; methods in `research/global_validation_notes.md`.

The new example uses `a=0.4`, `epsilon=0.08`, `delta=0.2`, with 16 through 4096 steps. Errors use the full exact Gaussian Wasserstein formula evaluated through stable logarithmic scale accumulation.

| Field and schedule | Endpoint order | Quarter-time order | Maximum-node order |
| --- | ---: | ---: | ---: |
| Cosine defect, uniform | 1.999795 | 1.001410 | 0.996874 |
| Cosine defect, perturbed quantile | 0.995523 | 1.001357 | 0.997689 |
| Acceleration-preserving, uniform | 1.999789 | 1.999789 | 1.999789 |
| Acceleration-preserving, perturbed quantile | 1.999775 | 1.999811 | 1.999775 |

Orders use the final doubling from 2048 to 4096 steps. At 4096 steps, the uniform cosine endpoint error is `5.36463254e-9` while its maximum-node error is `2.96663228e-6`. Under perturbed allocation, its endpoint error is `2.11769611e-6`; the valid control's endpoint remains `5.70331240e-9`.

Independent 2-by-2 covariance recurrences agree within `1.05e-13` in covariance entries. Twelve independent 100-digit mpmath evaluations agree with the floating-point Wasserstein errors within `1.45e-12` relative error. These are arithmetic checks, not formal interval enclosures. The proofs establish the asymptotic rates; the tests verify implementation.

The earlier nonradial experiments, global-versus-local correction search, path checks, noisy restricted regression and actual-cost comparisons are retained. Their numerical Wasserstein bounds are quadrature evaluations of proved continuum inequalities. The unfavorable equal-NFE results remain visible: Heun slightly outperforms corrected Euler on the nonradial example and RK3 is substantially better. No runtime or generative-model sampling speedup is claimed.

## 7. Verification and packaging

- All **80 tests passed**: the previous 72 plus 8 global-validation tests. The whole-suite receipt is `results/global_validation/full_test_receipt.txt`.
- Separate mathematical derivations checked the global expansion, uniform tangent argument, both converses, finite-schedule construction, nonlinear stress identity, level-set topology, mixture assumptions, quantitative floor, weak-Lipschitz compactness extension and polynomial/Jacobian consequences. No remaining gap was identified in those checks; that is not an assurance against undiscovered errors.
- The complete established numerical reproduction passed during the first revision, with its rounded manuscript macros reproduced byte-for-byte. It was not needlessly rerun for this follow-up; the unchanged experiments and prior receipts remain available.
- The title, abstract, introduction, related work, discussion, AI disclosure, README and reproduction instructions now reflect the stronger results and corrected prior-work comparison.
- Full proofs are retained. Roman appendix numbering avoids the 26-appendix limit; distinct hyperlink destinations preserve cross-references in the combined manuscript.
- The synchronized complete, main and supplement PDFs are generated by `make paper`. The combined PDF preserves internal cross-links; standalone subsets retain printed references and omit cross-document hyperlink annotations.
- The final build contains 8 main pages and 54 supplement pages (62 combined). It has no unresolved references or citations, overfull boxes, multiply defined labels or duplicate PDF-destination warnings. The main article and the new supplementary proof pages were rendered for visual inspection. Source files, data, tests and figures are included in the source archive. Downloaded third-party literature PDFs and temporary build files are excluded.

## 8. Remaining limits and submission status

The core order dichotomy concerns one fixed field, with the exact regularity hypotheses in the theorems. The mixture impossibility is for odd dimension, a fixed-shape scalar location-scale path, and the globally regular generator class. It is not an impossibility theorem for arbitrary learned transport paths, even dimensions, all numerical solvers, or fields with unrestricted derivative growth.

The result is a theoretical strengthening with exact-model validation. There is no neural mixture-transport benchmark, image benchmark, general correction-training convergence theorem, or practical speedup claim. The retained learned-field experiment is deliberately a one-coefficient noisy regression model.

Current TAI-specific submission limits remain unverified because the official author-information page could not be accessed during the first revision. Standard IEEEtran formatting and an AI-assistance disclosure do not certify submission readiness. The author should assess the mathematics, contribution framing and current journal requirements before submission. No journal submission, merge to the main branch, release or new DOI is asserted by this revision. Original version-1.0.0 citation metadata is retained as historical release metadata.
