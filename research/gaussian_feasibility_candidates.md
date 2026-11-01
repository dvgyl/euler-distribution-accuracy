# Nonlinear feasibility results and independent proof audit

## Implemented result

The strongest result is no longer restricted to Gaussian covariance paths.
It appears as `thm:nonlinear-odd-rigidity` in
`paper/revision_parity.tex`, with complete proofs in
`paper/revision_parity_proofs.tex`.

Let the ambient dimension be odd. Let the centered source have positive
density proportional to `exp(-V)`, where `V` is `C^3` and, outside a ball,

\[
x\cdot\nabla V(x)\ge c|x|^2,\qquad D^2V(x)\preceq M I.
\]

For the prescribed path `Law(m(t)+g(t)Z)`, with positive `g` and the stated
time regularity, an acceleration-preserving generator in the paper's
globally Lipschitz class exists **if and only if both the mean and scale
are affine in time**. The unique such generator is the affine dilation
field, has zero material acceleration, and Euler transports every step
exactly. The theorem permits arbitrary nonlinear, time-dependent gauges.

Every finite mixture of nondegenerate Gaussians meets these tail
assumptions, with arbitrary component means and covariance matrices after
centering the mixture. Equal covariance, radial symmetry, component
separation, and log-concavity are unnecessary.

## Mechanism and independent checks

For a centered dilation, put `alpha=g'/g`, `kappa=g''/g`, and
`r=u-alpha*x`. Path preservation means `div(p*r)=0`. Acceleration
preservation gives the stress equation

\[
\operatorname{div}\operatorname{div}(p rr^\top)
=-\kappa\operatorname{div}(px).
\]

Writing `p` proportional to `exp(-W)`, this becomes

\[
\operatorname{tr}((Dr)^2)+r^\top D^2W r
=\kappa(x\cdot\nabla W-d).
\]

The matrix trace is the trace of the square, not the squared Frobenius
norm. Its sign is not assumed positive. The mixed double-divergence terms
vanish exactly because `div(p*r)=0`.

Remote level sets of `W` are smooth radial graphs over the sphere. In odd
ambient dimension the hairy-ball theorem gives a point on each level at
which `r` is normal. Weighted divergence and the global Lipschitz bound
then force `|r|` at that point to tend to zero. The left side of the stress
identity is bounded above, while its right side diverges when `kappa>0`.
The energy identity

\[
\|r\|_{L^2(p)}^2=g g''\,\mathbb E|Z|^2
\]

excludes negative `g''` and gives uniqueness at zero. These arguments were
independently checked by the parent and experiments-audit agents.

For Gaussian mixtures with posterior component weights `w_j`, precision
matrices `Q_j`, and scores `s_j=Q_j(x-mu_j)`, the two needed estimates follow
from

\[
\nabla V=\sum_jw_js_j,\qquad
D^2V=\sum_jw_jQ_j-\operatorname{Cov}_w(s_j)\preceq\Lambda I,
\]

and `x dot grad(V) >= lambda*|x|^2-B*|x|`. Only an upper Hessian bound is
used; a potentially large negative Hessian does not invalidate the proof.

## Further implemented consequences

1. **Entire convex hierarchy separation.** Every centrally symmetric
   dilation with a convex positive scale passes every convex-observable
   and Hessian-weighted minimum-energy certificate in the paper, as well
   as the covariance semidefinite test. Odd-dimensional exponential
   Gaussian dilation nevertheless has no admissible preserving generator.
   The obstruction therefore cannot be repaired by adding more convex
   observable tests of this type.

2. **Explicit Gaussian residual floor.** For odd dimension, `g''>0`,
   `kappa=g''/g`, and `L=Lip(r)`, put
   `A_d=d+sqrt(d^2+d-2)`. The canonical projected acceleration norm obeys

   \[
   \varepsilon\ge\frac{g''}{2\sqrt{2\pi}}
   \exp\left[-\frac12\left(1+\max\{1,2A_dL/\sqrt\kappa\}\right)^2\right].
   \]

   Thus a fixed Lipschitz budget cannot achieve arbitrarily small residual,
   and near-cancellation requires at least square-root-logarithmic growth
   of the Lipschitz constant. The sharp local theorem converts half this
   norm into an actual Euler one-step coefficient.

3. **General positive gap.** The globally `L`-Lipschitz,
   weighted-divergence-free class is compact in `L^2(p)`. The acceleration
   defect functional is lower semicontinuous and has a strictly positive
   minimum for the entire tail class above. The proof extends the stress
   identity to Lipschitz limits through local Sobolev product rules.

4. **Jacobian growth outside the Lipschitz class.** Every hypothetical
   smooth Gaussian solution in odd dimension satisfies

   \[
   \sup_{|x|=R}\|Dr(x)\|_{\rm op}
   \ge\frac{\sqrt\kappa R}{g\sqrt d}
   \sqrt{\frac{R^2-dg^2}{R^2+dg^2}},\qquad R>g\sqrt d.
   \]

   A polynomial solution in any dimension is necessarily linear,
   `r(x)=Kx`, with `K` skew and `K^2=-kappa*I`. Consequently no polynomial
   correction evades odd-dimensional impossibility.

5. **Separate radial-score extension.** A uniformly positive, bounded
   radial score coefficient gives the same classification even without
   an upper Hessian bound. The supplement supplies a conditional variance
   and one-dimensional Stein-kernel proof.

## Primary literature comparison

The relevant primary papers were downloaded and checked directly:

- Obzherin and Khrulkov, *Density-Level Backward Error for Low-NFE
  Generative Flows*, [arXiv:2610.29475](https://arxiv.org/abs/2610.29475).
  Appendix J explicitly starts with linear lifts. Its Theorem 12 obtains
  parity from the skew-matrix identity `A^2=-kappa*I`, within that class.
  The local projected defect, modified-field conditions, and linear
  Gaussian parity should be attributed to this prior work.
- Obzherin and Khrulkov, *Finite-Step Geometry of Marginal-Preserving
  Velocity Fields in Generative Flows*,
  [arXiv:2610.37936](https://arxiv.org/abs/2610.37936).
  Theorem 34 proves parity for continuous tangential fields with a
  deterministic shell output radius. It expressly leaves angle-dependent
  redistribution outside that result. Its Appendix D also explains why
  nonlinear weighted-divergence-free fields need not inherit linear
  restrictions on local jets.

The new theorem removes both the linear-gauge and deterministic-shell
restrictions and covers multimodal, nonradial Gaussian mixtures. The
hairy-ball theorem itself is classical; the contribution is the weighted
stress identity and tail argument that force its applicability to an
unconstrained nonlinear generator. The comparison establishes a concrete
advance over these close papers, rather than an exhaustive claim that no
related result exists elsewhere.

## Independent audit of the global error addition

The companion `revision_global.tex` and `revision_global_proofs.tex` were
read independently in full. The following conclusions passed the audit
under exactly the paper's `C^2`, global spatial Lipschitz bounds on velocity
and acceleration, finite second moment, and uniform `L^2` material jerk:

- Exact global Wasserstein coefficient, uniformly in time, for recursive
  Euler and weak-star limiting step profiles.
- Exact maximum-over-nodes coefficient and its continuity.
- Equivalence of acceleration preservation, second order on every grid,
  and superlinear convergence of the full path on uniform grids.
- Equivalence with superlinear terminal convergence for every schedule in
  any small smooth neighborhood of uniform allocation.
- A finite collection of endpoint schedules can all have actual
  `O(N^-2)` errors while acceleration preservation fails.
- The explicit cosine example has the claimed terminal second-order,
  intermediate first-order, and perturbed-schedule first-order constants.

The proof correctly avoids hidden fourth-moment and projection-smoothness
assumptions: it uses a fixed pulled-back solenoidal space for coefficient
continuity; compact truncations plus the frozen-flow lemma for the uniform
tangent upper bound; finitely many dual tests for its lower bound; and
linear pointwise domination for the strong Euler expansion. One LaTeX
typo and the extension of a schedule-perturbation proof sentence from
uniform to arbitrary base schedule were reported and fixed by the author.
