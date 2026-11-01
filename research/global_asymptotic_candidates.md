# Sharp global Euler asymptotics: proved results and prior-work boundary

Status: full proofs are in paper/revision_global_proofs.tex; compact statements
are in paper/revision_global.tex. The feasibility-theory agent independently
audited the complete argument. This note records the mathematical decisions,
comparison with the closest primary sources, and limits of the conclusions.

## 1. Exact global theorem

For a fixed velocity field, define

\[
a_t=\partial_tu_t+Du_tu_t,\qquad
A_{s,t}=(DF_{s,t}\,a_s)\circ F_{t,s},\qquad
B_t^\sigma=\int_0^t\sigma(s)A_{s,t}\,ds.
\]

For grids with scale \(\eta_n\downarrow0\), maximum step
\(H_n=O(\eta_n)\), and piecewise constant normalized step profile
\(\sigma_n(s)=h_{n,k}/\eta_n\rightharpoonup^*\sigma\) in \(L^\infty\),
actual recursive Euler obeys

\[
\sup_{t\in[0,T]}
\left|
\frac{W_2(\widehat p_t^n,p_t)}{\eta_n}
-\frac12\|\Pi_{p_t}B_t^\sigma\|_{L^2(p_t)}
\right|\longrightarrow0.
\]

The numerical interpolation is the usual piecewise linear Euler trajectory.
Its node values are the actual recursive laws, not independent steps
restarted from each exact marginal. In particular,

\[
\lim_n\frac{\max_kW_2(\widehat p_k^n,p_{t_{n,k}})}{\eta_n}
=\frac12\max_t\|\Pi_{p_t}B_t^\sigma\|.
\]

Transported acceleration must be multiplied by the derivative of the flow.
Composition with the inverse flow alone would give the wrong coefficient.

### Assumptions

These are the original manuscript's assumptions, with acceleration
preservation omitted:

- \(p_0\in\mathcal P_2(\mathbb R^d)\).
- \(u\in C^2\) on a neighborhood of the closed time interval, globally
  spatially Lipschitz with a common constant, and \(u_t(0)\) uniformly bounded.
- \(a_t\) is globally spatially Lipschitz with a common constant.
- Acceleration and material jerk have uniformly bounded \(L^2(p_t)\) norms.

No density, lower density bound, Poincaré inequality, fourth moment,
regular projected field, or differentiable velocity-to-density map is
assumed. A different trained field at every resolution needs extra uniformity.

## 2. Proof details that prevent hidden assumptions

### Strong error expansion

With \(X_t=F_{0,t}\), \(J_t=DF_{0,t}\), and
\(c_t=J_t^{-1}a_t(X_t)\), the expansion is

\[
\left\|\sup_t\left|
\frac{\widehat X_t^n-X_t}{\eta_n}
+\frac12J_t\int_0^t\sigma(s)c_s\,ds
\right|\right\|_{L^2(p_0)}\longrightarrow0.
\]

The pointwise Euler error is at most \(CH_n(1+|z|)\). For each initial
point, uniform continuity of \(Du\) on its compact trajectory tube makes
the nonlinear remainder negligible. Its dominating function is linear in
\(|z|\), so a fourth moment is unnecessary. The jerk remainder is bounded by
\(C_j\sum h_k^3/\eta_n=O(H_n^2/\eta_n)=o(1)\).

Weak-star convergence of the step profiles suffices: for each initial point
it gives convergence of integrals against the continuous function \(c_s(z)\).
Equi-Lipschitz primitives make the convergence uniform in the upper time
limit, and a linear growth bound permits \(L^2\) dominated convergence.
There is no derivative bound on the step profile.

### Continuity of the coefficient

Let \(S_0=\mathcal G_{p_0}^\perp\) and \(C_t=\int_0^t\sigma c\).
Pushforward by the flow maps \(S_0\) bijectively to the solenoidal space
at time \(t\), so

\[
\|\Pi_{p_t}B_t^\sigma\|
=\inf_{z\in S_0}\|J_t(C_t-z)\|_{L^2(p_0)}.
\]

The multiplication operators \(J_t,J_t^{-1}\) are uniformly bounded.
The identity \(\partial_tJ_t=Du_t(X_t)J_t\) makes \(J_t\) Lipschitz in
operator norm, and \(C_t\) is Lipschitz in \(L^2\). Distances in these
uniformly equivalent Hilbert norms are therefore Lipschitz in time.
No continuity of the projected vector field is needed.

### Uniform tangent expansion

For \(b_t=-B_t^\sigma/2\), local spatial \(C^1\) bounds and linear
growth follow from \(u\in C^2\), \(a\in C^1\). Smooth spatial cutoffs
give globally Lipschitz fields. The existing frozen-flow projection lemma
gives their uniform upper tangent bound; uniform second-moment tails
remove the cutoff.

For the lower bound, smooth scalar tests approximate the projection's dual
norm. Continuity in time and compactness reduce these to finitely many
tests. Optimal-coupling Taylor expansion gives the matching uniform lower
bound. The tangent coefficient is proved, not assumed through a regular
Wasserstein chart.

## 3. Characterization and sharp order alternative

For each fixed field the following are equivalent:

1. \(\operatorname{div}(p_ta_t)=0\) at every time.
2. Every finite grid has maximum distribution error \(O(H^2)\).
3. Along one uniform refinement sequence, maximum error over nodes is \(o(h)\).
4. Every positive smooth quantile allocation density has terminal error
   \(o(1/N)\).

For item 4, all densities in any arbitrarily small \(C^1\) neighborhood
of the uniform density suffice.

The all-node converse pulls a vanishing integrated solenoidal defect back
to the reference measure, then differentiates its primitive in the fixed
closed solenoidal subspace. The schedule converse tests a continuous
Hilbert-valued defect against every smooth scalar time weight.

Thus a fixed field has either maximum error \(O(h^2)\), or

\[
\max_kW_2(\widehat p_k,p_{kh})=c_uh+o(h),\qquad c_u>0.
\]

No intermediate asymptotic order strictly between one and two is possible
for this maximum error under these hypotheses. The second-order branch
may be exact or higher order. The positive constant is field-dependent;
it need not be bounded away from zero over a changing family of fields.

Combining the theorem with the separate odd-dimensional location–scale
rigidity theorem yields first-order maximum error for every fixed
generator whenever the mean or scale is nonaffine. The rigidity theorem
covers Gaussian and finite nondegenerate Gaussian-mixture location–scale
paths. If both mean and scale are affine, its unique preserving generator
is Euler-exact; every other fixed generator has first-order maximum error.

## 4. Endpoint cancellation and finite deterministic validation

For a positive smooth density \(\rho\) with integral one, choose quantile
nodes \(\int_0^{t_k}\rho=k/N\). Then

\[
\lim_N N W_2(\widehat p_T^{N,\rho},p_T)
=\frac12\left\|\int_0^T\frac{g_s^T}{\rho(s)}\,ds\right\|,\qquad
g_s^T=\Pi_{p_T}A_{s,T}.
\]

The signed Hilbert-space integral lies inside the norm. This is different
from summing nonnegative local magnitudes. Any nonzero defect that cancels
at the terminal time for one schedule becomes detectable after an
arbitrarily small smooth schedule perturbation. This applies near any
positive base density, not only the uniform one.

Nevertheless, **any finite predetermined set of schedules can be fooled
simultaneously**, even on a path with a valid preserving generator.
On \(p_t=\mathcal N(0,e^{2at}I_2)\), set

\[
u_t(x)=(aI+b(t)J)x,\qquad b(t)=\sqrt{a^2-f(t)}.
\]

Its projected acceleration is \(f(t)x\). For prescribed
\(\rho_1,\ldots,\rho_m\), choose a nonzero smooth \(f\), with
\(\|f\|_\infty<a^2\), satisfying \(\int f/\rho_i=0\) for every \(i\).
An \(m\)-equation homogeneous system in \(m+1\) disjoint bump coefficients
provides such a function. Every tested endpoint error is actually
\(O(N^{-2})\), while the maximum over uniform-grid nodes is
\(\Theta(N^{-1})\).

The stronger \(O(N^{-2})\) claim follows from the exact Gaussian scalar
recurrence and a quadrature remainder bound; it does not promote a zero
first-order coefficient to a second-order rate without justification.
Finitely many intermediate observation times can also be hidden by adding
their integral constraints. This does not cover dense observations whose
number grows with refinement, and it is not a statistical power guarantee.

## 5. Cosine example: the endpoint coefficient also agrees

Take \(f(t)=\epsilon\cos(2\pi t)\), \(0<\epsilon<a^2\), \(T=1\).
For uniform steps the standard deviation is

\[
r_N=\prod_{k=0}^{N-1}
\sqrt{1+2a/N+(2a^2-f(k/N))/N^2},
\]

and

\[
\log(r_N/e^a)
=-\frac{2a^3}{3N^2}
+\frac{a^4-\epsilon^2/8}{N^3}+O(N^{-4}).
\]

Thus the non-preserving field has the same terminal second-order coefficient
as \(a(I+J)x\):

\[
N^2W_2(\widehat p_1,p_1)\longrightarrow\frac{2\sqrt2e^aa^3}{3}.
\]

Its entire first-order profile is

\[
NW_2(\widehat p_t,p_t)\longrightarrow
\frac{\sqrt2e^{at}\epsilon}{4\pi}|\sin(2\pi t)|,
\]

uniformly in time. At \(t=1/4\), with \(4\mid N\), this is positive.
For \(\rho_\delta=1+\delta\cos(2\pi t)\), \(0<|\delta|<1\),

\[
NW_2(\widehat p_1^{N,\rho_\delta},p_1)\longrightarrow
\frac{\sqrt2e^a\epsilon}{2|\delta|}
\left((1-\delta^2)^{-1/2}-1\right)>0.
\]

For \(a=0.4,\epsilon=0.08,\delta=0.2\), the analytic constants are:

| Quantity | Limit |
|---|---:|
| Uniform endpoint \(N^2W_2\) | 0.09001637205706355 |
| Quarter-time uniform \(NW_2\) | 0.009950034096858342 |
| Perturbed endpoint \(NW_2\) | 0.008700951365974504 |

The experiment agent implements scalar recurrence checks, independent matrix
covariance checks, and high-precision verification in dedicated source and
results files. The theorem does not depend on those computations.

## 6. Closest primary literature and the boundary of the claim

### Density-Level Backward Error for Low-NFE Generative Flows

Ewsey Obzherin and Valentin Khrulkov,
<https://arxiv.org/abs/2610.29475>, v1 October 21, 2026.
Primary PDF and extracted text: build/literature/2610.29475v1.pdf and .txt.

Theorem 2 already states the exact local projected Wasserstein coefficient,
including Euler's factor \(1/2\); Appendix H.1 assumes a regular Wasserstein
chart. Theorem 3, Appendix H.2 and Appendix L.1 give a Duhamel density first
variation and propagated norm upper bound. The solution map is assumed
differentiable into a chosen density trajectory space. Twice differentiability
and regular step-profile variation support a second-order remainder.
The schedule is explicitly optimal for a surrogate, not for the full
nonlinear terminal error.

Our integrated transported vector is the vector-field form of the same
classical Duhamel mechanism. That mechanism and the local projection
coefficient must not be advertised as new. The exact global full-\(W_2\)
coefficient for recursive Euler, uniform-in-time conclusion under explicit
Euclidean \(P_2\) assumptions, weak-star grid profiles, and all-node/all-profile
converses are not supplied by those cited statements.

### Finite-Step Geometry of Marginal-Preserving Velocity Fields in Generative Flows

Ewsey Obzherin and Valentin Khrulkov,
<https://arxiv.org/abs/2610.37936>, v1 October 27, 2026.
Primary PDF and extracted text: build/literature/2610.37936v1.pdf and .txt.

Appendix A.3, Theorem 7 and Corollary 8 establish sufficient projected
modified-field cancellations for higher local and global order on smooth
positive torus densities under stated modified-equation bounds. The paper
explicitly allows further terminal cancellation. Its tangent lemma concerns
smooth positive torus densities and negative-norm perturbations.
It does not state the exact global coefficient, the converse from one
uniform sequence's maximum node error, or the equivalence involving all
positive allocation profiles.

These distinctions follow from reading the primary theorems and proofs,
not from an exhaustive proof of bibliographic novelty. Classical ODE
variational error expansions and Wasserstein tangent geometry are not new.
The manuscript should emphasize the rigorous characterization, the explicit
failure of finite endpoint validation, and its combination with the nonlinear
rigidity obstruction proved separately.

## 7. Audit and limitations

The independent audit checked the global strong expansion, every \(L^2\)
dominating function, uniform tangent bounds, coefficient continuity,
weak-star passage, both converses, finite-schedule quadrature remainder,
and all cosine coefficients and the maximizing time. No mathematical gap
was found. A LaTeX delimiter typo, an accidental control character, and
the exposition of general-base schedule perturbations were corrected.

The theorem supplies no universal lower constant over all generators,
finite-\(N\) confidence intervals from samples, algorithm for globally
optimizing the exact signed coefficient, or preserving construction for an
arbitrary prescribed path. The finite-schedule example is linear in space
with a time-dependent rotation. It disproves an inference from endpoint
measurements; it is not the manuscript's non-Gaussian construction.
