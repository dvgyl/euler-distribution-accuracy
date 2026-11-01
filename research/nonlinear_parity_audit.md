# Independent audit: all-generator rigidity in odd dimensions

Date: 1 November 2026. This note records an independent derivation and audit of the new obstruction, including its extension from Gaussian sources to arbitrary finite nondegenerate Gaussian mixtures. It is a research note, not an assertion that the literature search establishes absolute novelty.

## Conclusion

The odd-dimensional obstruction is valid for **all smooth globally Lipschitz generators**, including nonlinear generators, of a fixed-shape dilation path. It extends to every finite mixture of nondegenerate Gaussians with arbitrary component means and covariance matrices. The mixture need not be radial, unimodal, or logconcave.

For a centered source density `rho`, a positive scalar scale `g(t)`, and an affine mean `m(t)`, consider

\[
p_t(x)=g(t)^{-d}\rho((x-m(t))/g(t)).
\]

If `d` is odd and the source obeys the tail conditions below, a smooth globally Lipschitz generator can satisfy

\[
\operatorname{div}(p_t a_t)=0,
\qquad a_t=\partial_tu_t+D u_tu_t,
\]

throughout a time interval **if and only if** `g` is affine on that interval. In the feasible case the generator is unique:

\[
u_t(x)=m'(t)+\frac{g'(t)}{g(t)}(x-m(t)),
\]

its material acceleration is zero, and Euler transports this path exactly on every grid contained in the interval. If the mean is not initially assumed affine, acceleration preservation and the first moment force it to be affine.

This conclusion concerns a *prescribed entire dilation path*. It does not rule out a different path joining the same endpoints, and it does not claim infeasibility for arbitrary paths in odd dimensions.

## 1. Exact instantaneous PDE identity

Work in coordinates centered at the affine mean, fixing a time, and write

\[
\alpha=g'/g,\qquad \kappa=g''/g,\qquad r=u-\alpha x,
\qquad p=e^{-V}.
\]

The continuity equation gives `div(pr)=0`. Differentiating the continuity equation along a smooth flow gives

\[
\partial_{tt}p
=\operatorname{div}\operatorname{div}(puu^\top)
-\operatorname{div}(pa).
\]

For the dilation path, with `D p = div(px)`,

\[
\partial_tp=-\alpha Dp,
\qquad
\partial_{tt}p=\alpha^2D^2p-\alpha'Dp
=\alpha^2\operatorname{div}\operatorname{div}(pxx^\top)
-\kappa\operatorname{div}(px).
\]

The sign and coefficient in the final term matter: `div div(pxx^T)=D^2p+Dp`, so `alpha'+alpha^2=kappa`. The mixed terms vanish exactly, since

\[
\operatorname{div}\operatorname{div}(pxr^\top)
=\operatorname{div}(pr)=0,
\]

and the transpose has the same double divergence. Thus acceleration preservation implies

\[
\boxed{\operatorname{div}\operatorname{div}(prr^\top)
=-\kappa\operatorname{div}(px).}
\tag{1}
\]

Moreover `div(pr)=0` is equivalent to `div r=grad V . r`, and

\[
\operatorname{div}\operatorname{div}(prr^\top)
=\operatorname{div}(p Dr\,r).
\]

Differentiating `div r=grad V . r`, the terms involving `grad V . Dr r` cancel, leaving

\[
\boxed{\operatorname{tr}((Dr)^2)+r^\top(D^2V)r
=\kappa(x\cdot\nabla V-d).}
\tag{2}
\]

Here `tr((Dr)^2)` is **not** the Frobenius norm squared. It can be negative; the proof uses only the valid upper bound

\[
\operatorname{tr}((Dr)^2)\le d\|Dr\|_{\rm op}^2.
\]

This is the key identity that removes the restriction to linear or exactly tangential gauges.

## 2. General tail criterion and topological argument

It suffices that a positive smooth density `p=e^{-V}` have all sufficiently high level sets of `V` smooth compact hypersurfaces diffeomorphic to `S^{d-1}`, and that, uniformly along those levels,

\[
x\cdot\nabla V\longrightarrow\infty,
\qquad |\nabla V|\longrightarrow\infty,
\qquad D^2V\preceq M I
\tag{3}
\]

for a finite constant `M`. The Hessian assumption is one-sided. An unbounded negative eigenvalue causes no problem. A weaker sufficient condition is a uniform upper bound on the positive part of `lambda_max(D^2V)/|grad V|^2` on those levels.

Let `L=sup_x ||Dr(x)||op<infinity`. On a high level set the tangential projection

\[
r_T=r-\frac{r\cdot\nabla V}{|\nabla V|^2}\nabla V
\]

is a continuous tangent vector field. Because `d` is odd, the level set has even dimension and the hairy-ball theorem supplies a point where `r_T=0`. (For `d=1`, a level set has zero-dimensional tangent spaces and the same conclusion is immediate.) At this point `r` is normal to the level set and weighted solenoidality yields

\[
|r|=\frac{|\operatorname{div}r|}{|\nabla V|}
\le\frac{dL}{|\nabla V|}.
\]

Consequently, at that point the left side of (2) is at most

\[
dL^2+\frac{M d^2L^2}{|\nabla V|^2},
\]

which stays bounded. For `kappa>0`, the right side diverges to positive infinity along high levels. This is impossible.

The argument does not assume that `r` is exactly tangent to any density contour. It derives an asymptotically small normal component from the global derivative bound and the continuity equation, then contradicts the acceleration equation.

## 3. Energy identity handles the other scale curvatures

Assuming finite second moment and the paper's growth/integrability hypotheses, testing (1) against `|x|^2`, using cutoffs, gives

\[
\mathbb E_p|r|^2=\kappa\mathbb E_p|X|^2
=gg''\mathbb E_\rho|Z|^2.
\tag{4}
\]

Thus `kappa<0` is impossible in every dimension. If `kappa=0`, then `r=0` almost everywhere, hence everywhere by continuity and positive density. The topological argument excludes the remaining case `kappa>0` in odd dimensions. The conclusion is pointwise in time; if acceleration preservation holds on an interval, `g''=0` throughout that interval.

For Gaussian mixtures all cutoffs are harmless because the tails are Gaussian and the Lipschitz field grows at most linearly. A centered source and an affine translation allow the same computation in moving coordinates. Acceleration preservation separately gives `m''=0` by testing the first moment.

## 4. Every finite nondegenerate Gaussian mixture satisfies the tail criterion

Let

\[
p(x)=\sum_{j=1}^K\pi_j\mathcal N(x;\mu_j,\Sigma_j),
\qquad \pi_j>0,\quad\Sigma_j\succ0,
\]

and define posterior component weights `w_j(x)`, precision matrices `H_j=Sigma_j^{-1}`, and scores `s_j=H_j(x-mu_j)`. Then

\[
\nabla V=\sum_jw_js_j,
\qquad
D^2V=\sum_jw_jH_j-\operatorname{Cov}_{w}(s_j).
\]

Set

\[
\lambda=\min_j\lambda_{\min}(H_j)>0,
\quad\Lambda=\max_j\lambda_{\max}(H_j),
\quad B=\max_j|H_j\mu_j|.
\]

Then, for every `x`,

\[
D^2V\preceq\Lambda I,
\qquad
x\cdot\nabla V\ge\lambda|x|^2-B|x|,
\qquad
|\nabla V|\ge\lambda|x|-B.
\tag{5}
\]

The last inequality is useful only when its right side is positive, which is sufficient in the tails. Covariance differences can make the Hessian's negative eigenvalues unbounded, but this does not invalidate the one-sided bound needed by the proof.

Choose `R0>B/lambda` and `s0=max_{|x|<=R0}V(x)`. For `s>s0`, every ray `x=r e`, `|e|=1`, meets `{V=s}` exactly once: it starts below level `s` at `r=R0`, tends to infinity, and

\[
\partial_rV(re)=e\cdot\nabla V(re)\ge\lambda r-B>0
\]

for `r>=R0`. The implicit function theorem makes the radial crossing a smooth positive function of `e`. Thus `{V=s}` is a smooth radial graph, hence diffeomorphic to `S^{d-1}`. Its minimum radius tends to infinity as `s` tends to infinity because `V` is bounded on every fixed compact ball. Equations (5) verify the remaining limits in (3).

This proof allows arbitrary component means and covariance matrices. In particular, it includes multimodal and nonradial mixtures. The density can always be centered before applying the scale-path theorem.

The same calculation also covers finite mixtures of densities `exp(-V_j)` whose component potentials satisfy common two-sided Hessian bounds `lambda I <= D^2V_j <= Lambda I`: replace `B` by `max_j|grad V_j(0)|`. This extension is not needed to state the Gaussian-mixture corollary.

## 5. Quantitative growth obstruction in the Gaussian special case

For `p=N(0,g^2 I)`, identity (2) becomes

\[
|r|^2+g^2\operatorname{tr}((Dr)^2)
=\kappa|x|^2-dg^2\kappa.
\tag{6}
\]

Even without assuming a global derivative bound, put

\[
M_R=\max_{|x|=R}\|Dr(x)\|_{\rm op}.
\]

At a hairy-ball pole on the radius-`R` sphere, weighted solenoidality gives `|r|<=dg^2 M_R/R`. Substituting into (6) yields, for `R>sqrt(d)g` and `kappa>0`,

\[
\boxed{
M_R^2\ge
\frac{\kappa R^2(R^2-dg^2)}{dg^2(R^2+dg^2)}.
}
\tag{7}
\]

Any smooth nonlinear escape from the odd-dimensional obstruction must therefore have Jacobian norm growing at least linearly along large spheres. This is stronger than merely observing that the affine skew-matrix construction has no solution in odd dimensions. It is not an existence claim for such irregular fields.

## 6. Polynomial rigidity for Gaussian sources

Equation (6) also classifies polynomial gauges without first imposing a global Lipschitz bound. If a nonzero polynomial vector field `r` has degree `m>=2`, then `|r|^2` has degree `2m` with a nonzero leading sum of squares, while `tr((Dr)^2)` has degree at most `2m-2`. These cannot sum to a polynomial of degree at most two. Thus `m<=1`.

Write `r(x)=Kx+b`. The equation `div r=x.r/g^2` then forces `b=0` and `K+K^T=0`. Comparing quadratic coefficients in (6) yields

\[
K^\top K=\kappa I,
\qquad K^2=-\kappa I.
\]

These conditions are also sufficient for the static acceleration-preservation equation. Therefore **every polynomial Gaussian gauge with zero acceleration defect is already an affine rotation gauge**, and positive scale curvature is impossible in odd dimensions even within polynomial classes whose degrees were not bounded beforehand. This argument is independent of the global Lipschitz topological proof.

## 7. Lipschitz closure and a positive residual floor at fixed derivative budget

The obstruction extends to globally Lipschitz `r` satisfying the static equations distributionally; classical second derivatives are not necessary for this closure statement. Since `div r=grad V.r`, the divergence belongs to `W^{1,infinity}_loc`. Mollify `r` on compact sets. For the smooth mollifications,

\[
\operatorname{div}(Dr_\eta r_\eta)
=\operatorname{tr}((Dr_\eta)^2)
+r_\eta\cdot\nabla\operatorname{div}r_\eta.
\]

The mollified first derivatives converge strongly in every finite local `L^q`; the divergence and its first derivatives do likewise because `div r` is locally Lipschitz. The identity therefore passes to distributions, with an `L^q_loc` right side. Weighted solenoidality again cancels the score terms, giving (2) almost everywhere.

The inequalities actually used by the proof are

\[
|\nabla V\cdot r|\le dL,
\qquad
\kappa(x\cdot\nabla V-d)-r^\top D^2V r\le dL^2.
\]

Their left sides are continuous, so their almost-everywhere validity extends to every point. The high-level-set argument thus applies unchanged to a Lipschitz limit.

This yields a useful nonconstructive quantitative corollary. Fix an odd-dimensional density `p` obeying the tail criterion, a positive `kappa`, and a finite Lipschitz budget `L`. Define the weighted negative Sobolev norm by

\[
\|\eta\|_{-1,p}
=\sup_{\varphi\in C_c^\infty,\ \|\nabla\varphi\|_{L^2(p)}\le1}
|\langle\eta,\varphi\rangle|.
\]

Then

\[
\boxed{
\inf_{\substack{\operatorname{div}(pr)=0\\\operatorname{Lip}(r)\le L}}
\left\|\kappa\operatorname{div}(px)
+\operatorname{div}\operatorname{div}(prr^\top)\right\|_{-1,p}
>0.
}
\tag{8}
\]

The norm equals `||Pi_p(kappa x+Dr r)||_{L^2(p)}` when the projection is defined using the closure of compactly supported gradients. For an actual generator of the dilation path, this is also the projected material-acceleration norm, because the weighted divergences agree.

To prove (8), suppose a sequence had defect norm tending to zero. Weighted solenoidality gives `E_p r=0`, hence

\[
|r(0)|\le L\mathbb E_p|X|,
\qquad |r(x)|\le L(|x|+\mathbb E_p|X|).
\]

Arzela-Ascoli supplies a locally uniform subsequence converging to a globally Lipschitz field with the same derivative bound. The common linear envelope and finite second moment give convergence in `L^2(p)`, and the quadratic fluxes converge in `L^1(p)`. The divergence condition and the double-divergence flux identity therefore pass to distributions. A defect norm tending to zero forces the limit defect to vanish on every compact smooth test function. This contradicts the Lipschitz obstruction just proved.

The bound in (8) depends on `p`, `kappa`, and `L`; this argument does not give a computable constant. It gives no lower bound uniform as `L` tends to infinity. Gaussian mixtures meet the finite-moment and tail assumptions used above.

## 8. Independent Gaussian conditional-moment check

The original proof from the feasibility agent gives an independent route for Gaussian sources. For each unit vector `e`, with `Y=e.X`,

\[
\mathbb E[e\cdot r(X)\mid Y=y]=0,
\qquad
\mathbb E[(e\cdot r(X))^2\mid Y=y]=\beta=gg''.
\]

These follow respectively from the one-dimensional marginal continuity equation and its second derivative, using the Gaussian heat identity. Polynomial growth eliminates the affine integration constants after integrating twice in `y`.

At a hairy-ball pole `Re`, `r(Re)` is parallel to `e`. For `Z` the independent perpendicular Gaussian and `X=Re+Z`, weighted solenoidality and the Lipschitz bound imply

\[
R|e\cdot r(Re+Z)|
\le dg^2L+L|Z|^2.
\]

The conditional second moment therefore tends to zero as `R` tends to infinity, uniformly in the pole direction. This forces `beta=0`. Continuity of the conditional moments, which follows from global Lipschitz growth and dominated convergence, justifies evaluating them at the varying pole direction and at the particular conditional value `Y=R`.

This independent route confirms the pointwise identity's Gaussian conclusion and avoids assuming that a nonlinear gauge is pointwise tangent to spheres.

## 9. Relation to the close October 2026 primary sources

The following primary manuscripts were read locally, not inferred from abstracts:

- Obzherin and Khrulkov, *Density-Level Backward Error for Low-NFE Generative Flows*, arXiv:2610.29475v1, 21 October 2026; local `build/literature/2610.29475v1.pdf` and extracted text. Appendix J's Gaussian Lyapunov/Riccati classification and parity result concern **linear Gaussian gauges**. These establish the relevant affine predecessor.
- *Finite-Step Geometry of Marginal-Preserving Velocity Fields*, arXiv:2610.37936v1, 27 October 2026; local `build/literature/2610.37936v1.pdf` and extracted text. Appendix D.2, Theorem 34, already uses the hairy-ball obstruction for **any continuous tangential field with deterministic output radius and a positive shell gap**. Its next paragraph explicitly leaves angle-dependent redistribution possible. The affine fixed-shape classifications elsewhere in that manuscript also do not cover all nonlinear globally Lipschitz generators.

Accordingly, neither “odd-dimensional obstruction” nor “hairy-ball proof” alone is an adequate novelty claim. The additional result here is the PDE argument that rules out *all* globally Lipschitz nonlinear generators for a broad prescribed family, without tangency, shellwise transport, Gaussianity, or radial-symmetry assumptions. The finite-mixture corollary extends the scope beyond the two primary sources' stated classes.

Targeted searches on 1 November 2026 for combinations of Gaussian/mixture scaling, acceleration preservation, odd-dimensional obstruction, weighted divergence, and hairy-ball terminology did not identify an additional exact match. This limited search is not evidence of absolute priority.

## 10. Limits that must remain explicit

1. The theorem requires a global spatial derivative bound. Equation (7) explains its role; the proof does not classify arbitrary smooth fields with unbounded derivatives.
2. It concerns scalar fixed-shape dilations, with affine translations, not every covariance path or every generative path.
3. At a fixed finite Lipschitz budget the compactness argument proves a positive residual floor, but its value is nonconstructive. There is no lower bound here uniform over classes with increasing Lipschitz constants. Tail-based obstructions can have very small probability mass.
4. The result does not say that all odd-dimensional target distributions are unreachable, or that choosing a different interpolating path cannot remove the obstruction.
5. Finite Gaussian mixtures need nondegenerate covariances. Singular measures and compactly supported laws require separate arguments.
6. The even-dimensional theorem is not a converse for arbitrary nonradial mixtures. Even-dimensional radial sources admit the familiar rotation construction, but feasibility for a general mixture is not established here.
7. The proof uses `tr((Dr)^2)`, not a squared Frobenius norm; replacing one with the other would be a substantive error.

