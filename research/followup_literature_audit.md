# Follow-up literature audit: global characterization and nonlinear rigidity

Date: 1 November 2026. This audit supersedes the novelty assessment of the first revision wherever they differ. It records a material omission in the first search and the resulting changes to the paper's contribution claims.

## Newly identified closest prior work

### Obzherin and Khrulkov, Density-Level Backward Error for Low-NFE Generative Flows

- Primary source: <https://arxiv.org/abs/2610.29475v1>.
- Primary PDF: <https://arxiv.org/pdf/2610.29475v1>.
- Version/date verified in the downloaded primary PDF: v1, 21 October 2026.
- Authors: Ewsey Obzherin and Valentin Khrulkov.
- Read: main Sections 3–7; Theorems 1–6; Appendices F, H, J, and L; the Gaussian, radial, and curvature examples in Appendix N. The root and separate proof agents inspected the overlapping theorem statements and proofs directly.

The paper explicitly states the leading local Wasserstein coefficient as half the projected material-acceleration norm for Euler. It also defines the projected gauge objective, proves a discrete-to-continuum result, supplies density-defect propagation bounds, and develops allocation rules. Those generic ideas and the local formula must not be claimed as new in this manuscript. The present Euclidean proof uses explicitly stated Lipschitz/P2 assumptions and avoids a regular projected flow, but that is a regularity-specific treatment of an already identified mechanism.

Theorem 12 gives a parity obstruction **within the linear Gaussian class**, through the eigenvalue pairing of a skew matrix. It does not quantify over arbitrary nonlinear globally Lipschitz gauges.

Theorem 3 and Appendix L give a differentiability-based density first variation and a propagation upper bound. In the inspected text, they do not give the present exact recursive global W2 coefficient under the stated Euclidean P2 hypotheses, the equivalence involving maximum-node superlinear convergence, the converse based on all nearby allocation densities, or the finite-schedule blindness theorem.

### Obzherin and Khrulkov, Finite-Step Geometry of Marginal-Preserving Velocity Fields in Generative Flows

- Primary source: <https://arxiv.org/abs/2610.37936v1>.
- Primary PDF: <https://arxiv.org/pdf/2610.37936v1>.
- Version/date verified in the downloaded primary PDF: v1, 27 October 2026.
- Authors: Ewsey Obzherin and Valentin Khrulkov.
- Read: main framing; Appendix A.1–A.3 (Theorem 4, Theorem 7 and Corollary 8); Appendix B's affine reachability and parity statements; Appendix D.1–D.2 (Theorems 29, 33 and 34, Propositions 30 and 31); scope statements adjacent to those results. This is a targeted theorem comparison, not an audit of all 60 pages or their neural experiments.

This paper gives sufficient projected modified-equation conditions for higher marginal order, approximate-cancellation budgets, affine finite-step reachability, sharp restricted errors, and several nonlinear constructions and obstructions. It already explains terminal cancellation and derives a fixed-field cubic local barrier from the third derivative of the second moment.

Theorem 34 uses the hairy-ball theorem for **continuous tangential fields with deterministic output shell radius and strictly positive shell gap**. Its following discussion explicitly leaves angle-dependent redistribution outside that classification. It therefore does not establish impossibility for every nonlinear weighted-solenoidal correction of a Gaussian mixture.

The revised manuscript must credit both the affine parity result and this nonlinear shellwise use of topology. Neither the hairy-ball theorem nor its applicability to tangent fields on spheres is itself new.

## What the follow-up results add

### 1. A global converse and exact recursive error coefficient

The signed, flow-transported acceleration sum gives the exact first-order W2 coefficient of the recursively computed Euler law, uniformly in physical time. This yields an equivalence between acceleration preservation, all-grid second-order maximum-node accuracy, superlinear maximum-node accuracy on uniform grids, and superlinear terminal accuracy for every positive smooth allocation density. Densities in an arbitrarily small neighborhood of uniform allocation suffice.

The distinction from a propagated upper bound matters: transporting and summing vectors before taking their tangent norm retains cancellation. The finite-schedule construction proves that no preselected finite collection of terminal asymptotic-order tests characterizes the property over the unrestricted smooth field class.

### 2. A nonlinear obstruction beyond affine and shellwise classes

The identity

  tr((Dr)^2) + r^T D²V r = kappa (x · grad V − d)

holds for any acceleration-preserving correction on a dilation path with density proportional to exp(−V), with div(p r)=0. It imposes no tangency assumption on r.

At a zero of the tangential projection of r on a remote density level set, weighted divergence forces r to be small in the normal direction. Global Lipschitzness then bounds the left side, while quadratic tail coercivity makes the right side diverge when kappa>0. This excludes **every globally Lipschitz nonlinear correction** in odd dimension.

All finite nondegenerate Gaussian mixtures satisfy the tail assumptions, with arbitrary component locations and covariances after centering. The result therefore covers multimodal and nonradial mixtures, not only isotropic Gaussians. The proof also provides a fixed-Lipschitz-budget residual gap; the Gaussian specialization gives explicit constants and a derivative-growth barrier outside the bounded-Jacobian class.

### 3. A separation from the entire convex-observable hierarchy

For symmetric sources and convex scalar dilations, every convex-observable and Hessian-weighted minimum-energy test passes. Odd-dimensional Gaussian dilation can nonetheless be infeasible by the new nonlinear theorem. This is a strict separation from the entire test family, not merely another example missed by a finite list of moments.

## Other primary checks

- Tao and Choi, <https://arxiv.org/abs/2605.06680v1>: primary PDF downloaded and the title, abstract, and particle-error framing inspected. It was already cited in the manuscript; no duplicate bibliography entry is retained.
- Hatcher, *Algebraic Topology*, <https://pi.math.cornell.edu/~hatcher/AT/AT.pdf>: the standard degree/hairy-ball argument is credited. The manuscript also states the elementary identity-to-antipodal homotopy so the topological step is transparent.
- Targeted searches combined Gaussian mixtures, odd dimension, nonlinear gauges, weighted divergence, material acceleration, and hairy-ball/topological obstructions. No source clearly subsuming the new all-generator theorem was found. This negative search is not a proof of priority.

## Scientific framing

The first revision's local projection formula should no longer appear as an independent novelty claim. The new paper is centered on the global necessary-and-sufficient characterization, the nonlinear fixed-shape rigidity theorem, quantitative regularity costs, and validation counterexamples. The previously proved reference-coordinate examples remain useful positive constructions.

The primary sources support this scoped comparison. They do not support claims of exhaustive literature coverage, independent human peer review, formal proof verification, guaranteed novelty, journal acceptance, or practical speedup.
