# Targeted literature and submission-format audit

**Follow-up notice:** the subsequent search identified two materially closer October 2026 papers that the first audit missed. Their local projected-defect and restricted parity results are now explicitly credited. See research/followup_literature_audit.md for the corrected comparison and the scope of the new global and nonlinear results. The historical first-audit conclusions below must be read subject to that correction.

Audit date: 1 November 2026. Scope: the new reference-coordinate construction, projected-acceleration Euler error, and IEEE TAI submission preparation. This is a targeted literature check, not a proof of priority or an exhaustive novelty certification. No manuscript or numerical source was changed during this audit.

## Findings that affect scientific framing

### 1. A recent related paper must be distinguished

**Amartya Banerjee, Harlin Lee, and Caroline Moosmüller, _Neural Wasserstein Splines: Representation-Invariant Second-Order Priors for Trajectory Inference_, arXiv:2610.19878v1, 14 October 2026.**

- Primary paper: <https://arxiv.org/abs/2610.19878v1>
- Primary PDF: <https://arxiv.org/pdf/2610.19878v1>
- Authors' implementation: <https://github.com/amartya21/Neural-Wasserstein-Splines>
- Actually read: downloaded v1 PDF, title/version stamp, Section 2, Sections 3.2–3.3 including equations (8)–(13), Proposition 1, and the adjacent theory discussion. The authors' README was also read. Full proof appendices and benchmarks were not independently audited.

The method projects generator velocity onto a gradient dictionary and differentiates that time-dependent projection to approximate intrinsic Wasserstein acceleration. The moving Gram matrix matters. It explicitly distinguishes this quantity from the projection of raw particle acceleration. For a fixed dictionary, its population penalty is representation invariant. This is close conceptual context but a different objective: our projected material acceleration is a defect of the chosen sampling field and can change when that field is changed on the same marginal path. No harmonic construction or second-order full-Wasserstein Euler theorem was found in the inspected material. A full-text search found Euler only in a benchmark protocol, which is useful screening evidence but not an exhaustive theorem audit.

**Recommended framing:** retain the phrase “projected material-acceleration residual.” Do not call it the intrinsic acceleration of the probability curve. Cite this paper when distinguishing the two notions.

### 2. Harmonic-coordinate fluid mechanics is an adjacent, different use of “harmonic”

**Olivia Constantin and María J. Martín, _A harmonic maps approach to fluid flows_, Mathematische Annalen 369, 1–16 (2017); online 4 June 2016. DOI: 10.1007/s00208-016-1435-9.**

- Primary publisher full text: <https://link.springer.com/article/10.1007/s00208-016-1435-9>
- Actually read: abstract, introduction, and governing-equation section, including the Lagrangian Jacobian constraint and pressure-gradient acceleration.

This classifies incompressible fluid Euler solutions with spatially harmonic Lagrangian maps. The present radial-envelope Fourier construction does not impose spatial harmonicity or the incompressible pressure equations. The shared use of reference coordinates does not establish the same result. Nevertheless, “harmonic map” without qualification is misleading here: `z² exp(-|z|²)` is not a harmonic function of the spatial label. “Angular harmonics,” “Fourier-mode construction,” or “reference-coordinate construction” accurately describe the proposed family.

### 3. The sharp local coefficient should remain a specialization of established geometry

The manuscript already attributes the tangent-space gradient projection to Ambrosio–Gigli–Savaré. Its local coefficient

`lim W2(Euler#p, exact#p)/h² = ||Π_p a||/2`

uses the Wasserstein metric derivative/tangent-norm principle applied to Euler's second-order defect. This is a useful exact identification in the paper's setting; it should not be framed as inventing tangent projection, Helmholtz decomposition, or a new notion of intrinsic acceleration. The main manuscript currently makes this qualification correctly. The primal/dual squared-gap certificate is also a Hilbert-space orthogonality identity, not independently a novel projection theorem.

**Gabriel Peyré, _Straighter flows, smaller Euler errors_, 16 April 2026.** Primary author exposition: <https://www.gpeyre.com/blog/2026/04/16/straight-flows-and-euler-error/>. The complete displayed theorem and proof were read. It gives the standard particle-error bound through material acceleration and Lipschitz stability. It does not remove marginal-preserving acceleration by changing the coupling. This supports the distinction between ordinary particle-error analysis and the new use of projection/coupling; it is not a substitute for the closest research-paper comparison.

### 4. Existing team comparisons and what this extra audit did not redo

The main revision separately inspected Hurault–Moreau–Peyré, _Geometry-Aware Discretization Error of Diffusion Models_, arXiv:2605.08392v2 (7 October 2026), including its weak/Fréchet and Gaussian-scale-mixture discussions; Horvat–Pfister (2024) on gauge freedom; Huang et al. (2025) on divergence alignment; Beyler–Bach arXiv:2508.03210v2; and Parker (2024), Definition 3.1. This extra audit did not repeat those complete inspections. Primary anchors are retained in the manuscript bibliography. Parker's publisher text was additionally located at <https://londmathsoc.onlinelibrary.wiley.com/doi/full/10.1112/blms.13030>.

Targeted searches combining Euler, distribution accuracy, weighted/projected acceleration, Wasserstein local error, and harmonic/reference-coordinate constructions did not identify a prior result that clearly subsumes the new construction and its full-Wasserstein Euler guarantee. Negative search results do not establish originality. The defensible contribution is the explicit construction plus its proved regularity, restricted global correction search, quantitative error theory, and stronger concrete feasibility examples—not the generic existence of gauge freedom, gradients, projections, or convex moment identities.

## IEEE TAI: verified requirements versus unresolved current details

### Official current IEEE AI-disclosure policy: verified

Primary IEEE Journal Author Center page, retrieved in full on the audit date:

<https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/>

Its author policy requires an acknowledgment disclosure for generated content, identifying the AI system, affected article sections, and the level of its use. Pure editing/grammar assistance is treated separately. The present revision involved mathematical development, writing, and code generation, so its disclosure should describe those roles specifically. The policy page also addresses prior publication and concurrent submissions. It does not supply journal-specific TAI page or impact-statement limits.

### TAI current author page: could not be verified

<https://cis.ieee.org/publications/ieee-transactions-on-artificial-intelligence/information-for-authors-tai>

Repeated direct retrieval returned HTTP 418. Official TAI call-for-papers search records point to this same page and the ScholarOne portal, but do not independently establish present length/format details. No current numerical limit should be claimed from those snippets.

### Historical primary evidence: useful but not current certification

1. A TAI editorial acceptance letter dated **20 September 2023**, publicly retained by the recipient's institution, was read in full: <https://yoksis.bilkent.edu.tr/pdf/files/87753_L.pdf>. It states ten regular-paper pages before overlength charges, a fifteen-page regular-paper maximum with charges, and exclusion of supplementary material from overlength counting. These are **2023 instructions**, not verified 2026 rules. No present fee estimate is endorsed here.
2. A guest editor's TAI special-issue call with **2022 deadlines** states that normal TAI instructions include an impact statement: <https://www.kpsuba.com/activities/ieee-tai-special-issue>. This confirms historical practice, not a current word limit. Search hits for other journals' impact-statement limits were rejected as irrelevant.

The prepared nine-page IEEEtran main paper and separate supplement are a reasonable submission-style layout choice given this evidence, but this audit cannot certify compliance with every current TAI limit. In particular, the complete supplement's admissibility, the exact impact-statement word limit, and current submission-file rules remain unverified. A concise impact statement is sensible; its inclusion should not be described as verified compliance with an inaccessible current rule.

## Disposition

No close result found in this limited audit invalidates the paper's carefully delimited construction/accuracy framing. The material addition is the October 2026 Neural Wasserstein Splines comparison and the explicit distinction between intrinsic curve acceleration and a chosen field's projected acceleration. Current TAI formatting details retain the specific verification limitation above. No claims of absolute novelty, comprehensive literature coverage, formal proof verification, or guaranteed journal suitability are supported by this audit.
