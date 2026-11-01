# Revision experiment audit

## Delivered

- `src/euler_path/revision_experiments.py`: exact harmonic-map field, contraction inverse, all-root quartic discovery, residual certificates, recursive Euler/Heun/RK3, Gaussian obstruction, one-coefficient learned-path floor.
- `scripts/run_revision.py`: deterministic regeneration, including all CSVs and five figures.
- `tests/test_revision.py`: nine independent tests (second differences of exact trajectories, exact moments, inverse recovery, objective checks, convergence, obstruction, NFE, learned-path reference).
- `paper/revision_experiments.tex`: concise main empirical section.
- `paper/revision_experiments_details.tex`: complete numerical assumptions and receipts.
- `results/revision/`: raw CSVs, summary, generated table, five PDFs, full test log, validation receipt.

## Main results

The global search yields omega=0.3999999999999999 and c=0.31999999999999995; the exact coefficient theorem identifies the solution as (a,2a^2). The evaluated residual upper is 2.07e-16, a floating-point number, not a rigorous enclosure. A bad local minimum is at omega=-0.392394, residual upper0.0724104.

At N512 corrected Euler has W2 lower/upper estimates3.438502e-7 /4.880645e-7. Their finest orders are1.99831 /1.99915. Particle RMS6.60856e-4 has order0.99925. Base Euler lower/upper are about3.30e-4 with first order. All distribution metrics are explicitly lower/upper bounds, evaluated by deterministic quadrature; no exact full-W2 or interval arithmetic claim.

At equal384 NFE, corrected Euler lower6.110505e-7 exceeds Heun upper6.085053e-7; given refinement accuracy this is convincing numerical evidence that Heun is slightly better, not an interval-certified proof. RK3 upper1.070191e-9 is substantially better. All stages include inversion cost; no speedup claim is made.

Continuous-time path preservation is analytic, with independent DOP853 trajectory checks within6.5e-12 at intermediate times on five labels. Quadrature refinement64x64 to96x128 alters corrected upper by less than8e-17.

The Gaussian path diag(1+t,1+t+2t^2) passes the scalar kinetic-energy budget, but the first coordinate forces an all-field residual lower1/[4(1+t)^(3/2)]. The affine projection minimum is computed globally by its even quartic. It is not asserted to equal the all-field minimum.

The learned experiment is one noisy regression coefficient, explicitly not a neural benchmark. Three seeds give nonzero continuous-path errors around6.7e-4 to5.4e-3 while numerical-to-learned-path errors fall below6e-6.

## Audit

All63 original tests passed; complete original reproduction succeeded. Original rounded numerical macros match byte-for-byte. Unrounded numerical summaries differ at roundoff-sensitive scales (order differences about4e-8 at the finest nonlinear mesh), with no failed validation thresholds. Final suite:72 tests pass (63 old plus9 new).

## Limits

- Finite-dimensional correction search uses a known analytic map and a very small parameter family; this does not solve arbitrary learned flows.
- No exact optimal-transport solve is used in the nonradial example. Valid coupling and moment bounds are computed, and quadrature convergence is tested.
- Reported residual upper uses an explicit solenoidal field, not necessarily the optimal gradient projection.
- Runtime is a CPU vectorized implementation measurement, with noise and all three repetitions retained; no claim about neural inference efficiency or peak-memory advantage.
- Test/experiment verification is numerical, not Lean/formal proof.
