# Endpoint-prescribing reference-coordinate constructions

Status: candidate research notes, 1 November 2026. The rotation construction and its norm bounds below have a direct proof. They have not been integrated into the manuscript or claimed to be a breakthrough. The more general vector-field extension still needs a complete boundary/time-regularity argument.

## Scope caveat

Endpoint reachability by itself is weak: at generator zero, straight displacement interpolation already connects the Gaussian to every sufficiently small diffeomorphic perturbation and makes Euler particle-exact. The potentially useful addition is an explicit, globally regular, **nonzero acceleration-preserving** realization for a broad prescribed endpoint, with first-order particle error and second-order distribution error. This does not solve acceleration preservation for an arbitrary already-prescribed intermediate probability path.

## 1. Rotation-generator endpoint theorem

Let `rho=N(0,I_2)`, identify the plane with the complex line, fix `T>0`, `c>0`, and let `G=Id+g`, where `g` is smooth and compactly supported. Seek

`F_tt = c D F Jz`, `F_0=Id`, `F_T=G`.

Put `lambda=sqrt(i c)` with positive real part and

`q(t)=cosh(lambda(t−T/2))/cosh(lambda T/2)`.

Let `B_s` be standard real Brownian motion started at `t` and stopped at its first exit time `tau` from `(0,T)`. Then define

`Q_t g(z)=E[1_{B_tau=T} g(R_{−c tau/2}z)]`,

`F_t(z)=q(t)z+Q_tg(z)`.

Equivalently, the complete boundary representation is

`E[1_{B_tau=0}R_{−c tau/2}z + 1_{B_tau=T}G(R_{−c tau/2}z)]`.

The joint process has generator `(1/2)partial_tt−(c/2)partial_theta`. Its Dirichlet harmonic extension therefore satisfies the desired reference equation. The formula for `q` is the characteristic transform of the exit time: its ODE is `q''=ic q`, with both boundary values one. This is a use of the standard stopped-process/Dirichlet representation, not a newly discovered probabilistic principle.

### Fourier construction and regularity

For each scalar component write the angular expansion `g(r,theta)=sum_n g_n(r)e^{in theta}`. The correction multiplier is

`m_0(t)=t/T`,

`m_n(t)=sinh(lambda_n t)/sinh(lambda_n T)`, `lambda_n=sqrt(i c n)`, `n!=0`,

choosing positive real part. Negative modes are conjugates of positive modes for real scalar data. The denominator cannot vanish because `Re lambda_n>0`. Thus this handles negative harmonics and the zero mode without an additional constraint. The Brownian formula gives the same multiplier by solving its scalar ODE.

For every finite `j,k`, compact smooth `g` gives angular Fourier coefficients rapidly decaying in the spatial `C^k` norm. Indeed integration by parts in the rotation parameter controls `||g_n||_{C^k}` by `C_{k,m,R}(1+|n|)^{-m}||g||_{C^{k+m}}`, where `R` bounds its support. Time derivatives of the multiplier are bounded uniformly on `[0,T]` by a constant times `(1+|n|)^{j/2}`. Taking `m>j/2+1` gives uniform absolute convergence of each required derivative. This proves smoothness through both physical-time endpoints and through the spatial origin, rather than relying on polar-coordinate smoothness heuristics.

### Sharp simple spatial bound

The hitting probability is `P(B_tau=T)=t/T`. Rotations are Euclidean isometries, so for every integer `k>=0`,

`||D^k Q_tg||_infinity <= (t/T)||D^k g||_infinity`.

All corrections remain supported in the same enclosing ball. Also

`|q(t)| >= m_c := 1/|cosh(lambda T/2)|`.

For the latter, writing `lambda(t−T/2)=x+ix` gives
`|cosh(x+ix)|²=(cosh(2x)+cos(2x))/2>=1`.

Consequently, if `d=||Dg||_infinity < m_c`, then uniformly in time

`|F_t(x)−F_t(y)| >= (m_c−d)|x−y|`.

The upper Lipschitz bound is equally immediate. The map is proper since outside a fixed ball it is multiplication by `q(t)`, and it is locally invertible. Alternatively its inverse is obtained by a contraction around the scalar-linear map. Hence it is a global smooth diffeomorphism, with uniform inverse-derivative bounds.

Since `m_c -> 1` as `c -> 0`, every compact smooth perturbation with `||Dg||<1` permits some strictly positive `c`. The endpoint is exactly `G#rho` and the starting distribution exactly `rho`.

### Acceleration preservation and integrability

Define `u(t,F_tz)=partial_tF_tz`. Then

`a(t,F_tz)=F_tt(z)=c DF_t(z)Jz`.

The reference rotation preserves `rho`; pushing it forward by `F_t` proves zero weighted divergence of the acceleration under `p_t=F_t#rho`. Outside an enclosing ball the field is linear, with bounded smooth coefficients on `[0,T]`. On the remaining compact region uniform inverse bounds and smoothness control all required derivatives. Thus global spatial Lipschitz bounds for velocity and acceleration, and finite Gaussian `L²` bounds for velocity, acceleration, and jerk, follow.

The existing fixed-field theorem yields second-order Euler distribution accuracy. Under the standard modified-flow expansion already used for the finite-mode construction, the leading Euler label drift is `−(c/2)h Jz`, so the terminal common-start error coefficient is

`(c T/2)||DG(Z)JZ||_{L²(rho)} > 0`.

The positivity follows from global inverse Lipschitzness. This gives a genuine first-order particle defect for every such endpoint, although an attained second-order distribution coefficient is not established for all endpoints. Special endpoint symmetries can cancel further terms.

## 2. Extension to a general reference-preserving vector field

Let `b` have a complete smooth flow `Phi_s`, and assume it preserves the reference law. A natural extension is

`F_t(z)=E[1_{B_tau=0}Phi_{−tau/2}(z)+1_{B_tau=T}G(Phi_{−tau/2}(z))]`.

The process generator is `(1/2)partial_tt−(1/2)b·grad_z`, so the same Dirichlet argument gives `F_tt=DF b` whenever the requisite differentiations are justified.

Suppose `||Db||<=L` and `d=||D(G−Id)||`. Flow derivatives give

`||D Phi_{−s/2}||<=exp(Ls/2)`,

`||D Phi_{−s/2}−I||<=exp(Ls/2)−1`.

The Brownian exit transform satisfies

`sup_t E exp(L tau/2)=sec(T sqrt(L)/2)` for `LT²<pi²`.

It follows that

`sup_t ||DF_t−I|| <= (1+d)sec(T sqrt(L)/2)−1`.

Thus `(1+d)sec(T sqrt(L)/2)<2` is an explicit sufficient global-diffeomorphism condition. This is useful for small compactly supported nonlinear Gaussian-preserving stream fields, not only rotations. Stronger derivative estimates require corresponding exponential exit-time moments, available when `b` is scaled sufficiently small.

**Outstanding obligation:** formulate and prove time-boundary regularity for this general transport-generator Dirichlet extension, with enough derivatives to check the full numerical theorem. The rotation case above avoids this gap through its explicit Fourier expansion. Do not present the general-b statement as fully established until this step is completed.

## 3. Novelty and significance status

The mathematical norm argument is elementary once the stopped-process representation is recognized. The appropriate contribution, if distinct from the October 2026 close prior work, would be a broad nongeodesic construction theorem and quantitative regularity bound. It is not arbitrary fixed-path feasibility, and it does not establish practical neural-sampling superiority. A primary-source novelty audit for this endpoint theorem was interrupted in favor of the potentially stronger odd-dimensional all-generator obstruction. No absolute novelty claim is supported by these notes.
