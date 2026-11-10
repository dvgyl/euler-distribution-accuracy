"""Independent numerical checks of the analytical claims in the article.

Every check is a separate implementation that does not import the package in
src/. The script writes results/verification/claims.json and exits with a
nonzero status if a check fails.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_scalar_population as scalar  # noqa: E402  (mixture quantiles and quadrature)

OUT = Path(__file__).resolve().parents[1] / 'results' / 'verification'
checks, failures = {}, []


def record(name, value, expected=None, tol=None):
    entry = {'value': value}
    if expected is not None:
        entry.update(expected=expected, tolerance=tol,
                     ok=bool(np.all(np.abs(np.asarray(value) - np.asarray(expected)) <= tol)))
        if not entry['ok']:
            failures.append(name)
    checks[name] = entry


def g(t, sigma=1.0):
    return np.sqrt((1 - t) ** 2 + sigma ** 2 * t ** 2)


def dg(t, sigma=1.0):
    return (-(1 - t) + sigma ** 2 * t) / g(t, sigma)


def euler_scale(coef, steps):
    """Scale factors of Euler for u = coef(t) x at all nodes of a uniform grid."""
    t = np.arange(steps) / steps
    return np.cumprod(1 + coef(t) / steps)


# 1. Introductory example: one step of 1/2 is exact, two steps of 1/4 are not.
secant = lambda h: (lambda t: (g(t + h) - g(t)) / (h * g(t)))
record('intro_target_sd_at_half', float(g(0.5)), 0.7071, 5e-5)
record('intro_one_step_sd', float(euler_scale(secant(0.5), 2)[0]), float(g(0.5)), 1e-14)
record('intro_two_step_sd', float(euler_scale(secant(0.5), 4)[1]), 0.8536, 5e-5)

# 2. Sharp scalar constant: N (E_c + E_r) -> (1/2)(1 - 1/m)(1 + pi/2) for the secant field.
limit = lambda m: 0.5 * (1 - 1 / m) * (1 + math.pi / 2)
record('integral_g_minus_4', quad(lambda s: g(s) ** -4, 0, 1)[0], 1 + math.pi / 2, 1e-10)
t_dense = np.linspace(0, 1, 2001)
gI = np.array([g(t) * quad(lambda s: g(s) ** -4, 0, t)[0] for t in t_dense])
record('g_times_I_is_increasing', int(np.all(np.diff(gI) > 0)), 1, 0)
sharp = {}
for m in (2, 3, 4):
    rows = []
    for n in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        nodes = np.arange(1, n + 1) / n
        e_c = np.max(np.abs(euler_scale(secant(1 / n), n) - g(nodes)))
        e_r = np.max(np.abs(euler_scale(secant(1 / n), m * n)[m - 1::m] - g(nodes)))
        rows.append({'N': n, 'E_c': float(e_c), 'E_r': float(e_r), 'N_times_sum': float(n * (e_c + e_r))})
    sharp[m] = rows
    record(f'sharp_constant_m{m}', rows[-1]['N_times_sum'], limit(m), 2e-3)
checks['sharp_constant_rows'] = sharp
record('bisection_constant', limit(2), (2 + math.pi) / 8, 1e-15)

# 3. The exact generator c = g'/g has scaled errors 1/2 and 1/4 of (1 + pi/2).
exact_rows = []
for n in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
    nodes = np.arange(1, n + 1) / n
    coef = lambda t: dg(t) / g(t)
    e_c = np.max(np.abs(euler_scale(coef, n) - g(nodes)))
    e_f = np.max(np.abs(euler_scale(coef, 2 * n)[1::2] - g(nodes)))
    exact_rows.append({'N': n, 'E_c': float(e_c), 'E_f': float(e_f), 'N_times_sum': float(n * (e_c + e_f))})
checks['exact_generator_rows'] = exact_rows
record('exact_generator_scaled_sum', exact_rows[-1]['N_times_sum'], 0.75 * (1 + math.pi / 2), 3e-3)

# 4. Blend family: leading coefficients (theta - 1)/2 and (theta - 1/2)/2 times (1 + pi/2).
n = 4096
nodes = np.arange(1, n + 1) / n
for theta in (0.5, 0.75, 1.0):
    coef = lambda t, th=theta: (1 - th) * dg(t) / g(t) + th * secant(1 / n)(t)
    e_c = n * np.max(np.abs(euler_scale(coef, n) - g(nodes)))
    e_f = n * np.max(np.abs(euler_scale(coef, 2 * n)[1::2] - g(nodes)))
    record(f'blend_theta_{theta}', [float(e_c), float(e_f)],
           [abs(theta - 1) / 2 * (1 + math.pi / 2), abs(theta - 0.5) / 2 * (1 + math.pi / 2)], 3e-3)

# 5. Even dimension: the skew field has straight trajectories, so Euler is exact.
sigma = 0.7
J = np.array([[0.0, -1.0], [1.0, 0.0]])
M = lambda t: (1 - t) * np.eye(2) + sigma * t * J
x = np.eye(2)
for k in range(5):
    t = k / 5
    x = x + 0.2 * (sigma * J - np.eye(2)) @ np.linalg.inv(M(t)) @ x
record('even_skew_euler_exact', float(np.max(np.abs(x - M(1.0)))), 0.0, 1e-14)
record('even_skew_covariance', float(np.max(np.abs(M(0.6) @ M(0.6).T - g(0.6, sigma) ** 2 * np.eye(2)))), 0.0, 1e-14)

# 6. Three-dimensional skew example: exact generator, a_0 = diag(-1/2, -1/2, 1) x, Delta_0 = 3.
J12 = np.zeros((3, 3))
J12[0, 1], J12[1, 0] = -1.0, 1.0
A = lambda t: dg(t) / g(t) * np.eye(3) + math.sqrt(1.5) / g(t) ** 2 * J12
eps = 1e-6
accel0 = (A(eps) - A(0.0)) / eps + A(0.0) @ A(0.0)
record('skew3_acceleration', accel0.tolist(), np.diag([-0.5, -0.5, 1.0]).tolist(), 1e-5)
record('skew3_generates_path', float(np.max(np.abs(A(0.3) + A(0.3).T - 2 * dg(0.3) / g(0.3) * np.eye(3)))), 0.0, 1e-14)
record('skew3_initial_excess', float(np.trace((A(0.0) + np.eye(3)).T @ (A(0.0) + np.eye(3)))), 3.0, 1e-12)

# 7. Endpoint example u = -5 t^2 x: terminal error is second order, path error is first order.
rows = []
for n in (16, 32, 64, 128, 256):
    scale = euler_scale(lambda t: -5 * t ** 2, n)
    nodes = np.arange(1, n + 1) / n
    rows.append({'N': n, 'terminal': float(abs(scale[-1] - math.exp(-5 / 3))),
                 'path': float(np.max(np.abs(scale - np.exp(-5 * nodes ** 3 / 3))))})
checks['endpoint_rows'] = rows
record('endpoint_terminal_order', math.log2(rows[-2]['terminal'] / rows[-1]['terminal']), 2.0, 0.05)
record('endpoint_path_order', math.log2(rows[-2]['path'] / rows[-1]['path']), 1.0, 0.05)
record('endpoint_integral', quad(lambda s: -10 * s + 25 * s ** 4, 0, 1)[0], 0.0, 1e-12)

# 8. Oscillating acceleration: full norm against the stated closed form.
for k in (2, 5):
    norm = math.sqrt(quad(lambda y: math.sin(k * y) ** 2 * math.exp(-y * y / 2) / math.sqrt(2 * math.pi),
                          -12, 12, limit=400)[0]) / k ** 2
    record(f'oscillating_norm_n{k}', norm, math.sqrt((1 - math.exp(-2 * k * k)) / 2) / k ** 2, 1e-9)

# 9. Scalar mixture: source acceleration of the conditional velocity equals (mu^2 + sigma^2) x.
import torch  # noqa: E402
xs = torch.linspace(-3, 3, 13, dtype=torch.float64, requires_grad=True)
h = 1e-5
v0 = scalar.exact_field(0.0, xs)
dv_dx = torch.autograd.grad(v0.sum(), xs)[0]
dv_dt = (scalar.exact_field(h, xs) - scalar.exact_field(0.0, xs)) / h
a0 = (dv_dt + v0 * dv_dx).detach().numpy()
record('mixture_source_acceleration', float(np.max(np.abs(a0 - (scalar.MU ** 2 + scalar.SIGMA ** 2) * xs.detach().numpy()))), 0.0, 1e-3)

# 10. Three-dimensional mixture: population W2 errors for both exact generators.
z, w = scalar.quadrature()
x0 = torch.tensor(z)
published = {8: (0.1932745, 0.0992033, 0.1511015, 0.0768981), 16: (0.0992033, 0.0503485, 0.0768981, 0.0388608),
             32: (0.0503485, 0.0253755, 0.0388608, 0.0195436), 64: (0.0253755, 0.0127400, 0.0195436, 0.0098015),
             128: (0.0127400, 0.0063833, 0.0098015, 0.0049083), 256: (0.0063833, 0.0031950, 0.0049083, 0.0024561)}
gs = lambda t: g(t, scalar.SIGMA)
plane_coef = lambda t: dg(t, scalar.SIGMA) / gs(t)
mixture_rows = []
with torch.no_grad():
    runs = {n: [s.numpy() for s in scalar.euler_states(scalar.exact_field, x0, n)] for n in (8, 16, 32, 64, 128, 256, 512)}
for n in (8, 16, 32, 64, 128, 256):
    nodes = np.arange(1, n + 1) / n
    mix_c = np.array([scalar.w2_to_target(runs[n][k], t, z, w)[0] for k, t in enumerate(nodes)])
    mix_f = np.array([scalar.w2_to_target(runs[2 * n][2 * k + 1], t, z, w)[0] for k, t in enumerate(nodes)])
    plane_c = np.abs(euler_scale(plane_coef, n) - gs(nodes))
    plane_f = np.abs(euler_scale(plane_coef, 2 * n)[1::2] - gs(nodes))
    row = {'N': n,
           'conditional_E_c': float(np.max(np.sqrt(mix_c ** 2 + 2 * plane_c ** 2))),
           'conditional_E_f': float(np.max(np.sqrt(mix_f ** 2 + 2 * plane_f ** 2))),
           'skew_E_c': float(np.max(mix_c)), 'skew_E_f': float(np.max(mix_f))}
    mixture_rows.append(row)
    record(f'mixture_table_N{n}', [row['conditional_E_c'], row['conditional_E_f'], row['skew_E_c'], row['skew_E_f']],
           list(published[n]), 6e-8)
checks['mixture_rows'] = mixture_rows
sums = [(r['conditional_E_c'] + r['conditional_E_f'], r['skew_E_c'] + r['skew_E_f']) for r in mixture_rows]
record('mixture_order_conditional', math.log2(sums[-2][0] / sums[-1][0]), 0.997, 1e-3)
record('mixture_order_skew', math.log2(sums[-2][1] / sums[-1][1]), 0.998, 1e-3)

# 11. A generator with nonzero acceleration that preserves each distribution.
#     u = tanh(t) x + J x in dimension two generates N(0, cosh(t)^2 I); a_t = 2 tanh(t) J x.
def rotation_errors(n):
    def run(steps):
        m, out = np.eye(2), []
        for k in range(steps):
            t = k / steps
            m = (np.eye(2) + (np.tanh(t) * np.eye(2) + J) / steps) @ m
            out.append(np.linalg.svd(m, compute_uv=False))
        return np.array(out)
    nodes = np.arange(1, n + 1) / n
    dist = lambda sv: np.sqrt(np.sum((sv - np.cosh(nodes)[:, None]) ** 2, axis=1))
    return float(np.max(dist(run(n)))), float(np.max(dist(run(2 * n)[1::2])))


t0 = 0.4
A_rot = lambda t: np.tanh(t) * np.eye(2) + J
accel = (A_rot(t0 + 1e-6) - A_rot(t0 - 1e-6)) / 2e-6 + A_rot(t0) @ A_rot(t0)
record('rotation_acceleration', accel.tolist(), (2 * np.tanh(t0) * J).tolist(), 1e-8)
rotation_rows = []
for n in (8, 16, 32, 64, 128, 256, 512, 1024):
    e_c, e_f = rotation_errors(n)
    rotation_rows.append({'N': n, 'E_c': e_c, 'E_f': e_f, 'N2_times_sum': n * n * (e_c + e_f)})
checks['rotation_rows'] = rotation_rows
record('rotation_second_order', math.log2((rotation_rows[-2]['E_c'] + rotation_rows[-2]['E_f'])
                                           / (rotation_rows[-1]['E_c'] + rotation_rows[-1]['E_f'])), 2.0, 0.01)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'claims.json').write_text(json.dumps(checks, indent=1) + '\n')
passed = sum(1 for c in checks.values() if isinstance(c, dict) and c.get('ok'))
print(f'{passed} checks passed, {len(failures)} failed', failures)
sys.exit(1 if failures else 0)
