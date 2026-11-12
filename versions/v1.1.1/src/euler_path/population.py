"""Evaluate population formulas with arbitrary precision."""

import mpmath as mp


def number(value):
    return mp.mpf(str(value))


def text(value):
    return mp.nstr(value, 50)


def radial(nodes, alpha, kappa, second_moment):
    times = [number(t) for t in nodes]
    alpha, kappa, second_moment = map(number, (alpha, kappa, second_moment))
    horizon = times[-1]
    action = second_moment * (
        kappa**2 * horizon + (alpha**2 - 1) * kappa * mp.atan(kappa * horizon)
    )
    particle_map = mp.mpc(1)
    rows = []
    for k, t in enumerate(times):
        if k:
            previous = times[k - 1]
            step = t - previous
            denominator = 1 + kappa**2 * previous**2
            particle_map *= 1 + step * (kappa**2 * previous + 1j * alpha * kappa) / denominator
        target_scale = mp.sqrt(1 + kappa**2 * t**2)
        exact_map = target_scale * mp.exp(1j * alpha * mp.atan(kappa * t))
        exact_zero = k == 0 or alpha == 1
        w2 = mp.mpf(0) if exact_zero else mp.sqrt(second_moment) * abs(target_scale - abs(particle_map))
        particle_rms = mp.mpf(0) if exact_zero else mp.sqrt(second_moment) * abs(particle_map - exact_map)
        rows.append({"node": k, "time": text(t), "scale": text(abs(particle_map)),
                     "target_scale": text(target_scale), "w2": text(w2), "w2_squared": text(w2**2),
                     "particle_rms": text(particle_rms), "algebraic_zero": exact_zero})
    return {"action": text(action), "nodes": rows}


def gaussian(nodes, rate, rotation, covariance_trace, mean_speed_squared):
    times = [number(t) for t in nodes]
    rate, rotation, covariance_trace, mean_speed_squared = map(
        number, (rate, rotation, covariance_trace, mean_speed_squared)
    )
    horizon = times[-1]
    integral = horizon if rate == 0 else mp.expm1(2 * rate * horizon) / (2 * rate)
    action = mean_speed_squared * horizon + (rate**2 + rotation**2) * covariance_trace * integral
    particle_map = mp.mpc(1)
    rows = []
    for k, t in enumerate(times):
        if k:
            particle_map *= 1 + (t - times[k - 1]) * (rate + 1j * rotation)
        target_scale = mp.exp(rate * t)
        scale = abs(particle_map)
        exact_map = mp.exp((rate + 1j * rotation) * t)
        w2 = mp.sqrt(covariance_trace) * abs(target_scale - scale)
        particle_rms = mp.sqrt(covariance_trace) * abs(particle_map - exact_map)
        rows.append({"node": k, "time": text(t), "covariance_factor": text(scale**2),
                     "scale": text(scale), "target_scale": text(target_scale),
                     "w2": text(w2), "w2_squared": text(w2**2),
                     "particle_rms": text(particle_rms), "algebraic_zero": k == 0})
    return {"action": text(action), "nodes": rows}


def calibration(rate_step):
    z = number(rate_step)
    delta = mp.expm1(2 * z) - 2 * z - z**2
    return delta


def negative_root():
    return mp.findroot(lambda z: mp.exp(z) + 1 + z, (-1.4, -1.1))


def asymptotic_constants(rate, horizon, covariance_trace):
    a, t, trace = map(number, (rate, horizon, covariance_trace))
    scale = mp.sqrt(trace) * mp.exp(a * t)
    weighted_integral = mp.quad(lambda s: mp.exp(2 * a * s) * s**2, [0, t])
    return {"base_terminal_w2_over_h": text(scale * a**2 * t / 2),
            "corrected_terminal_w2_over_h_squared": text(scale * 2 * a**3 * t / 3),
            "base_sum_squared_over_h": text(trace * a**4 * weighted_integral / 4),
            "corrected_sum_squared_over_h_cubed": text(trace * 4 * a**6 * weighted_integral / 9)}
