"""Reproducible experiments for the harmonic, nonradial path.

The distribution errors below are explicitly labelled lower/upper bounds.
The upper bound uses a measure-preserving rotation of Gaussian labels; it is
not an optimal-transport computation. Quadrature is checked separately.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import solve_ivp
from scipy.linalg import solve_sylvester
from scipy.optimize import least_squares
from scipy.special import roots_laguerre


@dataclass(frozen=True)
class HarmonicPath:
    a: float = 0.4
    deformation: float = 0.18
    horizon: float = 1.0

    def contraction(self, t):
        return self.deformation * np.exp(self.a * (np.sqrt(2) - 1) * t) * 4 / np.e

    def modes(self, t, z):
        z = np.asarray(z, dtype=complex)
        return (np.exp((1 + 1j) * self.a * t) * z,
                self.deformation * np.exp((1 + 1j) * self.a * np.sqrt(2) * t)
                * z**2 * np.exp(-np.abs(z)**2))

    def map(self, t, z):
        first, second = self.modes(t, z)
        return first + second

    def inverse(self, t, x, tol=2e-14, counters=None):
        """Contraction inversion with an a posteriori absolute error bound."""
        rho = self.contraction(t)
        if not 0 <= rho < 1:
            raise ValueError("The declared global contraction condition fails.")
        first = np.exp((1 + 1j) * self.a * t)
        ratio = self.deformation * np.exp((1 + 1j) * self.a * (np.sqrt(2)-1) * t)
        z = np.asarray(x, dtype=complex) / first
        for iteration in range(1, 101):
            nxt = x / first - ratio * z**2 * np.exp(-np.abs(z)**2)
            error_bound = float(np.max(np.abs(nxt-z))) * rho / (1-rho)
            z = nxt
            if error_bound <= tol:
                if counters is not None:
                    counters["inverse_iterations"] += iteration
                    counters["max_inverse_bound"] = max(counters["max_inverse_bound"], error_bound)
                return z
        raise ArithmeticError("Contraction inverse did not attain its tolerance.")

    def zeta(self, omega):
        modes = np.array([1.0, 2.0])
        return self.a * np.sqrt(modes) + 1j * (self.a*np.sqrt(modes) + modes*(omega-self.a))

    def field(self, t, x, omega, counters=None, inverse_tol=2e-14, delta=0.):
        if counters is not None:
            counters["field_evaluations"] += 1
        z = self.inverse(t, x, tol=inverse_tol, counters=counters)
        first, second = self.modes(t, z)
        coeff = self.zeta(omega)
        return coeff[0]*first + coeff[1]*second + delta*(first+2*(1-np.abs(z)**2)*second)

    def acceleration_labels(self, t, z, omega):
        first, second = self.modes(t, z)
        coeff = self.zeta(omega)**2
        return coeff[0]*first + coeff[1]*second

    def rotation_labels(self, t, z):
        first, second = self.modes(t, z)
        return 1j*first + 2j*second

    def exact(self, t, z, omega, delta=0.):
        return self.map(t, np.exp((delta+1j*(omega-self.a))*t)*z)

    def dilation_labels(self,t,z):
        first,second=self.modes(t,z)
        return first+2*(1-np.abs(z)**2)*second

    def learned_second_moment(self,t,delta):
        scale2=np.exp(2*delta*t)
        return float(2*np.exp(2*self.a*t)*scale2
                     +self.deformation**2*np.exp(2*self.a*np.sqrt(2)*t)
                     *8*scale2**2/(1+4*scale2)**3)

    def mode_masses(self, t):
        return np.array([2*np.exp(2*self.a*t),
                         (8/125)*self.deformation**2*np.exp(2*self.a*np.sqrt(2)*t)])

    def target_second_moment(self, t):
        return float(self.mode_masses(t).sum())

    def residual_vector(self, omega, c, t=None):
        if t is None:
            t = self.horizon
        coeff = self.zeta(omega)**2 - 1j*np.array([1., 2.])*c
        values = np.sqrt(self.mode_masses(t))*coeff
        return np.array([values[0].real, values[0].imag, values[1].real, values[1].imag])

    def residual_upper(self, omega, c, t=None):
        return float(np.linalg.norm(self.residual_vector(omega, c, t)))

    def best_rotation_coefficient(self, omega, t=None):
        if t is None:
            t = self.horizon
        masses = self.mode_masses(t)
        n = np.array([1., 2.])
        return float(np.sum(masses*n*(self.zeta(omega)**2).imag)/np.sum(masses*n*n))


def polar_quadrature(radial=64, angular=64):
    s, radial_weights = roots_laguerre(radial)
    theta = 2*np.pi*np.arange(angular)/angular
    z = (np.sqrt(2*s)[:, None] * np.exp(1j*theta[None, :])).ravel()
    weights = np.repeat(radial_weights/angular, angular)
    return z, weights


def discover_correction(path):
    """Enumerate every stationary point of the coercive quartic objective.

    Eliminating c leaves a quartic in omega. Its cubic derivative has at
    most three real roots. All are evaluated using the original sum of
    squares, avoiding cancellation in quartic evaluation near zero.
    """
    started = time.perf_counter()
    m = path.mode_masses(path.horizon)
    n = np.array([1., 2.])
    omega = Polynomial([0., 1.])
    zeta = [Polynomial([path.a*np.sqrt(j)*(1+1j)-1j*j*path.a, 1j*j]) for j in n]
    square = [v*v for v in zeta]
    cpoly = sum(m[i]*n[i]*Polynomial(square[i].coef.imag) for i in range(2))/float(np.sum(m*n*n))
    objective = Polynomial([0.])
    for i in range(2):
        real = Polynomial(square[i].coef.real)
        imag = Polynomial(square[i].coef.imag)-n[i]*cpoly
        objective += m[i]*(real*real+imag*imag)
    roots = objective.deriv().roots()
    candidates = []
    for root in roots:
        if abs(root.imag) <= 1e-10:
            value = float(root.real)
            c = float(cpoly(value))
            candidates.append({"omega":value,"c":c,
                               "residual_upper":path.residual_upper(value,c),
                               "second_derivative":float(objective.deriv(2)(value))})
    if not candidates:
        raise ArithmeticError("Coercive quartic unexpectedly has no real critical point.")
    best = min(candidates, key=lambda row: row["residual_upper"])
    local = []
    for initial in ([0.07, 0.03], [-0.3, 0.0], [0.8, -0.2]):
        history = []
        def residual(params):
            value = path.residual_vector(*params)
            history.append([float(params[0]), float(params[1]), float(np.linalg.norm(value))])
            return value
        solution = least_squares(residual, initial, ftol=1e-13, xtol=1e-13, gtol=1e-13,
                                 max_nfev=1000)
        local.append({"initial":initial,"omega":float(solution.x[0]),"c":float(solution.x[1]),
                      "residual_upper":float(np.linalg.norm(solution.fun)),"nfev":int(solution.nfev),
                      "success":bool(solution.success),"trace":history})
    return {"objective":"exact L2 residual upper bound squared at T, after eliminating c",
            "quartic_coefficients_low_to_high":objective.coef.tolist(),
            "stationary_points":candidates,"global_minimizer":best,"local_runs":local,
            "elapsed_seconds":time.perf_counter()-started}


def integrate(path, z, steps, method, omega, inverse_tol=2e-14, keep_nodes=False, delta=0.):
    x = path.map(0., z)
    h = path.horizon/steps
    counts = {"field_evaluations":0,"inverse_iterations":0,"max_inverse_bound":0.}
    snapshots = [(0., x.copy())] if keep_nodes else []
    started = time.perf_counter()
    f = lambda t,y: path.field(t,y,omega,counts,inverse_tol,delta)
    for k in range(steps):
        t = k*h
        first = f(t,x)
        if method == "Euler":
            x = x+h*first
        elif method == "Heun":
            second = f(t+h,x+h*first)
            x = x+h*(first+second)/2
        elif method == "RK3":
            second = f(t+h/2,x+h*first/2)
            third = f(t+h,x-h*first+2*h*second)
            x = x+h*(first+4*second+third)/6
        else:
            raise ValueError("Unknown integration method.")
        if keep_nodes and (k+1) in {steps//4,steps//2,3*steps//4,steps}:
            snapshots.append(((k+1)*h,x.copy()))
    counts["runtime_seconds"] = time.perf_counter()-started
    return x, counts, snapshots


def errors(path, numerical, z, weights, omega, steps, c=None, t=None, method="Euler"):
    if t is None:
        t = path.horizon
    h = path.horizon/steps
    exact = path.exact(t,z,omega)
    particle = float(np.sqrt(weights @ np.abs(numerical-exact)**2))
    if method == "Euler" and c is not None:
        rotated = path.map(t,np.exp(1j*((omega-path.a)*t-c*t*h/2))*z)
        upper = float(np.sqrt(weights @ np.abs(numerical-rotated)**2))
        upper_kind = "Gaussian-label rotation coupling"
    else:
        upper, upper_kind = particle, "common-start coupling"
    lower = abs(float(np.sqrt(weights @ np.abs(numerical)**2)) - np.sqrt(path.target_second_moment(t)))
    return {"w2_lower_second_moment":lower,"w2_upper_coupling":upper,
            "particle_rms":particle,"upper_bound_kind":upper_kind}


def gaussian_infeasibility():
    """Directional obstruction missed by the scalar energy budget."""
    records=[]
    for t in np.linspace(0,1,21):
        covariance=np.diag([1+t,1+t+2*t*t])
        first=np.diag([1.,1+4*t])
        second=np.diag([0.,4.])
        base=np.diag(np.diag(first)/(2*np.diag(covariance)))
        root=np.diag(np.sqrt(np.diag(covariance)))
        invroot=np.diag(1/np.sqrt(np.diag(covariance)))
        J=np.array([[0.,-1.],[1.,0.]])
        def residual(omega):
            field=base+omega*root@J@invroot
            normal=second-2*field@covariance@field.T
            projection=solve_sylvester(covariance,covariance,normal)
            return float(np.sqrt(np.trace(projection@covariance@projection)))
        gauge=root@J@invroot
        normal0=second-2*base@covariance@base.T
        normal1=-2*(base@covariance@gauge.T+gauge@covariance@base.T)
        normal2=-2*gauge@covariance@gauge.T
        s0=solve_sylvester(covariance,covariance,normal0)
        s1=solve_sylvester(covariance,covariance,normal1)
        s2=solve_sylvester(covariance,covariance,normal2)
        fourth=float(np.trace(s2@covariance@s2))
        quadratic=float(np.trace(s1@covariance@s1)+2*np.trace(s0@covariance@s2))
        optimum=float(np.sqrt(max(0.,-quadratic/(2*fourth))))
        energy=float(np.trace(base@covariance@base.T))
        lower=1/(4*(1+t)**1.5)
        records.append({"time":float(t),"half_scalar_second_derivative":2.,
                        "minimal_kinetic_energy":energy,"scalar_budget":2-energy,
                        "all_field_residual_lower":lower,"best_affine_omega":optimum,
                        "best_affine_projection_residual":residual(optimum),
                        "base_projection_residual":residual(0)})
    return records


def write_csv(path, rows):
    with Path(path).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def learned_path_experiment(path,output):
    """One fitted coefficient isolates path error from integration error.

    This is a restricted noisy linear regression, not a neural-network
    approximation benchmark. The learned continuous path has an exact
    reference for every fitted coefficient.
    """
    labels,weights=polar_quadrature(64,64)
    rows=[];fits=[]
    for seed in (7,19,42):
        rng=np.random.default_rng(seed)
        sample_count=512;noise_std=.08
        t=rng.uniform(0,1,sample_count)
        z=rng.normal(size=sample_count)+1j*rng.normal(size=sample_count)
        feature=path.dilation_labels(t,z)
        modes=path.modes(t,z);base=path.zeta(0.)[0]*modes[0]+path.zeta(0.)[1]*modes[1]
        response=base+noise_std*(rng.normal(size=sample_count)+1j*rng.normal(size=sample_count))
        started=time.perf_counter()
        delta=float(np.real(np.vdot(feature,response-base))/np.vdot(feature,feature).real)
        fit_time=time.perf_counter()-started
        residual=response-base-delta*feature
        standard_error=noise_std/np.sqrt(np.vdot(feature,feature).real)
        target=path.map(1,labels);learned=path.exact(1,labels,path.a,delta)
        path_upper=float(np.sqrt(weights@np.abs(learned-target)**2))
        path_lower=abs(np.sqrt(path.learned_second_moment(1,delta))-np.sqrt(path.target_second_moment(1)))
        fits.append({"seed":seed,"sample_count":sample_count,"noise_std_per_coordinate":noise_std,
                     "delta_fit":delta,"delta_known_noise_standard_error":float(standard_error),
                     "fit_time_seconds":fit_time,"training_rms_residual":float(np.sqrt(np.mean(np.abs(residual)**2))),
                     "continuous_path_w2_lower":float(path_lower),"continuous_path_w2_upper":path_upper})
        for steps in (16,32,64,128,256,512):
            numerical,counts,_=integrate(path,labels,steps,"Euler",path.a,delta=delta)
            phase=np.exp(-1j*path.a**2/steps)
            exact_phase=path.exact(1,phase*labels,path.a,delta)
            target_phase=path.map(1,phase*labels)
            numerical_rms=np.sqrt(weights@np.abs(numerical)**2)
            rows.append({"seed":seed,"delta_fit":delta,"steps":steps,"h":1/steps,
                         "numerical_to_learned_lower":float(abs(numerical_rms-np.sqrt(path.learned_second_moment(1,delta)))),
                         "numerical_to_learned_upper":float(np.sqrt(weights@np.abs(numerical-exact_phase)**2)),
                         "numerical_to_target_lower":float(abs(numerical_rms-np.sqrt(path.target_second_moment(1)))),
                         "numerical_to_target_upper":float(np.sqrt(weights@np.abs(numerical-target_phase)**2)),
                         "continuous_path_lower":float(path_lower),"continuous_path_upper":path_upper})
    write_csv(output/"learned_coefficients.csv",fits);write_csv(output/"learned_path_floor.csv",rows)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    fig,axes=plt.subplots(1,2,figsize=(6.8,2.65))
    for fit in fits:
        selected=[r for r in rows if r["seed"]==fit["seed"]]
        h=np.array([r["h"] for r in selected]);lab="seed "+str(fit["seed"])
        axes[0].loglog(h,[r["numerical_to_target_upper"] for r in selected],"o-",label=lab)
        axes[0].loglog(h,np.full_like(h,fit["continuous_path_w2_upper"]),":",color=axes[0].lines[-1].get_color())
        axes[1].loglog(h,[r["numerical_to_learned_upper"] for r in selected],"o-",label=lab)
    axes[0].set(title="Target error and path-error floor",xlabel="Step size h",ylabel="Coupling upper bound")
    axes[1].set(title="Error relative to learned path",xlabel="Step size h",ylabel="Coupling upper bound")
    for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/"learned_path_floor.pdf",bbox_inches="tight");plt.close(fig)
    return fits


def generate(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    path=HarmonicPath()
    discovery=discover_correction(path)
    recovered=discovery["global_minimizer"]
    # Use the actual optimized floating-point parameters, not a replaced analytic solution.
    omega,c=recovered["omega"],recovered["c"]
    z,weights=polar_quadrature(64,64)
    rows=[]; nodes=[]
    cases=[("base","Euler",0.,None),("discovered","Euler",omega,c),
           ("minus_2pct","Euler",.98*omega,path.best_rotation_coefficient(.98*omega)),
           ("plus_2pct","Euler",1.02*omega,path.best_rotation_coefficient(1.02*omega)),
           ("base","Heun",0.,None),("base","RK3",0.,None)]
    for name,method,w,rotation in cases:
        for steps in (8,16,32,64,128,256,512):
            numerical,counts,snapshots=integrate(path,z,steps,method,w,keep_nodes=(steps==128))
            metric=errors(path,numerical,z,weights,w,steps,rotation,method=method)
            rows.append({"field":name,"method":method,"steps":steps,"h":path.horizon/steps,
                         "omega":w,"c":rotation if rotation is not None else "",
                         **metric,**counts})
            for t,state in snapshots[1:]:
                nodes.append({"field":name,"method":method,"steps":steps,"time":t,
                              **errors(path,state,z,weights,w,steps,rotation,t,method)})
    write_csv(output/"convergence.csv",rows);write_csv(output/"intermediate_nodes.csv",nodes)
    residual_rows=[]
    for label,w,rotation in [("base",0.,path.best_rotation_coefficient(0.)),
                             ("discovered",omega,c),("minus_2pct",.98*omega,path.best_rotation_coefficient(.98*omega)),
                             ("plus_2pct",1.02*omega,path.best_rotation_coefficient(1.02*omega))]:
        for t in np.linspace(0,1,21):
            residual_rows.append({"field":label,"time":float(t),"omega":w,"c":rotation,
                                  "residual_upper_exact":path.residual_upper(w,rotation,t)})
    write_csv(output/"residuals.csv",residual_rows)
    # Independent quadrature and inverse-tolerance refinement at the finest grid.
    refinement=[]
    for radial,angular,tol in [(32,32,2e-14),(64,64,2e-14),(96,128,2e-14),(96,128,2e-15)]:
        q,p=polar_quadrature(radial,angular)
        for label,w,rotation in [("base",0.,None),("discovered",omega,c)]:
            numerical,counts,_=integrate(path,q,512,"Euler",w,inverse_tol=tol)
            refinement.append({"field":label,"radial_order":radial,"angular_order":angular,
                               "inverse_tolerance":tol,**errors(path,numerical,q,p,w,512,rotation),
                               "target_second_moment_quadrature":float(p@np.abs(path.map(1,q))**2),
                               "target_second_moment_exact":path.target_second_moment(1)})
    write_csv(output/"refinement.csv",refinement)
    # Accurate ODE integration checks the continuous path independently from the Euler update.
    labels=np.array([.1+.2j,.6-.4j,-.9+.3j,1.7+1.1j,-2.2-1.0j])
    path_validation=[]
    for label,w,_ in [("base",0.,None),("discovered",omega,c),("minus_2pct",.98*omega,None)]:
        sol=solve_ivp(lambda t,x:path.field(t,x,w),[0,1],path.map(0,labels),
                      method="DOP853",rtol=2e-12,atol=2e-13,dense_output=True)
        if not sol.success:raise ArithmeticError(sol.message)
        for t in (.25,.5,.75,1.):
            path_validation.append({"field":label,"time":t,
                                    "max_particle_reference_difference":float(np.max(np.abs(sol.sol(t)-path.exact(t,labels,w)))),
                                    "reference_nfe":int(sol.nfev)})
    write_csv(output/"continuous_path_validation.csv",path_validation)
    infeasible=gaussian_infeasibility();write_csv(output/"infeasible_gaussian.csv",infeasible)
    # Equal actual field-evaluation budgets; repeated timing has no accuracy role.
    costs=[]
    for budget in (48,96,192,384):
        for name,method,w,rotation,stages in [("discovered","Euler",omega,c,1),
                                             ("base","Euler",0.,None,1),
                                             ("base","Heun",0.,None,2),
                                             ("base","RK3",0.,None,3)]:
            times=[]
            for repeat in range(3):
                numerical,counts,_=integrate(path,z,budget//stages,method,w)
                times.append(counts["runtime_seconds"])
            costs.append({"field":name,"method":method,"nfe_budget":budget,"steps":budget//stages,
                          "median_runtime_seconds":float(np.median(times)),
                          "min_runtime_seconds":float(min(times)),"max_runtime_seconds":float(max(times)),
                          "batch_size":len(z),"inverse_iterations":counts["inverse_iterations"],
                          **errors(path,numerical,z,weights,w,budget//stages,rotation,method=method)})
    write_csv(output/"equal_cost.csv",costs)
    def tex_number(value):
        mantissa,exponent=f"{value:.3e}".split("e")
        return "$"+mantissa+r"\times10^{"+str(int(exponent))+"}$"
    table=[r"\begin{tabular}{@{}lrrr@{}}",r"\toprule",
           r"Method & $W_2$ lower & $W_2$ upper & Particle RMS\\",r"\midrule"]
    for label,name,method in [("Corrected Euler","discovered","Euler"),("Base Heun","base","Heun"),("Base RK3","base","RK3")]:
        row=next(r for r in costs if r["field"]==name and r["method"]==method and r["nfe_budget"]==384)
        table.append(label+" & "+" & ".join(tex_number(row[key]) for key in
                      ["w2_lower_second_moment","w2_upper_coupling","particle_rms"])+r"\\")
    table.extend([r"\bottomrule",r"\end{tabular}"])
    (output/"equal_cost_table.tex").write_text("\n".join(table)+"\n")
    orders=[]
    for name,method,_,_ in cases:
        pair=[r for r in rows if r["field"]==name and r["method"]==method and r["steps"] in (256,512)]
        for metric in ("w2_lower_second_moment","w2_upper_coupling","particle_rms"):
            orders.append({"field":name,"method":method,"metric":metric,
                           "order_256_to_512":float(np.log2(pair[0][metric]/pair[1][metric])),
                           "fine_error":pair[1][metric]})
    write_csv(output/"orders.csv",orders)
    import scipy
    learned_fits=learned_path_experiment(path,output)
    summary={"parameters":vars(path),"contraction_upper_T":path.contraction(1),
             "quadrature":{"radial":64,"angular":64,"nodes":len(z)},
             "correction_discovery":discovery,"orders":orders,
             "continuous_path_max_difference":max(r["max_particle_reference_difference"] for r in path_validation),
             "infeasible_scalar_budget_minimum":min(r["scalar_budget"] for r in infeasible),
             "learned_coefficient_fits":learned_fits,
             "environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
                            "platform":platform.platform(),"machine":platform.machine(),"processor":platform.processor(),
                            "cpu_count":os.cpu_count(),"threads":{k:os.environ.get(k) for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]}},
             "metric_warning":"Continuum W2 lower and upper bounds evaluated by deterministic quadrature; no exact full-W2 claim or interval-arithmetic certification.",
             "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    render(output,rows,costs,residual_rows,discovery,infeasible)
    return summary


def render(output,rows,costs,residuals,discovery,infeasible):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    colors={"base":"#466b92","discovered":"#c44b30","minus_2pct":"#849d41","plus_2pct":"#845596"}
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.55))
    for name in ("base","discovered"):
        selected=[r for r in rows if r["field"]==name and r["method"]=="Euler"]
        h=np.array([r["h"] for r in selected]);lo=np.array([r["w2_lower_second_moment"] for r in selected]);up=np.array([r["w2_upper_coupling"] for r in selected])
        axes[0].loglog(h,up,"o-",color=colors[name],label=name+" upper")
        axes[0].loglog(h,lo,"--",color=colors[name],label=name+" lower")
        axes[0].fill_between(h,lo,up,color=colors[name],alpha=.12)
        axes[1].loglog(h,[r["particle_rms"] for r in selected],"o-",color=colors[name],label=name)
    for name in ("discovered","minus_2pct","plus_2pct"):
        selected=[r for r in rows if r["field"]==name and r["method"]=="Euler"]
        axes[2].loglog([r["h"] for r in selected],[r["w2_upper_coupling"] for r in selected],"o-",color=colors[name],label=name.replace("_"," "))
    for ax,title in zip(axes,["Distribution bounds","Particle error","Perturbed corrections"]):
        ax.set(xlabel="Step size h",ylabel="Error",title=title);ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(output/"harmonic_convergence.pdf",bbox_inches="tight");plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.55))
    for name in ("base","minus_2pct","plus_2pct"):
        selected=[r for r in residuals if r["field"]==name]
        axes[0].semilogy([r["time"] for r in selected],[r["residual_upper_exact"] for r in selected],color=colors[name],label=name.replace("_"," "))
    axes[0].set(title="Residual certificate",xlabel="Time t",ylabel="L2 residual upper bound")
    for run in discovery["local_runs"]:
        trace=np.array(run["trace"])
        axes[1].semilogy(np.maximum(trace[:,2],1e-16),label="initial omega="+str(run["initial"][0]))
    axes[1].set(title="Local optimizer can stall",xlabel="Residual function call",ylabel="Residual upper bound")
    axes[2].plot([r["time"] for r in infeasible],[r["scalar_budget"] for r in infeasible],label="scalar budget (positive)")
    axes[2].plot([r["time"] for r in infeasible],[r["all_field_residual_lower"] for r in infeasible],label="all-field residual lower")
    axes[2].plot([r["time"] for r in infeasible],[r["best_affine_projection_residual"] for r in infeasible],label="minimum affine residual")
    axes[2].set(title="Directional infeasibility",xlabel="Time t",ylabel="Value")
    for ax in axes:ax.legend(fontsize=5.5);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/"residual_discovery.pdf",bbox_inches="tight");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(6.8,2.65))
    for name,method in [("discovered","Euler"),("base","Euler"),("base","Heun"),("base","RK3")]:
        selected=[r for r in costs if r["field"]==name and r["method"]==method]
        for ax,abscissa in zip(axes,["nfe_budget","median_runtime_seconds"]):
            ax.loglog([r[abscissa] for r in selected],[r["w2_upper_coupling"] for r in selected],"o-",label=name+" "+method)
    axes[0].set(xlabel="Field evaluations",ylabel="W2 coupling upper bound")
    axes[1].set(xlabel="Median batch runtime (s)",ylabel="W2 coupling upper bound")
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/"harmonic_cost.pdf",bbox_inches="tight");plt.close(fig)
    # Density contours are generated parametrically from equal source radii.
    path=HarmonicPath();fig,axes=plt.subplots(1,2,figsize=(6.3,2.8))
    theta=np.linspace(0,2*np.pi,401)
    for t,ax in zip([0.,1.],axes):
        for radius in [.25,.5,.75,1.,1.5,2.]:
            points=path.map(t,radius*np.exp(1j*theta))
            ax.plot(points.real,points.imag,lw=1)
        ax.set(title="Mapped source circles at t="+str(t),xlabel="x1",ylabel="x2",aspect="equal")
    fig.tight_layout();fig.savefig(output/"harmonic_shape.pdf",bbox_inches="tight");plt.close(fig)
