#!/usr/bin/env python3
"""Run the complete numerical reproduction command."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

def source_hashes(config, nonlinear_config, equal_evaluation_config=None, gaussian_higher_order_config=None):
    files = (list((ROOT / "src").rglob("*.py"))
             + list((ROOT / "experiments").glob("*.py"))
             + list((ROOT / "tests").glob("*.py"))
             + [Path(__file__), Path(__file__).with_name("render_mechanism.py"), config, nonlinear_config,
                ROOT / "requirements.txt", ROOT / "requirements.lock.txt"])
    files.append(equal_evaluation_config or ROOT / "experiments" / "equal_evaluation_config.json")
    files.append(gaussian_higher_order_config or ROOT / "experiments" / "gaussian_higher_order_config.json")
    return {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name:
            hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}


def verify_source_snapshot(initial, config, nonlinear_config, equal_evaluation_config=None, gaussian_higher_order_config=None):
    if source_hashes(config, nonlinear_config, equal_evaluation_config, gaussian_higher_order_config) != initial:
        raise RuntimeError("The numerical source or configuration changed during reproduction. Run again.")


def main():
    parser = argparse.ArgumentParser(description="Regenerate and check the population data and figures.")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "reproduction")
    parser.add_argument("--config", type=Path, default=ROOT / "experiments" / "config.json")
    parser.add_argument("--nonlinear-config", type=Path, default=ROOT / "experiments" / "nonlinear_config.json")
    parser.add_argument("--equal-evaluation-config", type=Path, default=ROOT / "experiments" / "equal_evaluation_config.json")
    parser.add_argument("--gaussian-higher-order-config", type=Path, default=ROOT / "experiments" / "gaussian_higher_order_config.json")
    parser.add_argument("--replace-results", action="store_true", help="Permit an explicit output path to replace the included results.")
    args = parser.parse_args()
    if args.output.resolve() == (ROOT / "results").resolve() and not args.replace_results:
        parser.error("Replacing included results requires --replace-results.")
    sources = source_hashes(args.config, args.nonlinear_config, args.equal_evaluation_config, args.gaussian_higher_order_config)
    from run import generate
    from euler_path.figures import render
    from euler_path.nonlinear_figures import render as render_extensions
    from run_nonlinear import generate as generate_extensions
    from euler_path.equal_evaluation import generate as generate_equal_evaluation
    from euler_path.gaussian_higher_order import generate as generate_gaussian_higher_order
    started = time.perf_counter()
    result = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"], cwd=ROOT)
    if result.returncode:
        raise SystemExit(result.returncode)
    summary = generate(args.config, args.output)
    render(args.output)
    extensions = generate_extensions(args.nonlinear_config, args.config, args.output)
    equal_evaluation = generate_equal_evaluation(args.equal_evaluation_config, args.nonlinear_config, args.output)
    gaussian_higher_order = generate_gaussian_higher_order(args.gaussian_higher_order_config, args.output)
    figure_count = 3 + render_extensions(args.output)
    subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("render_mechanism.py")),
                    "--output", str(args.output / "figures")], check=True)
    write_manuscript_values(args.output, extensions)
    verify_source_snapshot(sources, args.config, args.nonlinear_config, args.equal_evaluation_config, args.gaussian_higher_order_config)
    print(json.dumps({"baseline_runs": summary["runs"],
                      "baseline_node_comparisons": summary["node_comparisons"],
                      "nonlinear_runs": extensions["nonlinear"]["runs"],
                      "nonlinear_node_comparisons": extensions["nonlinear"]["node_comparisons"],
                      "sensitivity_runs": extensions["solver"]["sensitivity_runs"],
                      "solver_runs": extensions["solver"]["solver_runs"],
                      "flow_matching_runs": extensions["solver"]["flow_matching_runs"],
                      "equal_evaluation_rows": equal_evaluation["rows"],
                      "gaussian_higher_order_rows": gaussian_higher_order["rows"],
                      "figures": figure_count, "elapsed_seconds": time.perf_counter() - started}, sort_keys=True))


def write_manuscript_values(output, extensions):
    nonlinear, solver = extensions["nonlinear"], extensions["solver"]
    values = {"NonlinearQuadratureOrder": str(nonlinear["primary_quadrature_order"]),
              "NonlinearBaseAction": f"{nonlinear['action_base']:.4f}",
              "NonlinearCorrectedAction": f"{nonlinear['action_corrected']:.4f}",
              "NonlinearActionDifference": f"{nonlinear['action_difference']:.4f}",
              "HeunErrorRatio": f"{solver['heun_corrected_over_base_finest']:.4f}"}
    for row in nonlinear["orders"]:
        label = {"base_w2": "Base", "corrected_w2": "Corrected", "corrected_particle_rms": "Particle"}[row["quantity"]]
        values[f"Nonlinear{label}Order"] = f"{row['observed_order']:.4f}"
        values[f"Nonlinear{label}Coefficient"] = f"{row['independent_coefficient']:.4f}"
        error = row["scaled_fine_error"] * (nonlinear["horizon"] / row["fine_steps"])**row["expected_order"]
        values[f"Nonlinear{label}FineError"] = f"{error:.3e}"
    paths = [f"figures/{name}.pdf" for name in ("nonlinear_convergence", "nonlinear_shape", "gaussian_sensitivity", "solver_dependence", "equal_evaluation")]
    (output / "manuscript_values.json").write_text(json.dumps({"macros": values, "figure_paths_relative_to_results": paths}, indent=2, sort_keys=True) + "\n")
    commands = []
    for name, value in values.items():
        if "e" in value and "Error" in name:
            mantissa, exponent = value.split("e")
            value = mantissa + r"\times 10^{" + str(int(exponent)) + "}"
        commands.append("\\newcommand{\\" + name + "}{" + value + "}")
    (output / "numerical_macros.tex").write_text("\n".join(commands) + "\n")


if __name__ == "__main__":
    main()
