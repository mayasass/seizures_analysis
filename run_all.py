"""
run_all.py  -  runs the analysis pipeline in the right order, by itself.

Lives in the top of the project folder (/Users/maya/PycharmProjects/seizures_analysis).
Scripts are in analysis/ and figures_code/; csv results go to results/, figures to figures/.
Run:

    /usr/local/bin/python3 run_all.py                 # everything, in order
    /usr/local/bin/python3 run_all.py topography ratios   # only these groups

Groups (in the order they run):
    tables         statistical_anaylsis_updated.py   (builds the clean 1,029 table)
    poster_figs    poster_figures_updated.py         (Figs 1-5)
    decomposition  delta_gamma_decomposition.py      (delta / gamma / ratio, UC incl + excl)
    lopo           leave_one_patient_out.py  x2      (UC included, then UC excluded)
    normalization  prepare_patient_normalized.py -> patient_normalized_analysis.py
    topography     prepare_topography_data.py   -> topography_analysis.py
    ratios         prepare_more_ratios.py       -> compare_ratios.py
    figures        followup_figures.py               (Figs 6-8; needs decomposition, lopo, topography first)
    vigilance      vigilance_covariate_models.py     (ratio + vigilance state, Miriam's suggestion)

Everything printed is ALSO saved to run_logs/run_<date>_<time>.txt
-> after a run, just upload that one file to Claude.

The run stops at the first script that fails and tells you which one.
"""

import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")   # csv outputs (decomposition_*, lopo_summary_*)
FIGURES_DIR = os.path.join(HERE, "figures")   # all fig*.png / fig*.svg
FINAL_OUTPUT = "/Users/maya/Documents/lab_project/final_tables/final_output.csv"

# (group, script, extra environment variables, label)
STEPS = [
    ("tables",        "analysis/statistical_anaylsis_updated.py", {}, "clean tables (1,029 cohort)"),
    ("poster_figs",   "figures_code/poster_figures_updated.py",       {}, "poster figures 1-5"),
    ("decomposition", "analysis/delta_gamma_decomposition.py",    {}, "delta vs gamma vs ratio"),
    ("lopo",          "analysis/leave_one_patient_out.py",        {"EXCLUDE_UC": "0"}, "LOPO, UC included"),
    ("lopo",          "analysis/leave_one_patient_out.py",        {"EXCLUDE_UC": "1"}, "LOPO, UC excluded"),
    ("normalization", "analysis/prepare_patient_normalized.py",   {}, "build patient-normalized table"),
    ("normalization", "analysis/patient_normalized_analysis.py",  {}, "patient / stage normalization"),
    ("topography",    "analysis/prepare_topography_data.py",      {}, "build all-electrode table"),
    ("topography",    "analysis/topography_analysis.py",          {}, "topography (global vs focal)"),
    ("ratios",        "analysis/prepare_more_ratios.py",          {}, "build extra-ratios table"),
    ("ratios",        "analysis/compare_ratios.py",               {}, "delta/gamma vs delta/beta vs delta/total"),
    ("figures",       "figures_code/followup_figures.py",             {}, "Figs 6-8 (read the results saved above)"),
    ("vigilance",     "analysis/vigilance_covariate_models.py",   {}, "ratio + vigilance state (covariate and interaction)"),
]
GROUPS = list(dict.fromkeys(s[0] for s in STEPS))


def main():
    wanted = sys.argv[1:] or GROUPS
    unknown = [g for g in wanted if g not in GROUPS]
    if unknown:
        sys.exit(f"Unknown group(s): {unknown}. Choose from: {GROUPS}")
    steps = [s for s in STEPS if s[0] in wanted]

    # --- checks before starting, so nothing fails halfway for a silly reason ---
    missing = sorted({s[1] for s in steps if not os.path.exists(os.path.join(HERE, s[1]))})
    if missing:
        sys.exit(f"These scripts are missing (paths are relative to {HERE}):\n  " + "\n  ".join(missing)
                 + "\nDid organize_folder.sh run? Check the analysis/ and figures_code/ folders.")
    if not os.path.exists(FINAL_OUTPUT):
        sys.exit(f"Cannot find {FINAL_OUTPUT}")

    os.makedirs(os.path.join(HERE, "run_logs"), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    log_path = os.path.join(HERE, "run_logs", f"run_{datetime.now():%Y-%m-%d_%H%M}.txt")
    log = open(log_path, "w")

    def out(line=""):
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    out(f"run_all.py started {datetime.now():%Y-%m-%d %H:%M}; groups: {', '.join(wanted)}")
    summary = []
    for i, (group, script, extra_env, label) in enumerate(steps, 1):
        env = os.environ.copy()
        env.update(extra_env)
        env["MPLBACKEND"] = "Agg"          # never open figure windows that block the run
        env["RESULTS_DIR"] = RESULTS_DIR
        env["FIGURES_DIR"] = FIGURES_DIR
        out("\n" + "=" * 78)
        out(f"[{i}/{len(steps)}] {label}   ({script}"
            + (f", {' '.join(f'{k}={v}' for k, v in extra_env.items())}" if extra_env else "") + ")")
        out("=" * 78)
        t0 = time.time()
        proc = subprocess.Popen([sys.executable, os.path.join(HERE, script)], cwd=RESULTS_DIR, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            out(line.rstrip("\n"))
        code = proc.wait()
        mins = (time.time() - t0) / 60
        summary.append((label, "OK" if code == 0 else f"FAILED (code {code})", mins))
        if code != 0:
            out(f"\n!!! {script} FAILED - stopping here. Send this log to Claude.")
            break

    out("\n" + "=" * 78)
    out("SUMMARY")
    for label, status, mins in summary:
        out(f"  {status:<18} {mins:5.1f} min   {label}")
    not_run = len(steps) - len(summary)
    if not_run:
        out(f"  ({not_run} step(s) not run because of the failure above)")
    out(f"\nFull log saved to: {log_path}")
    log.close()


if __name__ == "__main__":
    main()