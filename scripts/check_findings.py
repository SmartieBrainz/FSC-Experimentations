#!/usr/bin/env python3
"""Regression guard for the three findings notebooks and the artifacts they read.

The reorg into findings/ + experiments/ must not weaken the pipeline's fidelity to the two
upstream repos or move the numbers the notebooks already report. This script checks that.

Fast path (default): checks committed artifacts and the committed notebook outputs, no
retraining, a few seconds.
  - the two upstream repos and the datasets are on disk,
  - the FSC selected for random_init is M* = 2 (parsimony pick),
  - the win-stay / lose-shift rates read straight off each dataset still match what the
    notebooks state (random_init is a hard 1.0/1.0 rule; rwsls_det_init is the ~0.67/0.20
    stochastic control),
  - each findings notebook's committed cell outputs carry zero errors.

Full path (--run): additionally re-executes the three findings notebooks from the committed
fits (loading only, no refit) and asserts every cell runs with zero errors. This exercises the
FSC load, the trained-GRU load, the numpy GRU step, the off-manifold probe, and the analytical
solve, so any silent drift in a committed fit surfaces as a failed cell.

Exit code 0 = all checks passed, 1 = something drifted. Run from the repository root.
"""
import sys, os, glob, json, subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FINDINGS = [
    "finding_1_the_pipeline.ipynb",
    "finding_2_rnn_offmanifold.ipynb",
    "finding_3_solving_the_rnn.ipynb",
]
# dataset -> (expected P(stay|win), P(shift|lose), tolerance)
RATE_EXPECT = {
    "wsls_actions_observations_random_init.npz": (1.0, 1.0, 0.01),
    "rwsls_actions_observations_det_init.npz":   (0.665, 0.201, 0.03),
}

passes, fails = [], []
def check(name, ok, detail=""):
    (passes if ok else fails).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


def wsls_rates(npz_path):
    import numpy as np
    d = np.load(npz_path)
    a, o = d["actions"], d["observations"]
    win = int(o.max())  # outcomes are {0,1}, 1 = win
    stay_w = tot_w = shift_l = tot_l = 0
    for i in range(a.shape[0]):
        for t in range(1, a.shape[1]):
            if o[i, t-1] == win:
                tot_w += 1; stay_w += int(a[i, t] == a[i, t-1])
            else:
                tot_l += 1; shift_l += int(a[i, t] != a[i, t-1])
    return (stay_w / tot_w if tot_w else float("nan"),
            shift_l / tot_l if tot_l else float("nan"))


def notebook_errors(path):
    nb = json.load(open(path))
    n_err = sum(1 for c in nb["cells"] if c["cell_type"] == "code"
                for out in c.get("outputs", []) if out.get("output_type") == "error")
    n_fig = sum(1 for c in nb["cells"] if c["cell_type"] == "code"
                for out in c.get("outputs", []) if "image/png" in out.get("data", {}))
    return n_err, n_fig


def main():
    run = "--run" in sys.argv
    os.chdir(ROOT)
    print(f"check_findings: {'full (re-executes notebooks)' if run else 'fast (committed artifacts)'}\n")

    print("Environment:")
    for repo in ("tinyRNN", "FSC-inference-MAPSO"):
        check(f"upstream repo {repo}/ present", os.path.isdir(repo),
              "clone it via section 0 of a findings notebook" if not os.path.isdir(repo) else "")
    for nb in FINDINGS:
        check(f"{nb} present", os.path.isfile(nb))

    print("\nFSC selection (random_init):")
    fsc = glob.glob("fitted_models/random_init/fsc/best_fsc_M*.pkl")
    m_star = [int(os.path.basename(p).split("_M")[1].split(".pkl")[0]) for p in fsc]
    check("M* = 2 (parsimony pick)", m_star == [2], f"found {sorted(m_star)}")

    print("\nWin-stay / lose-shift rates read off the data:")
    for fname, (esw, esl, tol) in RATE_EXPECT.items():
        p = os.path.join("datasets", fname)
        if not os.path.isfile(p):
            check(f"{fname} present", False); continue
        sw, sl = wsls_rates(p)
        check(f"{fname} P(stay|win)={sw:.3f}, P(shift|lose)={sl:.3f}",
              abs(sw - esw) <= tol and abs(sl - esl) <= tol,
              f"expected ~{esw}/{esl} (tol {tol})")

    if run:
        print("\nRe-executing findings notebooks from committed fits (no refit):")
        for nb in FINDINGS:
            r = subprocess.run(
                [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                 "--execute", "--inplace", "--ExecutePreprocessor.timeout=600", nb],
                capture_output=True, text=True)
            check(f"{nb} executed cleanly", r.returncode == 0,
                  (r.stderr.strip().splitlines() or [""])[-1][:120] if r.returncode else "")

    print("\nCommitted notebook outputs carry no errors:")
    for nb in FINDINGS:
        if not os.path.isfile(nb):
            continue
        n_err, n_fig = notebook_errors(nb)
        check(f"{nb}: {n_err} error outputs, {n_fig} figures", n_err == 0)

    print(f"\n{'='*60}\n{len(passes)} passed, {len(fails)} failed")
    if fails:
        print("FAILED:", ", ".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
