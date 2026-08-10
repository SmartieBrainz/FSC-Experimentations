#!/usr/bin/env python3
"""Run basics_h1_det_init.ipynb §2.5's perturbation ladder with one process per run.

WHY THIS EXISTS
---------------
The ladder is 25 independent training runs of 2000 epochs. In-notebook they run one after
another and take about 100 minutes, which means any change to a FIGURE downstream costs a
100-minute re-execution. The runs share nothing, so we run them in parallel here and write the
results to a cache the notebook loads in milliseconds.

This is the same move `scripts/train_rnn_parallel.py` makes for the tinyRNN sweep, for the same
reason, and it produces the same numbers: each run is deterministic given its starting weights,
and every worker pins itself to a single BLAS thread exactly as the notebook does.

USAGE
-----
    python scripts/basics_h1_perturb_sweep.py                  # default det_init, H = 1
    python scripts/basics_h1_perturb_sweep.py --jobs 4
    python scripts/basics_h1_perturb_sweep.py --force          # ignore an existing cache

The notebook regenerates this cache itself if it is missing, just serially. Deleting the cache
is always safe.
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def cache_path(dataset, H):
    return REPO / "fitted_models" / "basics_h1" / f"perturb_sweep_{dataset}_H{H}.pkl"


# ---- the config the cache is keyed on. Any change here invalidates a stored sweep. ----------
def sweep_config(dataset, H):
    return dict(dataset=dataset, H=H, epochs=2000, lr=0.005, l1_weight=1e-5,
                eps_ladder=[0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0, 16.0], seeds=[0, 1, 2],
                n_random=3, kmax=80, conf=1e-3, n_sessions=1000, version=1)


def _pin_threads():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"


def _build_context(dataset, H):
    """Everything a worker needs: data tensors, the solved weights, and the metric functions.
    Imported lazily so the parent process can fork cheaply."""
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    sys.path.insert(0, str(REPO / "scripts"))
    from basics_h1_common import load_context      # shared with the notebook, see that file
    return load_context(dataset, H)


def run_one(job):
    """One training run. `job` says where to start; everything else is deterministic."""
    _pin_threads()
    import numpy as np
    ctx = _CTX
    kind, eps, seed = job["kind"], job["eps"], job["seed"]

    if kind == "perturbed":
        P0 = ctx["solved"] if eps == 0 else ctx["perturb"](ctx["solved"], eps, seed)
    else:
        P0 = ctx["random_init"](1000 + seed)

    ai, di, _ = ctx["verdict"](P0)
    P1, curve = ctx["train_from"](P0, seed=seed)
    af, df, ok = ctx["verdict"](P1)
    return dict(idx=job["idx"], kind=kind, eps=eps, seed=seed,
                dtheta=float(np.linalg.norm(ctx["flat"](P0) - ctx["theta0"])),
                acc_i=ai, drift_i=di, acc_f=af, drift_f=df, ok=ok,
                loss_f=float(curve[-1]), curve=curve.astype("float32"))


_CTX = None


def _init_worker(dataset, H):
    global _CTX
    _pin_threads()
    _CTX = _build_context(dataset, H)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="det_init")
    p.add_argument("--hidden", type=int, default=1)
    p.add_argument("--jobs", type=int, default=None,
                   help="default: one per performance core, capped at the number of runs")
    p.add_argument("--force", action="store_true", help="recompute even if a cache exists")
    args = p.parse_args()

    _pin_threads()
    out = cache_path(args.dataset, args.hidden)
    cfg = sweep_config(args.dataset, args.hidden)
    if out.exists() and not args.force:
        with open(out, "rb") as f:
            got = pickle.load(f)
        if got.get("config") == cfg:
            print(f"cache already present and matches config: {out}")
            print("nothing to do (use --force to recompute)")
            return
        print("cache present but its config does not match; recomputing")

    jobs = [dict(kind="perturbed", eps=e, seed=s)
            for e in cfg["eps_ladder"] for s in (cfg["seeds"] if e > 0 else [0])]
    jobs += [dict(kind="random", eps=float("nan"), seed=s) for s in range(cfg["n_random"])]
    for i, j in enumerate(jobs):
        j["idx"] = i          # carried through the worker: nan never compares equal, so the
                              # random runs cannot be matched back up by their eps value

    n_jobs = args.jobs or min(len(jobs), max(1, (os.cpu_count() or 4) // 2))
    print(f"{len(jobs)} runs of {cfg['epochs']} epochs across {n_jobs} processes")

    import multiprocessing as mp
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_jobs, initializer=_init_worker, initargs=(args.dataset, args.hidden)) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_one, jobs), 1):
            tag = "random" if r["kind"] == "random" else f"eps={r['eps']:g}"
            print(f"  [{i:>2}/{len(jobs)}] {tag:>9} s{r['seed']}  acc {r['acc_f']:.5f}  "
                  f"drift {r['drift_f']:.1e}  {'holds' if r['ok'] else 'broken'}"
                  f"   ({time.time() - t0:.0f}s elapsed)")
            results.append(r)

    # Sort back into the deterministic order the notebook prints, so the cache does not depend
    # on which worker happened to finish first.
    results.sort(key=lambda r: r["idx"])
    for r in results:
        del r["idx"]
    assert len(results) == len(jobs), (len(results), len(jobs))

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(dict(config=cfg, results=results), f)
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
