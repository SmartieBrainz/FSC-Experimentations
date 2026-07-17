#!/usr/bin/env python3
"""Train the notebook's RNN sweep with one process per hidden_dim.

WHY THIS EXISTS
---------------
tinyRNN deliberately refuses to parallelise RNN training: `agent_pool_auto_train`
(training_experiments/training.py) hard-overrides whatever `n_jobs` you pass back to 1
whenever `agent_type == 'RNN'`. So the notebook's §3.2 cell is single-process no matter what,
and the full sweep is slow.

The fits are independent across `hidden_dim`, though, so we can just run one OS process per
size ourselves. That's what this does. It is exactly equivalent to §3.2 -- same configs, same
output folders -- only faster.

USAGE
-----
    # 1. run the notebook down to and including §3.1 (it writes behavior_train.pkl)
    # 2. then, from the repo root:
    python scripts/train_rnn_parallel.py --dataset det_init
    # 3. re-run the notebook from §3.2; it finds the finished fits and skips training.

    python scripts/train_rnn_parallel.py --dataset det_init --dims 2 4    # only some sizes
    python scripts/train_rnn_parallel.py --dataset det_init --jobs 4      # fewer processes

SAFETY
------
Already-trained configs are skipped by tinyRNN itself (it checks for `temp_summary.pkl` in
each model folder), so re-running this is safe and resumable -- it only trains what's missing.

NOTE ON GPU: don't bother. The models are tiny (hidden_dim <= 8, input_dim 4), so an MPS/CUDA
device is *slower* than CPU here -- per-step kernel launch overhead dominates a 4x4 matmul.
tinyRNN also runs the network in float64 (`RNNAgent.__init__` calls `.double()`), which Apple's
MPS backend cannot do at all. Parallel CPU processes are the real speedup.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TINYRNN_ROOT = REPO_ROOT / "tinyRNN"
DATA_REL = "files/SimpleDatasetInput/behavior_train.pkl"   # written by the notebook's §3.1
DEFAULT_DIMS = [1, 2, 3, 4, 5, 6, 7, 8]                    # keep in sync with HIDDEN_DIM_CANDIDATES


def build_config(dataset, hidden_dim, A, Y):
    """Mirror the notebook's §3.2 `base_config` EXACTLY.

    This has to match field-for-field: the model's folder name is derived from the config, so
    any drift means these fits land somewhere the notebook won't look for them (and it would
    silently retrain from scratch instead of reusing them)."""
    behav_data_spec = {
        "dataset": "Simple",
        "data": DATA_REL,
        "input_format": [dict(name="action", one_hot_classes=A),
                         dict(name="observation", one_hot_classes=Y)],
        "output_dim": A,
        "target_name": "action",
    }
    return {
        "dataset": "Simple", "behav_format": "tensor", "behav_data_spec": behav_data_spec,
        "agent_type": "RNN", "rnn_type": "GRU", "input_dim": A + Y,
        "hidden_dim": hidden_dim, "output_dim": A, "device": "cpu",
        "output_h0": True, "trainable_h0": False, "readout_FC": True, "one_hot": False,
        "lr": 0.005, "l1_weight": 1e-5, "weight_decay": 0, "penalized_weight": "rec",
        "max_epoch_num": 2000, "early_stop_counter": 200,
        "outer_splits": 5, "inner_splits": 4, "seed_num": 2,
        "save_model_pass": "minimal", "training_diagnose": None,
        "exp_folder": f"exp_{dataset}",
    }


def worker(dataset, hidden_dim):
    """Train every fold/seed at ONE hidden_dim. Runs inside tinyRNN/ as its own process."""
    os.chdir(TINYRNN_ROOT)
    sys.path.insert(0, str(TINYRNN_ROOT))

    import torch
    # Each process would otherwise grab every core for BLAS and fight the other processes.
    torch.set_num_threads(2)

    import joblib
    import numpy as np
    from training_experiments.training import behavior_cv_training_config_combination

    # Infer the action/observation alphabet sizes from the data the notebook already encoded
    # to 0..K-1 in §1.2, rather than hardcoding 2/2 -- so this keeps working on a new dataset.
    data = joblib.load(DATA_REL)
    A = int(max(np.max(a) for a in data["action"])) + 1
    Y = int(max(np.max(o) for o in data["observation"])) + 1

    base_config = build_config(dataset, hidden_dim, A, Y)
    config_ranges = {"rnn_type": ["GRU"], "hidden_dim": [hidden_dim],
                     "readout_FC": [True], "l1_weight": [1e-5]}
    behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=1, verbose_level=1)
    print(f"DONE dataset={dataset} hidden_dim={hidden_dim}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help='e.g. "random_init" or "det_init" -- must match §1.1a')
    p.add_argument("--dims", type=int, nargs="+", default=DEFAULT_DIMS)
    p.add_argument("--jobs", type=int, default=None, help="default: one per dim, capped at cores-2")
    p.add_argument("--_worker", type=int, default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args._worker is not None:
        return worker(args.dataset, args._worker)

    data_file = TINYRNN_ROOT / DATA_REL
    if not data_file.exists():
        sys.exit(f"ERROR: {data_file} not found.\n"
                 f"Run the notebook down to §3.1 first (with DATASET = \"{args.dataset}\") -- "
                 f"that cell writes the file this trains on.")

    # behavior_train.pkl has one fixed name and is rewritten every time §1.1a's DATASET changes,
    # so the data sitting on disk may belong to a DIFFERENT dataset than the one named here.
    # Training it into exp_<dataset>/ would silently produce fits that are wrong but look fine.
    # §3.1 stamps the dataset next to the file; refuse to run if it disagrees.
    stamp_file = TINYRNN_ROOT / "files/SimpleDatasetInput/behavior_train.dataset"
    if not stamp_file.exists():
        sys.exit(f"ERROR: {stamp_file} not found, so I can't tell which dataset\n"
                 f"{data_file} actually holds. Re-run the notebook's §3.1 cell with\n"
                 f'DATASET = "{args.dataset}" -- it writes this stamp -- then try again.')
    stamped = stamp_file.read_text().strip()
    if stamped != args.dataset:
        sys.exit(f'ERROR: refusing to train.\n'
                 f'  You asked for  --dataset {args.dataset}\n'
                 f'  but {DATA_REL} currently holds "{stamped}" data.\n\n'
                 f'Set DATASET = "{args.dataset}" in §1.1a and re-run the notebook down to §3.1,\n'
                 f'then run this again. (Otherwise "{stamped}" data would be trained into\n'
                 f'exp_{args.dataset}/ and the results would be silently wrong.)')

    jobs = args.jobs or min(len(args.dims), max(1, (os.cpu_count() or 4) - 2))
    print(f"dataset={args.dataset}  dims={args.dims}  -> {jobs} parallel processes "
          f"({os.cpu_count()} cores)\nAlready-finished configs are skipped automatically.\n")

    running, failed = [], []
    queue = list(args.dims)
    while queue or running:
        while queue and len(running) < jobs:
            d = queue.pop(0)
            cmd = [sys.executable, __file__, "--dataset", args.dataset, "--_worker", str(d)]
            log = open(REPO_ROOT / f"train_{args.dataset}_h{d}.log", "w")
            print(f"  launch hidden_dim={d}  (log: train_{args.dataset}_h{d}.log)")
            running.append((d, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        for item in running[:]:
            d, proc, log = item
            if proc.poll() is not None:
                log.close(); running.remove(item)
                status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
                if proc.returncode != 0:
                    failed.append(d)
                print(f"  hidden_dim={d} finished: {status}")
        if running:
            import time; time.sleep(2)

    if failed:
        sys.exit(f"\n{len(failed)} size(s) FAILED: {failed}. See train_{args.dataset}_h<dim>.log")
    print(f"\nAll sizes done. Re-run the notebook from §3.2 -- it will load these and skip training.")


if __name__ == "__main__":
    main()
