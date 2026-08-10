#!/usr/bin/env python3
"""Give a worker process the same namespace basics_h1_det_init.ipynb has at the end of its §2.1.

WHY IT IS DONE THIS WAY
-----------------------
`scripts/basics_h1_perturb_sweep.py` has to train exactly the network the notebook trains, with
exactly the notebook's optimiser, loss, data layout and metrics. The obvious way to arrange that
is to copy those definitions into the script, which is what `scripts/train_rnn_parallel.py` does
-- and note the warning it carries, that the copy "has to match field-for-field" or the results
land somewhere the notebook will not look.

So instead of copying, this executes the notebook's own cells. There is one source of truth, and
a cached sweep cannot silently belong to a different model than the one the notebook describes.

It runs cells §0.1 through §2.1, which are setup and definitions. Everything expensive in the
notebook (the training runs) lives in later cells and is not touched. The §0.3 solve does run,
because the solved weights are what the sweep perturbs; that is about 12 seconds per worker.
"""
import os

# BEFORE ANYTHING PULLS IN NUMPY. The notebook's §0.1 sets these too, but by the time its cells
# are exec'd here numpy is already loaded and the setting no longer takes. Without it the
# multi-threaded BLAS makes the §0.3 least-squares solve land on a mirrored solution, and the
# sweep would then be perturbing a different point than the notebook describes.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "basics_h1_det_init.ipynb"
LAST_SETUP_CELL = "# --- 2.1 --"        # inclusive: the last cell that is pure definitions


def _setup_cells(nb_path=NOTEBOOK):
    """The notebook's code cells up to and including §2.1, in order."""
    nb = json.loads(Path(nb_path).read_text())
    cells = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
    stop = [i for i, s in enumerate(cells) if s.startswith(LAST_SETUP_CELL)]
    if not stop:
        raise RuntimeError(f"{nb_path} has no cell starting with {LAST_SETUP_CELL!r}; "
                           "the sweep script and the notebook are out of sync")
    return cells[:stop[0] + 1]


def load_context(dataset="det_init", H=1, nb_path=NOTEBOOK, verbose=False):
    """Execute those cells and hand back the resulting namespace as a plain dict.

    Returns the names the sweep needs: solved weights, the training function, the metrics, and
    small helpers. Raises if the notebook's config does not match what was asked for, rather
    than quietly sweeping a different dataset than the caller believes.
    """
    import matplotlib
    matplotlib.use("Agg")                      # workers draw nothing; keep any figure headless
    import torch
    torch.set_num_threads(1)                   # belt and braces alongside the env vars above

    ns = {"__name__": "__main__"}
    cwd = os.getcwd()
    os.chdir(REPO)                             # the cells use os.path.abspath(".")
    try:
        buf = io.StringIO()
        for src in _setup_cells(nb_path):
            with redirect_stdout(buf):
                exec(compile(src, "<notebook cell>", "exec"), ns)
        if verbose:
            print(buf.getvalue())
    finally:
        os.chdir(cwd)

    if ns["DATASET"] != dataset or ns["H"] != H:
        raise RuntimeError(f"notebook is configured for dataset={ns['DATASET']!r} H={ns['H']}, "
                           f"but the sweep asked for dataset={dataset!r} H={H}")

    import numpy as np
    import torch

    def random_init(seed):
        torch.manual_seed(seed)
        return ns["to_params"](ns["TinyGRU"](H).double())

    return {
        "solved":      ns["SOLVED"],
        "train_from":  ns["train_from"],
        "verdict":     ns["verdict"],
        "perturb":     ns["perturb"],
        "flat":        ns["flat"],
        "theta0":      ns["TH0"],
        "random_init": random_init,
        "ns":          ns,
    }


if __name__ == "__main__":
    ctx = load_context(verbose=True)
    print("context loaded; solved weights have norm "
          f"{__import__('numpy').linalg.norm(ctx['theta0']):.3f}")
