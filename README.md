# FSC ↔ tinyRNN pipeline

Two independent ways of extracting a **minimal, interpretable model of decision-making**
from the same behavioral data, run side by side on a win-stay-lose-shift (WSLS) dataset:

1. **Finite State Controller (FSC)** — Nicoletti & Celani, *"Decoding behavior with
   minimal and interpretable agent models"* (2026). A discrete-state controller fit by
   MAPSO (adaptive particle-swarm). Upstream: [`FSC-inference-MAPSO`](https://github.com/giorgionicoletti/FSC-inference-MAPSO).
2. **tinyRNN** — Ji-An, Benna & Mattar, *"Discovering cognitive strategies with tiny
   recurrent neural networks"* (Nature, 2025). A small GRU trained with the paper's own
   nested cross-validation to find the smallest RNN that explains the behavior.
   Upstream: [`tinyRNN`](https://github.com/cgc/tinyRNN).

Everything lives in **[`fsc_tinyrnn_pipeline.ipynb`](fsc_tinyrnn_pipeline.ipynb)**.

## What this is (and isn't)

Both upstream repos are used as-is (vendored via `git clone`, not reimplemented); the
training/inference calls follow the authors' own example notebooks and experiment
scripts, so nothing about *how* the models are fit is invented here. Section 4 (the
FSC↔RNN comparison) is our own analysis code and is clearly marked as such.

## Running it

The two upstream libraries are **not** committed here. Section 0 of the notebook
`git clone`s them fresh into `FSC-inference-MAPSO/` and `tinyRNN/`, then applies two
small **automated** patches (idempotently, every run):

- `tinyRNN/path_settings.py` — swap the upstream hardcoded Windows `MODEL_SAVE_PATH`
  for a local relative path (portability).
- `tinyRNN/datasets/SimpleDataset.py` — a one-line upstream bug fix for variable-length
  sessions.

Both patches live in the notebook, so a fresh clone + run reproduces them exactly — no
manual edits to the libraries are required, and none of the fitted results depend on
hand-editing. Just open the notebook, select a Python ≥3.9 kernel, and run top to
bottom (Section 0 installs dependencies).

`fitted_models/` ships the already-fit artifacts (the FSC M-sweep and the full RNN
nested-CV run) so you can explore the analysis without repeating the slow fits. Delete
it, or flip the `FORCE_REFIT_*` / `FORCE_RETRAIN_RNN` toggles in the notebook, to
regenerate from scratch.

> Training the full RNN sweep (`5 outer × 4 inner × 2 seeds × 8 hidden dims = 320`
> model fits) is CPU-bound and slow single-threaded. The upstream library deliberately
> forces `n_jobs=1` for RNNs, so the notebook cell is effectively single-process; the
> committed results were produced by running one process per `hidden_dim` in parallel,
> which is equivalent (each fit is independent) and just faster.

## The data

`wsls_actions_observations.npz` — 100,000 sessions × 200 trials (the notebook subsamples
1,000 by default). Verified empirically:

- `actions` ∈ {−1, +1}: which of two options was chosen.
- `observations` ∈ {0, 1}: the **outcome** of that trial (1 = win, 0 = lose).

It is exact win-stay-lose-shift: `observation[t]==1` → next action stays, `==0` → next
action switches. So the observation is the *result* of an action (known only after it)
and deterministically drives the next action.

## Results (on the shipped fit)

Both model families solve WSLS to **≈0.004 nats/trial** on held-out data (near-perfect,
as expected for a deterministic rule), and their model-selection curves agree in shape.
The automated selectors return FSC `M*=6` and RNN `H*=4`.

## Honest caveats (read before building on this)

These are not bugs — the pipeline faithfully follows each paper's own procedure — but
they're the things worth knowing and the questions a reviewer will ask:

1. **Both selectors over-pick complexity on this near-deterministic task.** WSLS is
   behaviorally a 2-state strategy, yet the raw arg-min held-out loss (FSC) and the
   significance-test selector (RNN) return `M*=6` / `H*=4`, because once every `M≥2` /
   `d≥2` is near-perfect, the leftover differences are tiny but still "win." Read these
   as upper bounds and prefer the **parsimonious** choice (the smallest size within
   noise of the best). The notebook flags this inline.
2. **The FSC is fit once per `M`; the authors run many random restarts and keep the best
   by held-out loss.** MAPSO is stochastic and can hit local optima. It doesn't bite on
   this easy data (a single fit is already near-perfect), but for real/noisier data,
   add restarts to match their methodology and get a robust `M*` curve.
3. **The two models condition on slightly different information.** The RNN predicts
   `action[t]` from strictly earlier trials; the FSC's likelihood formally conditions
   `action[t]` on `observation[t]`. Immaterial for WSLS (`action[t]` is independent of
   `observation[t]`), but relevant for tasks where the observation is a pre-action
   stimulus — don't over-read small FSC-vs-RNN gaps in the head-to-head loss.

## Layout

```
fsc_tinyrnn_pipeline.ipynb   the whole pipeline
wsls_actions_observations.npz   the behavioral data
fitted_models/fsc/           cached FSC fits (M-sweep + selected model)
fitted_models/rnn/           cached RNN nested-CV run (all hidden dims / folds / seeds)
FSC-inference-MAPSO/  tinyRNN/   cloned by Section 0 (git-ignored, not committed)
```
