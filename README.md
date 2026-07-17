# FSC ↔ tinyRNN pipeline

A working bench for fitting two families of **minimal, interpretable decision models** to
the same behavioral data and lining them up against each other:

- **Finite State Controller (FSC)** — a discrete memory-state controller (Nicoletti &
  Celani 2026), fit by MAPSO. Upstream: [`FSC-inference-MAPSO`](https://github.com/giorgionicoletti/FSC-inference-MAPSO).
- **tinyRNN** — the smallest GRU that still explains the behavior (Ji-An, Benna & Mattar,
  Nature 2025), found by nested cross-validation over hidden size. Upstream:
  [`tinyRNN`](https://github.com/cgc/tinyRNN).

Everything runs from one notebook: **[`fsc_tinyrnn_pipeline.ipynb`](fsc_tinyrnn_pipeline.ipynb)**.
The current dataset is a win-stay-lose-shift (WSLS) task, but the pipeline is written to be
retargeted — the intent is to grow new experiments on top of it, not to freeze it here.

## Pipeline at a glance

The notebook is linear; each section hands typed objects to the next:

| § | Does | Produces |
|---|------|----------|
| 0 | Clones both upstream repos, installs deps, applies two patches, sets up import switching | working libs on disk |
| 1 | Loads data → one canonical `sessions` format → integer-encodes → train/test split | `encoded_sessions`, `ActSpace`, `ObsSpace` |
| 2 | FSC: sweep memory-state count `M`, pick `M*` by held-out loss, refit on all data | `best_fsc`, `M_STAR`, per-`M` losses |
| 3 | tinyRNN: nested-CV sweep over `hidden_dim`, pick `H*`, extract hidden states; put FSC & RNN losses in the same units | `H_STAR`, `rnn_hidden_states`, loss overlay |
| 4 | **Our own analysis** — decode the FSC belief state, align it trial-by-trial with the RNN hidden state, PCA, etc. | comparison figures/tables |

Sections 0–3 are thin wrappers around the two libraries' own APIs (settings mirror their
example notebooks / experiment scripts, so *how* the models are fit is theirs, not ours).
**Section 4 is the sandbox** — it's our code and the place to build.

## The data contract

Everything downstream keys off a list of **sessions** (one continuous behavioral block
each), so plugging in a new task means producing this and nothing else:

```python
sessions = [
    {"actions": [...], "observations": [...], "reward": [...optional...]},  # length-T each
    ...
]
```

Values can be ints/strings/float-categories; §1.2 encodes them to `0..K-1` once, so the FSC
and RNN see identical trials (this is what makes the state-by-state comparison in §4 valid).
Sessions may be different lengths. §1.1 has three loaders — a ready array (`1.1a`), a
long-format CSV (`1.1b`), or a synthetic generator (`1.1c`) — pick one.

For the shipped WSLS data (`wsls_actions_observations.npz`): `actions ∈ {−1,+1}` is the
choice, `observations ∈ {0,1}` is the **trial outcome** (1 = win). It's exact WSLS —
`observation[t]` is the result of `action[t]` and deterministically drives `action[t+1]`.

## Mechanics you'll touch when extending

- **Import switching.** Both repos ship top-level modules with colliding names (`utils`,
  `FSC`, `MAPSO`, …), so they can't sit on `sys.path` together. `_use_fsc_path()` /
  `_use_tinyrnn_path()` swap between them and clear the stale modules. Call the right one
  before importing from either side; §3 also `os.chdir`s into `tinyRNN/` (its training code
  assumes repo-root as cwd) and `chdir`s back.
- **The two libraries are not vendored into this repo.** §0 clones them fresh and applies
  two automated, idempotent patches: a portable `MODEL_SAVE_PATH` in `path_settings.py`, and
  a one-line variable-length-session bug fix in `datasets/SimpleDataset.py`. Because the
  patches live in the notebook, a clean checkout + run reproduces them exactly.
- **Caching / re-fitting.** FSC fits cache to `fitted_models/fsc/`, RNN fits to
  `tinyRNN/files/saved_model/` (the library's own `temp_summary.pkl` mechanism). Re-running a
  training cell reloads instead of recomputing. Force a rebuild with `FORCE_REFIT_FSC_SWEEP`
  / `FORCE_REFIT_BEST_FSC` / `FORCE_RETRAIN_RNN`, or by deleting the cache dir. `fitted_models/`
  ships the current fits so §2.3/§3.3 onward run without repeating the slow work.
- **Knobs.** FSC: `M_CANDIDATES`, `N_EPOCHS_MAPSO` (budget; see below), `FSC_SWEEP_SEED`.
  RNN: `HIDDEN_DIM_CANDIDATES`, `outer_splits`/`inner_splits`/`seed_num`, `max_epoch_num`,
  `early_stop_counter`. The FSC and RNN size ranges are kept equal so the §3.5 overlay lines
  up 1:1.
- **RNN fits are archived, not just cached.** tinyRNN writes fits under `tinyRNN/files/saved_model/`, which is inside the git-ignored clone. §3.2 mirrors them to `fitted_models/<dataset>/rnn/` (committed) and restores from there when the live cache is missing — so a fresh clone does not repeat the sweep.
- **RNN training speed — already handled, nothing to run by hand.** The upstream library
  forces `n_jobs=1` for RNNs (`agent_pool_auto_train` overrides whatever you pass), so the
  library call alone is single-process. The fits are independent across `hidden_dim`, so §3.2
  farms them out to one OS process per size via `scripts/train_rnn_parallel.py` and *then*
  calls the library, which finds everything finished and skips. Executing the notebook top to
  bottom already gets you this (~8x on a 10-core Mac). `PARALLEL_TRAIN = False` restores the
  single-process path.

  The script is also runnable standalone (`python scripts/train_rnn_parallel.py --dataset
  det_init`) — it's resumable, and it refuses to run if `behavior_train.pkl` belongs to a
  different `DATASET` than the one named (that file has one fixed name and is rewritten on
  every dataset switch, so this would otherwise silently train the wrong data into the right
  folder). §3.1 writes the stamp that makes the check possible.
- **Don't reach for the GPU.** The models are tiny (`hidden_dim ≤ 8`, `input_dim` 4), so MPS
  is *measurably slower* than CPU (~1.4x on an M-series Mac) — per-step kernel-launch overhead
  swamps a 4×4 matmul. tinyRNN also runs the network in float64 (`RNNAgent` calls `.double()`),
  which Apple's MPS backend cannot do at all. Parallel CPU processes are the real speedup.

## Things worth knowing before you build on the numbers

- **Both selectors over-pick capacity on near-deterministic data.** WSLS is behaviorally a
  2-state strategy, yet the FSC's arg-min held-out loss lands on `M*=6` and the RNN's
  significance test on `H*=4`, because once every size is near-perfect the leftover
  differences are tiny but still "win." Treat them as upper bounds; prefer the smallest size
  within noise of the best (the notebook flags this at the selection cells).
- **MAPSO has no early stop, on purpose.** Its late stages inject fresh mutations that can
  still escape local optima, so a training-loss patience break (what the RNN uses) would cut
  that off. Instead `N_EPOCHS_MAPSO` is a fixed budget read off the §2.1b convergence curve
  (every `M` here flattens by ~45 iterations). Re-read that plot and raise it for harder data.
- **The two models condition on slightly different information.** The RNN predicts
  `action[t]` from strictly earlier trials; the FSC's likelihood also sees `observation[t]`.
  It's immaterial for WSLS (`action[t]` is independent of `observation[t]`) but matters where
  the observation is a pre-action stimulus — don't over-read small FSC-vs-RNN loss gaps.

## Layout

```
fsc_tinyrnn_pipeline.ipynb      the whole pipeline
wsls_actions_observations.npz   the current dataset
fitted_models/<dataset>/fsc/     committed FSC fits (M-sweep + selected model)
fitted_models/<dataset>/rnn/     committed RNN nested-CV run (all sizes / folds / seeds);
                                 §3.2 restores this into tinyRNN/files/saved_model/ on a
                                 fresh clone, so the sweep is not repeated
FSC-inference-MAPSO/  tinyRNN/   cloned by §0 (git-ignored, not committed)
```
