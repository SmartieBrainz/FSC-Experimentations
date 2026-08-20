# Two minimal models of decision-making, fit side by side

This project fits two small, readable models to the same behavioral data and asks what each one
actually learned. The data is a win-stay/lose-shift (WSLS) task: two arms, each trial a win or a
loss, and the rule is stay after a win and switch after a loss.

- A **Finite State Controller** (FSC), a discrete-state controller from Nicoletti and Celani
  (2026), fit by MAPSO. Upstream: [FSC-inference-MAPSO](https://github.com/giorgionicoletti/FSC-inference-MAPSO).
- A **tinyRNN**, the smallest GRU that still explains the behavior, from Ji-An, Benna and Mattar
  (Nature, 2025), found by nested cross-validation over hidden size. Upstream:
  [tinyRNN](https://github.com/cgc/tinyRNN).

Both libraries are used as written, not reimplemented. The one question underneath the whole
project: the FSC is cheaper, needs less data, and keeps its parts named, so what does the GRU's
extra machinery buy?

## Start here: the three findings

Read these three notebooks. Each one is a working log and a plain-language guide at the same time,
written for a reader with no background in the models or in programming. Each runs top to bottom
from the committed fits, so every figure regenerates in seconds without retraining.

1. **[finding_1_the_pipeline.ipynb](finding_1_the_pipeline.ipynb)** builds the bench: it joins
   both libraries under one data format, fits an FSC and a GRU to the same trials, and shows they
   match on every ordinary measurement. By held-out likelihood, the win-stay/lose-shift rates, and
   the full action-transition matrix, the GRU looks like it learned the rule. The FSC does it with
   two states and stays readable; the GRU spreads the same one bit of memory across more units and
   buys no extra behavior.

2. **[finding_2_rnn_offmanifold.ipynb](finding_2_rnn_offmanifold.ipynb)** shows that match is only
   skin deep. Feed the GRU a run of wins longer than any streak in the data and its confidence that
   it should stay slides, while the FSC holds it flat forever. The GRU learned an approximation that
   is right where the data lives and wrong outside it. It never held the rule itself.

3. **[finding_3_solving_the_rnn.ipynb](finding_3_solving_the_rnn.ipynb)** takes the direct route:
   it solves by hand for GRU weights that ARE the controller, exactly. That solved network holds the
   off-manifold probe flat, so the drift is about how training arrives at weights, not the
   architecture. Two results carry it: a single hidden unit reproduces a three-state machine's
   behavior, and the correct correspondence between the two memories is an affine map that exists
   exactly when the hidden size is at least the number of states minus one. This notebook also lays
   out the mathematics of the solve step by step, and gathers the whole L1 regularizer story in one
   place: the penalty buys fewer effective dimensions and a more decisive size pick, and pays for it
   with a slightly worse fit and worse off-manifold behavior.

## Running them

Run any findings notebook from the repository root, top to bottom. The committed models in
`fitted_models/` load by default, so nothing retrains. Three switches, `FORCE_REFIT_FSC_SWEEP`,
`FORCE_REFIT_BEST_FSC`, and `FORCE_RETRAIN_RNN`, all default to `False`; flip one only when you
want to refit and overwrite the committed models for a new test. Finding 1's section 0 clones the
two upstream repos and applies two small automated patches (a portable model-save path and a
one-line fix for variable-length sessions); run it once and the libraries appear locally.

To confirm the reorganization did not move any established number, run the regression guard:

```
python scripts/check_findings.py          # fast: committed artifacts and outputs
python scripts/check_findings.py --run     # full: re-executes the three notebooks from cache
```

It checks the upstream repos and datasets are present, the FSC selected for random_init is still
M* = 2, the win-stay/lose-shift rates read straight off the data still match what the notebooks
report, and every findings notebook runs with zero errors.

## The detailed record: experiments/

The [experiments/](experiments/) folder holds the raw notebooks behind each finding, kept as the
full working record. They carry more datasets, more robustness checks, and the intermediate steps
the findings summarize. Each one has a small guard cell at the top so it runs whether you launch it
from `experiments/` or from the repository root.

- `fsc_tinyrnn_pipeline.ipynb` and `fsc_tinyrnn_pipeline_ts.ipynb`: the original pipeline behind
  Finding 1.
- `rnn_offmanifold_finding.ipynb`: the standalone off-manifold study behind Finding 2.
- `analytical_one_unit_fsc.ipynb`, `analytical_rnn_fsc_wsls.ipynb`, `analytical_rnn_fsc_rwsls.ipynb`,
  `basics_h1_det_init.ipynb`: the analytical solves behind Finding 3, including the stochastic
  control and the two-way affine map worked out in full.
- `fsc_rnn_wsls_paper.ipynb` and `fsc_rnn_wsls_paper_ts.ipynb`: a cleaned rewrite of the pipeline
  that trains without the L1 penalty, with the representable-versus-reachable study.

Sections 11 and 12 of Finding 1 (the `det_init` dataset and the four-dataset determinism spectrum)
extend the analysis to a teammate's later datasets. The L1 regularizer comparison they motivated now
sits with the rest of the L1 story in Finding 3.

## The data

Everything reads one format, a list of **sessions**. A session is one continuous block of behavior:

```python
sessions = [
    {"actions": [...], "observations": [...], "reward": [...optional...]},  # length-T each
    ...
]
```

Values can be integers, strings, or float categories; the pipeline encodes them to `0..K-1` once,
so the FSC and the GRU see identical trials. Sessions can have different lengths. For the shipped
WSLS data, `actions` is the arm chosen and `observations` is the outcome (1 = win); the outcome of
trial `t` drives the choice on trial `t+1`. The `datasets/` folder ships five WSLS variants that
span how strict the rule is, from a stochastic control to a hard deterministic rule.

## Layout

```
finding_1_the_pipeline.ipynb       the bench: fit both models, line them up
finding_2_rnn_offmanifold.ipynb    the GRU never held the rule off the data
finding_3_solving_the_rnn.ipynb    solve for the GRU that is the controller
scripts/check_findings.py          regression guard for the above
experiments/                       the raw notebooks behind each finding
datasets/                          the five WSLS datasets
fitted_models/                     committed FSC and GRU fits (loaded by default)
FSC-inference-MAPSO/  tinyRNN/      cloned by Finding 1's section 0 (git-ignored)
```

## Notes worth knowing

- **Both size selectors over-pick capacity on near-deterministic data.** WSLS is a two-state
  strategy, yet the FSC's held-out loss and the RNN's significance test both land on larger sizes,
  because once every size is near-perfect the leftover differences are tiny but still win. The
  RNN's real effective usage (participation ratio) stays near two dimensions whatever the nominal
  size. Prefer the smallest size within noise of the best.
- **The RNN fits hit their epoch cap without converging by loss.** Cross-entropy on
  near-deterministic data has no finite minimum, so the loss keeps falling while the behavior
  settles hundreds of epochs earlier. Finding 1's section 3.2b works through why that is expected
  here and not a problem.
- **The models condition on slightly different information.** The RNN predicts `action[t]` from
  strictly earlier trials; the FSC's likelihood also sees `observation[t]`. It is immaterial for
  WSLS but matters where the observation is a pre-action stimulus, so do not over-read small
  FSC-versus-RNN loss gaps.
