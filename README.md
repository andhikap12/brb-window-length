# Window duration vs. domain-generalization objectives in broken rotor bar detection

Code for the CENIM 2026 paper *"Analysis Window Duration Outweighs Domain-Generalization
Objectives in Vibration-Based Broken Rotor Bar Detection Under Unseen Loads"*
(A. Putra, R. F. Rachmadi, M. H. Purnomo).

The paper holds out the two extreme load levels of a public induction-motor database, keeps
architecture and protocol fixed, and compares nine objective-side interventions (adversarial,
conditional, contrastive, orthogonal-subspace) against an empirical-risk baseline over 8 to 20
seeds. The only lever that moves the lightest-load result is the analysis window.

## Status of this release

- **Verified:** after the final prune to the paper's scope, `pytest tests` gives 39 passed, 2 skipped
  without the processed `.npz` files. The earlier run with the files present (109 passed, 1 skipped)
  predates the prune and was not repeated.
- **Verified:** every row of Table I (mean, standard deviation, low-load and high-load Win-F1)
  was recomputed from the stored per-seed results and matches the paper. The deltas agree to
  within 0.01, because the paper subtracts rounded means. This checks the table against the
  stored results, not against a retraining.
- **Not verified:** no experiment has been re-run end to end from a fresh checkout of this
  folder.
- **Runners for 6 of the 15 Table I rows only.** Nine arms (ERM, SupCon, the three orthogonal
  feature separation settings, three DANN weights, DANN with label smoothing) were launched on
  26 and 27 June 2026 from commands that were not saved as scripts. Their logs record the
  method, weights, seeds, epochs and learning rate, but not every flag, so the commands under
  "Table I arms without a saved launcher" are reconstructions and have not been shown to
  reproduce the logged numbers.
- **Edits made for release:** `set -o pipefail` was added to the runners where it was missing,
  the Windows-only `./venv/Scripts/python.exe` was replaced by `${PYTHON:-python}`, and the
  absolute paths in the audit scripts were replaced by paths relative to the repository root.
  The audit scripts read `results/`, `models/` and `data/` produced by the runners and
  preprocessing, so none of them run on a fresh checkout until those exist.
- **Language:** code comments and runner headers are mostly Indonesian. They document the
  pre-registered criteria written before each experiment was run. References to the paper
  read "Sec. X" and "Table N". The experiment codes that appear in comments and printed
  headings (Uji A to F, M13, M14, criterion IDs such as L-P1) are explained under "Internal
  labels" below.
- Exploratory scripts written after the paper's experiments were frozen are not included.
- **Scope:** only the Treml motor pipeline and the experiments the paper reports are kept. The
  Ottawa/bearing/gearbox pipelines, order-normalised and per-sensor representations, the other
  backbones, augmentation tracing, the GRL on/off schedules and the exploratory input-representation
  runners were removed (they are in the git history). The same holds for the exploratory
  w65536 and global-normalization arms, the amplitude-shuffle and transient-window checks,
  and the Rec-Accuracy sweep over the whole family, none of which the paper reports. The
  `--env-band` option of `run_preprocessing.py` and the token cap in the backbones remain in
  the code but no result in the paper uses them.

## Setup

Python 3.11, NVIDIA GPU (the paper's runs used a 6 GB RTX 4050 Laptop, fp32).

```bash
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
export PYTHON=python                                   # runners read $PYTHON
```

## Data

The vibration database is public: Treml et al., *Experimental Database for Detecting and
Diagnosing Rotor Broken Bar in a Three-Phase Induction Motor*, IEEE DataPort, 2020,
[doi:10.21227/fmnm-bn95](https://doi.org/10.21227/fmnm-bn95). Place the extracted folder at
`DatasetRaw/banco de dados experimental/` (see `src/config.py`). Then build the windowed
datasets used in the paper:

```bash
python scripts/run_preprocessing.py --motor                                  # w4096, stride 4096
python scripts/run_preprocessing.py --motor --window 16384 --stride 4096     # w16384, 75% overlap
python scripts/make_subsample_w4096.py                                       # w4096, window count matched to w16384
```

The 4096-sample window is 0.539 s and the 16384-sample window is 2.156 s at the database's
7.6 kHz vibration rate.

## Protocol

| item | value |
|---|---|
| split | `--extremes_out`: train on the 6 interior load levels, test on the 2 extremes |
| backbone | `tsitplus` (TSiTPlus), also `tssequencerplus` for the robustness check |
| training | 50 epochs fixed (`--fixed_budget`), batch 64, lr 1e-3, fp32, `--no_augment` |
| seeds, 8-seed set | `42,123,2025,7,17,2024,2026,88` |
| seeds, 20-seed extension | the 8 above plus `1,3,5,11,13,23,31,57,99,101,202,777` |
| seeds, leave-one-load-out | `42,123,2025,7,17` (8 folds x 5 seeds) |
| model selection | none. No path from the test loads to model choice. |

## Where each result comes from

Each runner's header states its pre-registered pass/fail criteria. Run from the repository root.

### Table I, row by row

Cross-checked against the checkpoint names read by `audit/audit_ckpt_ext.py` and against the
training logs. "Run name" is the `--name` the arm was trained under.

| # | arm | run name | launcher |
|---|---|---|---|
| 1 | ERM | `FP32_tsit_ext_baseline` | none saved, see below |
| 2 | SupCon | `SupCon_l10_t007` | none saved, see below |
| 3 | OFS o3/d0.5 | `DSN_o3_d05` | none saved, see below |
| 4 | DANN + SupCon | `DANNSupCon_l10_sc10` | `_run_dannsupcon_dsranfix.sh` |
| 5 | OFS o1/d1 | `DSN_tsit_ext` | none saved, see below |
| 6 | CDAN + SupCon | `CDANSupCon_l10_sc10` | `_run_cdan_ext.sh` |
| 7 | OFS o5/d1.0 | `DSN_o5_d10` | none saved, see below |
| 8 | DANN, lambda 1.0 | `FP32_tsit_ext_dann_lam10` | none saved, see below |
| 9 | DANN, lambda 0.1 | `FP32_tsit_ext_dann_lam01` | none saved, see below |
| 10 | CDAN + entropy | `CDAN_l10_ent` | `_run_cdan_ext.sh` |
| 11 | DSRAN, lambda 1.0, o3/d0.5 | `DSRAN_l10_o3_d05` | `_run_dsran_sweep.sh` |
| 12 | DANN + label smoothing | `FP32_tsit_ext_dann_els01` | none saved, see below |
| 13 | DANN, lambda 0.3 | `FP32_tsit_ext_dann_lam03` | none saved, see below |
| 14 | DSRAN, lambda 0.5, o3/d0.5 | `DSRAN_l05_o3_d05` | `_run_dsran_sweep.sh` |
| 15 | DSRAN variant (o1/d1/r1) | `DSRANfix_l10_o1d1r1` | `_run_dannsupcon_dsranfix.sh` |

The 20-seed extension of ERM, SupCon, DANN lambda 1.0 and DANN + SupCon is
`_run_byvar_seeds.sh` (run names `S20_*`).

### Table I arms without a saved launcher

The commands below are reconstructed. The flags in `BASE` are those that `_run_byvar_seeds.sh`
later used for ERM, SupCon and DANN lambda 1.0, and that script re-ran seeds 42 and 123 as
reproducibility controls. For the orthogonal feature separation arms, lambda 0.1 and 0.3, and
label smoothing, the flags are inferred from the log headers and have no such control.

```bash
BASE="--backbone tsitplus --extremes_out --no_augment --fixed_budget --fp32 --epochs 50 \
      --seeds 42,123,2025,7,17,2024,2026,88"
SC="--supcon_lambda 1.0 --supcon_temp 0.07 --supcon_proj_dim 128"
T=scripts/train_motor_dann.py
python $T $BASE --lambda_max 0.0 --name FP32_tsit_ext_baseline                  # 1  ERM
python $T $BASE --method supcon $SC --name SupCon_l10_t007                      # 2  SupCon
python $T $BASE --lambda_max 1.0 --name FP32_tsit_ext_dann_lam10                # 8  DANN 1.0
python $T $BASE --lambda_max 0.3 --name FP32_tsit_ext_dann_lam03                # 13 DANN 0.3
python $T $BASE --lambda_max 0.1 --name FP32_tsit_ext_dann_lam01                # 9  DANN 0.1
python $T $BASE --lambda_max 1.0 --domain_label_smoothing 0.1 \
       --name FP32_tsit_ext_dann_els01                                          # 12 DANN + smoothing
python $T $BASE --method dsn --name DSN_tsit_ext                                # 5  OFS o1/d1
python $T $BASE --method dsn --dsn_lambda_ortho 3.0 --dsn_lambda_domain 0.5 \
       --name DSN_o3_d05                                                        # 3  OFS o3/d0.5
python $T $BASE --method dsn --dsn_lambda_ortho 5.0 --dsn_lambda_domain 1.0 \
       --name DSN_o5_d10                                                        # 7  OFS o5/d1.0
```

Other results:

| paper | runner / script |
|---|---|
| Window lever (short vs long window, volume control) | `_run_window16384.sh`, `_run_w4096_sub.sh`, `_run_w16384_ovl.sh`, `_run_w16384_method.sh` |
| Robustness across learning rate and backbone | `_run_lr_faktorial.sh`, `_run_tsseq_faktorial.sh`, `_run_hp_faktorial.sh`, `_antre_lr3e4_tsseq.sh` |
| Per-channel normalization, 38 seeds (short window) | `_run_a3b_byvar_n38.sh` |
| Gradient-balanced DANN (`BT-DANN`, `--bt_ratio`): the arm rerun at the long window and the third arm of the LODO sweep | `_run_bt_dann.sh`, `_run_domain_index.sh`, `_run_w16384_method.sh` |
| Window length vs window count (2x2) | `audit/audit_r3_faktorial.py` |
| Leave-one-load-out sweep (LODO figure, panel a) | `_run_lodo.sh`, `audit/audit_lodo.py` |
| Leave-one-load-out for the three strongest arms | `_run_lodo_metode.sh`, `audit/audit_lodo_metode.py` |
| Per-load results of a trained arm | `audit/audit_ckpt_ext.py` |
| Probing the load information (INLP, linear and nonlinear) | `audit/npdann_inlp.py`, `_run_inlp_nl.sh` |
| Signal-level readability without a model (LODO figure, panel b) | `audit/audit_a6_resolvabilitas.py` |
| Current-channel separability | `audit/audit_kanal_arus.py` |
| Resolution arithmetic (bin width, periods per window) | `audit/audit_resolusi_frekuensi.py`, `audit/audit_slip.py` |
| Sampling rate recovered from the recordings (median 7600.4 Hz) | `audit/audit_fs_ratio.py` |
| Shortest steady-state segment (17.96 s, 136470 samples), bound on `--window` | `audit/audit_steady_state.py` |
| Independent re-extraction check, window uniqueness, channel scale ratio | `audit/audit_dataset.py` |
| Table I low-load and high-load columns, residual error, per-load comparisons | `audit/audit_ckpt_ext.py`, then `audit/audit_r2_per_torsi.py` |
| 20-seed extension statistics | `audit/aggregate_seeds.py` |
| Frozen-backbone load probe, ERM vs DANN lambda 1.0 (96.16% to 93.57%) | `audit/audit_stats2.py` |
| Online discriminator accuracy and the ln 6 ceiling | `audit/audit_deep.py`, `audit/audit_stats_fix.py` |
| Gradient-norm ratio over training | `audit/audit_grad_traj.py` |
| Learning rate by backbone grid, read-out rule | `audit/audit_hp_grid.py` |
| Convex hull comparison | `audit/audit_hull_window.py` |
| Normalization screening probe | `audit/audit_normalisasi.py` |
| Implementation checks run before the BT and domain-index runners | `audit/sanity_bt_dann.py`, `audit/sanity_domain_index.py` |

## Layout

```
src/            models, preprocessing, training, evaluation
scripts/        training entry points and experiment runners (_run_*.sh)
scripts/audit/  post-hoc analyses, probes, statistics
tests/          unit and characterization tests
```

## Internal labels

Comments, runner headers and registry keys use short codes from the experiment log. The ones
that matter for reading the code:

| code | meaning | paper |
|---|---|---|
| Uji C | short window (4096) against long window (16384, no overlap) | Sec. IV-B |
| Uji E | long window with 75% overlap (stride 4096) | Sec. IV-B |
| Uji F | objective-side arms rerun at the long overlapped window | Sec. IV-A |
| Uji A | domain-index regression head on top of BT-DANN + SupCon | Sec. VI (LODO arm) |
| M3 | inference (not measured directly) that a categorical domain head lets the backbone satisfy the adversarial objective by permuting labels instead of removing information; the evidence cited is a discriminator loss above ln 6 | Sec. IV-D |
| M9 | eight seeds mislead: the baseline falls from 89.40 to 87.64 at twenty | Sec. IV-A |
| M10 | no method separates from the baseline at twenty seeds | Sec. IV-A |
| M11 | seed noise dominates; the contrastive arm would need about 74 seeds for power 0.80 | Sec. IV-A, VI |
| M13 | the test loads lie outside the convex hull of the training loads (53.1% at the lightest load and 18.2% at the heaviest in the original recording-level measurement; `audit/audit_hull_window.py` redoes it per window); established as a description, whether it is the cause was left open | Sec. V-A, V-E |
| M14 | at light load the short window is shorter than one modulation period of the sideband spacing; measured, with the causal test supporting it | Sec. V-B, V-E |
| L-P*, R1-P*, R2-P*, R3-P*, A2-P*, A6-P* | pass/fail criteria written in the docstring of the script that applies them, before any number was read | as cited in each docstring |
| ERM, OFS, BT | empirical risk baseline, orthogonal feature separation, gradient-balanced (BT-DANN) | Table I, Sec. IV-A |

Uji B (an amplitude arm and a frequency arm of augmentation), its criteria B-P1 to B-P3 and
the run U2 that `audit/audit_hull_window.py` gates were never run: the gate failed. U3 (an
envelope-channel lever) was cancelled because its premise reversed at window level. None of
these has a result in the paper, which uses no augmentation (Sec. III-B). U1 is a different
thing: the twenty-seed extension of Uji E, which the paper does report (Sec. IV-B).

## AI assistance

Claude (Anthropic) edited the paper's draft for English readability without changing its content
and helped brainstorm code, which the authors implemented. The release itself (this README, the
repository-relative paths in the audit scripts, and the test re-run) was prepared with Claude Code.

## License and citation

MIT, see `LICENSE`. The dataset keeps its original license. Cite via `CITATION.cff`.
