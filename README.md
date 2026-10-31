# MC-HGNet

A framework for physiological estimation from multiband PPG, recorded conditions,
and the latest eligible unlabeled PPG measurement from the same person and day.
Current and historical waveforms share an encoder. Historical messages update
band features before explicit band-pair modeling and condition-aware fusion.

This release follows the **current shared-encoder manuscript**, not earlier
independent-history, reconstruction, staged-refinement, or contrastive experiments.
It includes the seven encoder adapters, matched concatenation baselines, and the
three core ablations. **M = ModernTCN; S = SRFormer.**

No raw/processed datasets, subject-level predictions, pretrained or trained neural
weights, environment directories, or earlier experiment checkpoints are included.
Fitted PCA projections and normalization parameters are not included. Data
preparation fits them locally using each fold's training observations only.
Download datasets and upstream assets separately. This repository is licensed
under MIT; upstream assets retain their own terms (see [third-party notices](THIRD_PARTY.md)).

## Tasks and protocols

| Dataset | Task | PPG input | Development / fixed test |
|---|---|---|---|
| Private glucose cohort | Glucose | Six bands, 50 Hz, 6 s, 300 samples | 1,121 / 371 acquisition groups |
| OpenOx 1.1.1 | SpO₂ and SaO₂, trained separately | Red/IR, resampled 86→50 Hz, 6 s | 543 / 123 measurements |
| CRHE | Joint SBP/DBP | Green/red/IR, adopted 40 Hz, 8 s, 320 samples | 3,388 / 788 measurements |

Five development folds share one fixed future test. These are **within-person
future-measurement evaluations**, not unseen-person cross-validation. OpenOx uses
time blocks within its development observations. CRHE and glucose use grouped
dates. The five reported test values come from five development-fold models,
not an ensemble and not five independently sampled test cohorts.

The private glucose dataset cannot be reconstructed from a public download.
Its configuration and measurement interface are retained for authorized users.
See [data preparation](docs/DATA.md), [training protocol](docs/PROTOCOL.md), and
[verification scope](docs/VERIFICATION.md).

## Installation

Use Python 3.11 and an isolated environment. Commands below work from the repository root.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

For CPU use, install the corresponding official PyTorch CPU wheels. GPU training
uses float32. A single encoder is loaded per process because upstream projects
use overlapping Python module names. The launcher provides that isolation.

## Download assets

The downloader pins the upstream code revisions used by the experiments and
keeps downloaded files under `assets/` (ignored by Git).

```bash
python -m mchgnet.fetch code --encoder moderntcn
python -m mchgnet.fetch code --encoder srformer
python -m mchgnet.fetch pretrained --encoder qwen
```

For an additional encoder, replace `ENCODER` with `pulseppg`,
`papagei_pretrained`, `anyppg`, `sigmappg`, or `timefilter`:

```bash
python -m mchgnet.fetch code --encoder ENCODER
# The first four require official weights; TimeFilter does not.
python -m mchgnet.fetch pretrained --encoder ENCODER
```

The Qwen download uses the current `hf download` CLI. Text is encoded locally
with frozen Qwen3-Embedding-0.6B. Dataset metadata is never sent to an API.

## Prepare public data

CRHE can be downloaded directly from the pinned official repository:

```bash
python -m mchgnet.fetch crhe --dest data/raw/crhe
python -m mchgnet.prepare --dataset crhe --raw data/raw/crhe --out data/processed/crhe --device cuda:0
```

Obtain **OpenOx 1.1.1** through its [official PhysioNet page](https://physionet.org/content/openox-repo/1.1.1/)
under its access conditions. Keep the original CSV files and `waveforms/` tree.
Then prepare both saturation targets together:

```bash
python -m mchgnet.prepare --dataset openox --raw data/raw/openox/1.1.1 --out data/processed/openox --device cuda:0
```

The released OpenOx protocol contains selection/window indices, not signal or
label values. It preserves the independently checked stable-plateau subset used
by the manuscript. Processing every OpenOx record as if labels were already
synchronized would define a different experiment.

Preparation always fits **PCA, feature scaling and target normalization on each
training fold**. Validation and test observations only receive the resulting
transformations. The generated parameters are saved beside the locally processed
data for subsequent training and evaluation. No fitted preprocessing assets are
loaded from the release. SpO₂ and SaO₂ share the same freshly fitted input features.
Use the pinned software versions in `requirements.txt`: low-rank PCA can choose
different null-space coordinates across numerical libraries, so a fresh fit need
not reproduce the archived predictions bit for bit.

## Train and evaluate

Example: CRHE, ModernTCN, complete framework, fold 0:

```bash
python -m mchgnet.train --data data/processed/crhe/bp_joint --encoder moderntcn --variant full --fold 0 --out runs/bp_joint/moderntcn/full/fold0 --device cuda:0
python -m mchgnet.evaluate --data data/processed/crhe/bp_joint --run runs/bp_joint/moderntcn/full/fold0 --device cuda:0
```

To run both M/S variants and their baselines across all five folds on two GPUs:

```bash
python -m mchgnet.launch --data data/processed/crhe/bp_joint --out runs/bp_joint --encoders moderntcn srformer --variants baseline full --gpus 0 1
```

Use `data/processed/openox/openox_spo2` or `.../openox_sao2` for the two oxygen
tasks. Use all seven names in `--encoders` to reproduce the encoder adaptation
comparison. Each GPU takes the next pending experiment when it becomes free.
Each device runs one experiment at a time. Completed runs are reused;
partial runs resume from their saved optimizer and RNG state.

Core ablations use `--variants full no_history no_pairs no_graph`. The paper
uses ModernTCN for glucose/oxygen ablations and SRFormer for BP ablations.
All arms inherit the corresponding full model's training budget.

```bash
python -m mchgnet.launch --data data/processed/openox/openox_spo2 --out runs/openox_spo2 --encoders moderntcn --variants full no_history no_pairs no_graph --gpus 0 1
python -m mchgnet.report --runs runs/openox_spo2 --out results/reproduced_spo2.csv
```

`metrics.json` reports train, validation and fixed-test metrics. The report tool
requires five folds and reports mean and sample SD of the five models. Glucose
RMSE is converted from mmol/L to mg/dL using the manuscript's factor 18. Zone A
uses the Parkes Type 1 grid. Oxygen RMSE is in percentage points; BP is in mmHg.
The small CSV files in `results/` are manuscript reference metrics, not results
generated by the release smoke tests.

## Checks

```bash
python -m unittest discover -s tests -v
```

For a short execution check after preparing a dataset, pass `--max-updates 2`
to the training command and use a separate output directory. Such a run is
explicitly marked as a smoke test and cannot be reported as a paper result.

## Repository layout

```text
mchgnet/                    model, data preparation, training and evaluation
mchgnet/resources/configs/   per-fold settings extracted from current paper model records
mchgnet/resources/protocol/  fixed public-data indices and source-integrity checks
results/                    aggregate manuscript reference metrics
tests/                      synthetic behavior and leakage-guard checks
docs/                       data, protocol and verification documentation
provenance/                 source-definition hashes and release verification
licenses/                   notices for attributed upstream components
```

The release was checked against the original project without packaging its
datasets or weights. See `docs/VERIFICATION.md` for the precise tests performed
and what remains dependent on the user's downloaded assets and hardware.
