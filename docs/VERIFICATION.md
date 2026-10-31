# Release verification

Version 0.1.1 fits PCA and all data-dependent normalization locally from each
training fold. No fitted preprocessing assets, datasets or neural weights are
included. Validation and test observations do not participate in fitting.

## Data and features

* CRHE cleaning reconstructs 4,178 retained measurements. All five development
  partitions and their fixed 788-measurement test match the original records.
  The labels match exactly. Waveform views differ by at most about 1.5e-8 in
  normalized amplitude. All five causal history manifests match exactly.
* OpenOx reconstructs the same 666 eligible measurements and five partitions,
  including the fixed 123-measurement test. The 50 Hz waveforms, SpO2 labels,
  active-record amplitude statistics and five history manifests match the
  archived inputs. The 125 Hz waveform difference is at most about 1.5e-8.
* Default preparation was rerun from the official downloaded OpenOx and CRHE
  files with fresh frozen-Qwen text encoding and no fitted assets available in
  the release. All five folds were checked for training-only fitting, unchanged
  splits/history, correct training-label normalization and finite input features.
  SpO2 and SaO2 receive identical freshly fitted input features.
* A held-out-data perturbation test verifies that changing validation/test
  features does not change fitted PCA/scaling parameters or training features.

## Model compatibility and execution

The following model checks were performed during initial code extraction using
the archived input preprocessing supplied locally. They verify that the compact
model implementation preserves the original computations. They do not establish
identical predictions after fitting a new PCA basis.

* All seven encoder families strictly load their original fold-0 concatenation
  and full-framework checkpoints. No missing parameters are ignored.
* Original and release predictions were compared on public test records.
  ModernTCN and SRFormer were also checked over the whole 123-measurement
  SpO2 test. SRFormer GPU predictions match the archived predictions in that
  check. The other encoder checks use eight fixed records per arm.
* The full model and `no_history`, `no_pairs`, `no_graph` arms strictly load
  their original fold-0 checkpoints on SpO2, SaO2 and joint BP. Their eight-record
  GPU prediction differences are below 2e-5 in the target's units.
* Two-update ModernTCN/CRHE and SRFormer/SpO2 training checks exercise loss,
  backward propagation, optimizer steps, validation and checkpoint saving.
  They are marked as smoke runs and are excluded from paper reporting.
  An interrupted run resumed before the first selectable checkpoint produces
  exactly the same parameters, buffers and logs as uninterrupted execution.
* Synthetic tests check incident-pair aggregation, the exact zero-history
  fallback, masked history gradients, duration-preserving resampling,
  chronological split guards and glucose metric units.
* Packaging checks cover installed package resources, command-line entry points,
  and exclusion of data, fitted preprocessing assets, neural weights, temporary
  runtimes and local paths.

Use Python 3.11 and the package versions in `requirements.txt`. Rank-deficient
PCA can yield different null-space bases across numerical libraries. Newly
prepared features should be used to train new models together with their locally
generated normalizers. Archived checkpoints expect their original PCA basis.
Exact numerical agreement across fresh fits and hardware is not guaranteed.

The complete five-fold training matrix was **not retrained during packaging**.
CSV files under `results/` are reference results from the manuscript, not new
results from the short execution checks. Private-glucose preprocessing and
training cannot be independently verified without authorized access to that
cohort. Its data and fitted preprocessing parameters are not distributed.

Scalar audit results are in `provenance/verification.json`. File integrity for
the source release is recorded in `MANIFEST.sha256`.
