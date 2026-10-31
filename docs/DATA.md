# Data preparation

## OpenOx

Official source: [PhysioNet OpenOx 1.1.1](https://physionet.org/content/openox-repo/1.1.1/).
Access and data-use conditions are handled by PhysioNet. The original raw PPG
files are not officially synchronized to the reference labels. The paper uses
666 measurements from 63 people after independent timing/plateau checks.
The fixed test has 123 measurements from 17 people. The five development folds
have 107–108 validation observations. Five single-block development observations
always remain in training.

Required files: `patient.csv`, `encounter.csv`, `bloodgas.csv`,
`pulseoximeter.csv`, and `waveforms/<prefix>/<encounter>_ppg.hea/.dat`.

`mchgnet/resources/protocol/openox_windows.json` identifies each selected encounter/sample by a
SHA256 key and its exact 516-sample red/IR window. It also records the reference
window's clock offsets needed for causal ordering. No patient IDs, ages, sexes,
saturation labels, waveforms, or embeddings are included in this manifest.
`openox_fold0.json` through `openox_fold4.json` preserve the fixed split indices.
The array retains 727 indexing positions, of which 666 are eligible. Unused
positions never contribute to fitting, training, validation or reporting.

SpO₂ is the arithmetic mean of available valid device readings (50–100%) for
the same encounter/sample, with each device counted once. SaO₂ is the median
of finite blood-gas `so2` readings for that encounter/sample. Both use the same
measurement cohort and folds. SaO₂ replicates must differ by at most one point.
Neither reference enters the model inputs or history selection.

Each preparation run fits field-wise PCA means, bases and RMS scales, plus
amplitude normalization, using only each fold's training observations. Target
normalization is likewise fitted on training labels. The generated parameters
are saved locally in `foldN/normalizer.json`; none are distributed in this release.
SpO₂ and SaO₂ use the same freshly fitted input features. Their target scales are
fitted separately. Age/sex embeddings can have low rank, so numerical-library
changes can alter null-space coordinates for ages not observed in training.

Waveforms are resampled 86→50 Hz with `resample_poly(..., padtype='line')` and
processed with the fixed Chebyshev-II 0.5–12 Hz filter. The encoder waveform
is standardized per channel. Mean and log standard deviation are extracted
from the raw window before that normalization. The 125 Hz branch preserves
the six-second duration. Actual band wavelengths are unknown and stay masked.

Only **age and sex** are active condition fields for all current oxygen
baselines and MC-HGNet variants. The four-slot storage shape is preserved for
checkpoint compatibility; skin-type and warming slots are zero/masked.
Current oxygen MC-HGNet uses bounded condition-to-pair modulation with bound
0.1 (the P1 implementation). Removing its graph component removes this
calibration. It does not introduce the glucose/BP band–field graph.

## CRHE

Official source: [CRHE Long-Term PPG-BP Dataset](https://github.com/FDU-Bio-Monitoring/CRHE-Long-Term-PPG-BP-Dataset),
commit `298c6c8c50b0b04062b1c8776d18a61a4eb242e6`.
The loader checks original file hashes before cleaning. It joins green/red/IR
channels at each person/timestamp, resolves identical duplicate rows, and removes
ambiguous multiwave timestamps, inconsistent metadata, missing/nonfinite/constant
channels, logically invalid targets, placeholder dates, and identical waveforms
reused at a different timestamp. This produces 4,178 measurements from 29 people.

The fixed future test is selected from each person's last dates, near 20% of
their records. The remaining development dates form five grouped folds.
A 30-minute boundary embargo excludes two records from development. No target
values are used to choose split boundaries or development folds.

| Fold | Train | Validation | Fixed test |
|---|---:|---:|---:|
| 0 | 2,705 | 683 | 788 |
| 1 | 2,725 | 663 | 788 |
| 2 | 2,715 | 673 | 788 |
| 3 | 2,707 | 681 | 788 |
| 4 | 2,700 | 688 | 788 |

The current manuscript interprets the released 320 samples as **40 Hz / 8 s**.
This is the adopted processed-CSV interpretation. The acquisition description's
100 Hz should not silently replace it when reproducing these experiments.
Processing subtracts a 41-sample moving mean, applies a fifth-order 0.5–8 Hz
Butterworth filter, then scales each channel to [0,1]. Native 50/125 Hz views are
resampled over the same eight seconds and re-scaled. No output zero-padding is
used. Wavelength metadata are 550/660/880 nm; conditions are age and sex.
SBP and DBP are trained jointly and evaluated separately.

History must be from the same person, day and split, at least 300 seconds
earlier, and nonoverlapping under the ±68-second timestamp-support guard.
Only the latest eligible predecessor contributes. Time gaps for CRHE are
timestamp-to-timestamp; OpenOx uses current-start minus historical-end.

## Private glucose interface

The selected glucose cohort is private. This release cannot supply its signal,
labels, patient metadata, condition sentences, or fitted preprocessing values.
The paper uses acquisition groups rather than treating repeated captures as
independent targets. Its six bands are 660/905/1200/1300/1460/1550 nm.

An authorized prepared dataset follows the same directory schema:

* `spec.json`: `dataset="glucose"`, six channels/wavelengths, the recorded field
  slots, `pooling="capture_median"`, `input_samples=300`, `outputs=1`, and
  `late_text_dim=33*number_of_slots`.
* `wave.npy` and `wave50.npy`: standardized record-level waveform arrays
  `[records,6,300]`; `wave125.npy`: `[records,6,750]`.
* `labels.npy`: `[measurements,1]` in mmol/L.
* `units.csv`: measurement `unit,subject,day,start,end`. `end` is acquisition
  completion time. All splits and history selection operate on groups.
* `structure.npz`: `index`, `record_mask`, `owner`, `membership` and
  `capture_mask`. `membership` contains within-capture averaging coefficients;
  captures are median-pooled to the measurement level.
* `foldN/fold.json`: disjoint `train,validation,test` measurement-index lists.
* `foldN/normalizer.json`: train-only `fit_units`, `target_center` and
  `target_scale`, provided by the authorized local data preparation.
* `foldN/inputs.npz`: `fields` `[measurements,slots,33]`, train-normalized
  `statistics` `[measurements,6,2]`, and `index,mask,gap` for history.
  The first 32 field coordinates hold the projected text feature; the last is
  its availability. Historical glucose/reference values are absent.
* SRFormer additionally needs authorized `spectra.npz` with `led` and `pd`
  arrays `[6,3]`, calculated from the verified instrument curves as in its
  original adapter. Public OpenOx/CRHE experiments do not need those curves.

Record windows are averaged within captures, then captures are median pooled.
Do not move individual repeats across splits. Use a completed previous group,
not an earlier repeat inside the current acquisition group, as history.
