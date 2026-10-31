# Current paper protocol

`mchgnet/resources/configs/paper_recipes.json` holds 280 dataset/encoder/variant/fold entries
extracted from the current manuscript's model registry. It contains settings,
not model checkpoints or paths to the original workstation. A baseline and
full model have matched fields, preprocessing, splits and training budgets.
All three removal ablations inherit the full model's recipe.

The seed is 2026. Each epoch traverses every training measurement once, without
replacement, with effective batch size 128 and a final incomplete batch retained.
Microbatch sizes are recorded explicitly. Complete training uses 300 epochs for
glucose, 250 for each oxygen task, and 46 for joint BP. Validation is evaluated
at the recorded update intervals; checkpoints minimize measurement-weighted
validation MAE (mean SBP/DBP MAE for joint BP). The fixed test is evaluated only
after checkpoint selection. The report aggregates each model's metrics, without
averaging its predictions into an ensemble.

ModernTCN, TimeFilter and pretrained-PPG adapters use AdamW, the recorded
cosine learning-rate schedule, gradient clipping, and Smooth L1 regression.
Band and condition auxiliary heads each carry weight 0.25 in the current
recipes. TimeFilter retains its native 0.01 routing auxiliary loss. Pretrained
PPG encoders are frozen during the initial warmup updates and then fine-tuned
at their recorded encoder learning rate. Their BatchNorm running statistics
remain frozen. ModernTCN and TimeFilter are trained from random initialization.

SRFormer retains its native Adam/StepLR training: learning rate 0.001, weight
decay 0, step decay 0.9 every ten epochs, gradient clip 1, and the original
0.9 regression-MSE + 0.1 subject-identity contrastive objective. The archived
SRFormer runs do not use the two M-style auxiliary losses. The released recipes
preserve these actual settings. SRFormer IDs are mapped on training people and
reused; this protocol contains known people at evaluation. Its pretrained
status is random initialization, unlike the four PPG foundation models.

The paper's current full S/BP recipe uses internal PatchTST dropout 0.1;
the corresponding concatenation baseline uses 0. These values are explicitly
recorded in the configuration, including for the S/BP ablations.

All models produce one prediction per measurement. Loss and metrics give each
measurement equal weight. Models receive their native-duration waveform view,
raw-amplitude statistics, and the same recorded condition fields. MC-HGNet also
receives an eligible unlabeled history, so an encoder-adaptation gain evaluates
the complete framework and additional historical input together.

## Modules and removals

* `full`: shared online encoder and projection; module A compares matching
  current/history bands and pairs using Q/K scores, time/condition penalties,
  a learned null option, and difference-value messages. The residual enters
  current band features before module B.
* `no_history`: removes retrieval/encoding of historical measurements and all
  history parameters. The current pair and condition fusion remain.
* `no_pairs`: removes current and historical explicit pair paths and their
  gates. On oxygen this also removes the pair-dependent condition calibration.
* `no_graph`: retains history, explicit pairs and direct late condition context,
  but removes typed band–field graph messages (or oxygen pair calibration).
* `baseline`: current waveform and condition concatenation, with the same
  encoder family and regression recipe.

The latest history is selected independently of physiological labels, within
the same person/day/split. Earlier validation/test observations can serve as
unlabeled history for later observations in that split. Parameters are fixed
through evaluation. Missing eligible history produces an exact zero residual.
