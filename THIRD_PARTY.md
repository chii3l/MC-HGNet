# Third-party components

MC-HGNet code is MIT licensed. This does not relicense downloaded third-party
implementations, pretrained weights or datasets. The downloader obtains these
separately; none of their weights or complete code repositories are bundled.

| Asset | Upstream | Pinned source license |
|---|---|---|
| ModernTCN | https://github.com/luodhhh/ModernTCN | MIT |
| TimeFilter implementation | https://github.com/thuml/Time-Series-Library | MIT |
| PaPaGei | https://github.com/Nokia-Bell-Labs/papagei-foundation-model | BSD-3-Clause |
| Pulse-PPG | https://github.com/maxxu05/pulseppg | MIT |
| AnyPPG | https://github.com/Ngk03/AnyPPG | No top-level license file found in the pinned snapshot |
| SIGMA-PPG | https://github.com/ZonghengGuo/SigmaPPG | No top-level license file found in the pinned snapshot |
| SRFormer | https://github.com/chii3l/SRFormer | No top-level license file found in the pinned snapshot |
| Qwen3-Embedding-0.6B | https://huggingface.co/Qwen/Qwen3-Embedding-0.6B | See the pinned model card and upstream license |

Exact commits are in `mchgnet/fetch.py`. Missing top-level licenses are not
treated as MIT grants. Consult the respective upstream terms for reuse of
those assets. SRFormer includes FEDformer and PatchTST components; their
original headers are preserved when downloaded.

The small deterministic TimeFilter gate adaptation in `timefilter_gate.py`
derives from the MIT-licensed Time-Series-Library implementation. Its full
license is retained under `licenses/`. SRFormer's Fourier execution adapters
implement the published FEDformer arithmetic while preserving native mode
placement and complex activations; original upstream notices are retained.

Dataset licenses are separate. OpenOx is obtained through PhysioNet access
controls. CRHE is fetched from its official repository. The private glucose
cohort is not redistributed. All downloaded assets, caches and outputs are
excluded by `.gitignore`.
