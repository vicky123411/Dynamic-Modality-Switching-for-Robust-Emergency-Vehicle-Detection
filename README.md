# Robust Emergency Vehicle Detection with Audio-Visual Late Fusion: A Benchmark and Diagnosis

Code, data splits, trained weights, logs and per-sample predictions for the paper
**"Robust Emergency Vehicle Detection with Audio-Visual Late Fusion: A Benchmark and Diagnosis"**
(Vikash, Yash Yadav, Aman Singh, Madan Lal Saini — IICIT 2027).

Every number, table and figure in the paper can be rebuilt from this repository.
No dataset is redistributed: the scripts download the public data themselves.

> **Note:** this version replaces the earlier YOLOv8 prototype that was in this repository.
> The paper uses a MobileNetV3-Small vision branch and a log-Mel CNN audio branch.
> The old code is still available in the git history (commit `cfa2e73`).

## What the project does

* **Vision branch:** MobileNetV3-Small image classifier (ImageNet initialised).
* **Audio branch:** compact 3-block CNN on 64-band log-Mel spectrograms (4 s window).
* **Fusion:** temperature-scaled confidences combined by 11 late-fusion rules — mean, max,
  noisy-OR, static weighted, learned late fusion, hand-crafted context rules, a **logistic
  reliability gate** and an **MLP reliability gate** (17-d context descriptor, log-odds pooling),
  and an oracle.
* **Benchmarks:** EV-AVSim-L (leakage-free composite image + audio pairs), VGGSound-EV
  (real audio-visual clips) and an out-of-source test (train on ImageNet + UrbanSound8K only,
  test on Open Images + VGGSound after near-duplicate removal).
* **Statistics:** 3 training seeds × 3 pairing seeds, clustered-bootstrap 95% CIs,
  exact McNemar tests with Holm correction (42 tests).

## Main results (AUPRC, mean over 3 seeds)

| Benchmark | Vision | Audio | Noisy-OR | Logistic gate | MLP gate | Oracle |
|---|---|---|---|---|---|---|
| EV-AVSim-L (composite) | 0.853 | 0.786 | 0.917 | 0.919 | 0.920 | 0.985 |
| VGGSound-EV (real clips, in-domain) | 0.642 | 0.969 | 0.973 | 0.974 | 0.970 | 0.996 |
| Out-of-source (real clips) | — | — | 0.905 | 0.892 | 0.891 | — |

Fusion clearly helps; the learned gates give a small, non-significant gain on composite pairs,
none on real clips, and a loss under source shift (though they are better calibrated).
Full system: 1.67 M parameters, 82.6 M MACs.

## Repository layout

```
code/          experiment code
paper/         rebuilds the paper's LaTeX source from the result files
results/       main results: per-seed JSON, corruption sweeps, statistics, predictions, weights
results_ood/   out-of-source test: results, duplicate report, test lists, weights
splits/        exact image / audio / video split lists
logs/, logs_ood/  training and evaluation logs (with timestamps)
```

| File | What it does |
|---|---|
| `code/evcore.py` | Core library: leak-free pairing, corruptions, 17-d reliability descriptors, temperature scaling, logistic/MLP gate, log-odds pooling, all fusion baselines, metrics, McNemar |
| `code/ev_v3.py` | Experiment driver (stages: `index`, `index_av`, `vision`, `audio`, `score`, `eval`, `sweep`) |
| `code/ev_ood.py` | Out-of-source test: retrain on ImageNet + UrbanSound8K only, near-duplicate removal, test on Open Images + VGGSound |
| `code/stats_v4.py` | Clustered-bootstrap 95% CIs (gate − noisy-OR), exact McNemar tests with Holm correction, per-sample prediction CSVs |
| `code/stats_ood.py` | CIs and McNemar tests for the out-of-source test |
| `code/oi_download.py` | Downloads the Open Images V7 images (human-verified positives and hard negatives) |
| `code/inet_get.py` | Extracts the ImageNet class subsets (ambulance, fire engine, police van + 15 ordinary vehicle classes) |
| `code/vgg_stream.py`, `code/vgg_local.sh` | Downloads the VGGSound mirror and keeps only the 10 road/siren classes (frame at 5 s, audio 3–7 s) |
| `code/make_figs_v3.py` | Degradation-curve figure (Fig. 2) from the sweep results |
| `paper/build_v4.py` | Writes every number and table of the paper into `main_v4.tex` from `results/` and `results_ood/` |

## Data

| Source | Items | Positives | Use |
|---|---|---|---|
| Open Images V7 (human-verified image-level labels) | 3,140 | 1,244 | images |
| ImageNet (ILSVRC-2012), 3 EV + 15 vehicle classes | 9,293 | 4,050 | images |
| UrbanSound8K (official folds; 95 slices of recording 180937 dropped because it crosses folds) | 8,637 | 929 | audio |
| VGGSound, 10 classes (3 siren + 7 road/other) | 9,155 | 3,042 | audio + real synchronised test |

* Images: 70/15/15 split stratified by source and label (8,703 / 1,865 / 1,865).
* EV-AVSim-L: 20,000 / 4,000 / 4,000 composite pairs, 5 scenarios, 3 pairing seeds (42, 43, 44).
* VGGSound-EV: video-disjoint split (one split per YouTube ID), 5,487 / 1,834 / 1,834 clips.

## How to run

```bash
pip install -r requirements.txt
cd code
# 1. data (see the download scripts above)
python ev_v3.py index          # image index + composite pairs (asserts no leakage)
python ev_v3.py index_av       # VGGSound-EV + audio index (asserts video-disjoint split)
# 2. train both branches, 3 seeds
for s in 0 1 2; do python ev_v3.py vision --seed $s; python ev_v3.py audio --seed $s; done
# 3. scores, evaluation, corruption sweep
for s in 0 1 2; do python ev_v3.py score --seed $s; python ev_v3.py eval --seed $s; python ev_v3.py sweep --seed $s; done
python stats_v4.py
python make_figs_v3.py degradation_curves_v4.png work/sweep_s0.json work/sweep_s1.json work/sweep_s2.json
# 4. out-of-source test
python ev_ood.py index    # then vision / audio / score / eval for each seed
python stats_ood.py
```

Set `EV_DATA` (default `data/`) and `EV_WORK` (default `work/`) to choose folders.
Everything ran on a 2-core CPU (no GPU).

To rebuild the paper source from the saved results (no training needed):

```bash
python paper/build_v4.py      # writes paper/main_v4.tex
cd paper && pdflatex main_v4.tex && pdflatex main_v4.tex
```

## Settings

* Vision: MobileNetV3-Small, ImageNet init, AdamW lr 3e-4, weight decay 1e-4, batch 32, cosine schedule, ≤12 epochs, early stopping (patience 4), random crop + flip.
* Audio: 3-block CNN on 64-band log-Mel (4 s, 22.05 kHz), lr 2e-3, batch 64, 30 epochs, time/frequency masking + time shift, positive class weight.
* Fusion: temperature scaling on validation; logistic and MLP gate (32 hidden units) on train; bias and all thresholds on validation; test set used once.

## Evidence for every number in the paper

* `results/results_s{0,1,2}.json` — all metrics, thresholds, confusion matrices, gate coefficients (one file per training seed)
* `results/sweep_s{0,1,2}.json` — corruption sweep (Fig. 2)
* `results/stats/auprc_diff_ci95.json`, `results/stats/mcnemar_gate_vs_noisyor.csv` — Table 3
* `results/stats/predictions_*.csv` — per-sample test scores of every method (`y` = ground truth)
* `results/*_history.json`, `logs/*.log` — training and evaluation logs
* `results/vision_s*.pt`, `results/audio_s*.pt` — trained weights
* `results_ood/` — out-of-source results (`results_ood_s*.json`, Table 4), duplicate report, test lists, weights; `logs_ood/` — their logs

## Datasets and licences

Open Images V7, ImageNet (ILSVRC-2012), UrbanSound8K and VGGSound are used under their own
licences and terms; please obtain them from their official sources. `paper/llncs.cls` is the
Springer LNCS class file.
