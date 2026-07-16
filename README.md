# GraFPrint: Audio Identification with Graph Neural Networks

Code for **GraFPrint** (A. Bhattacharjee, S. Singh, E. Benetos, *"GraFPrint: A
GNN-based Approach for Audio Identification"*, IEEE ICASSP 2025) and the audio
fingerprinting experiments of thesis Chapter 5. This repository is a fork of
the official implementation
([chymaera96/GraFP](https://github.com/chymaera96/GraFP)) used for the thesis
experiments.

GraFPrint learns compact, degradation-robust audio fingerprints by treating
time-frequency points of a spectrogram as graph nodes: a k-NN graph over local
spectral features is processed by a GNN encoder, and the network is trained
with a SimCLR-style contrastive objective (NT-Xent loss) on pairs of clean and
augmented (noise + room impulse response) audio segments. At retrieval time,
fingerprints are matched against a reference database with FAISS.

## Repository layout

```
config/            YAML configs (grafp.yaml = main model; nfp/resnet = baselines)
encoder/           GraFPrint GNN encoder (graph_encoder.py, gcn_lib/) and baselines
modules/           dataset (data.py) and GPU audio augmentations (transformations.py)
simclr/            SimCLR wrapper and NT-Xent loss
train.py           contrastive training
eval.py            FAISS retrieval evaluation (top-1 hit rate)
test_fp.py         fingerprint database creation + retrieval test
peak_extractor.py  spectral peak-based front-end utilities
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

The experiments use the [FMA dataset](https://github.com/mdeff/fma) (Creative
Commons audio), [MUSAN](https://www.openslr.org/17/) noise, and the
[Aachen Impulse Response database](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/)
for reverberation:

- **Training**: `fma_small` (8,000 tracks, 30 s clips)
- **Reference / evaluation database**: `fma_medium` (25,000 tracks)
- **Augmentation**: MUSAN noise (train/test splits) + AIR impulse responses

Update the paths at the top of `config/grafp.yaml` (`train_dir`, `val_dir`,
`ir_dir`, `noise_dir`) to your local dataset locations. JSON file lists for the
FMA subsets are created under `data/`.

## Training

```bash
python train.py --config config/grafp.yaml --ckp grafp_run1 \
    --train_dir /path/to/fma_small --val_dir /path/to/fma_medium
```

Key settings (see `config/grafp.yaml`): 1 s segments at 16 kHz (64-band
log-mel, `n_fft=1024`, hop 512), batch size 128, Adam with lr 8e-5 and cosine
annealing to 7e-7, temperature tau = 0.05, 400 epochs, SNR 0-20 dB noise and
impulse-response augmentation applied on GPU. Checkpoints are written to
`checkpoint/`.

## Evaluation (top-1 hit rate)

Build the fingerprint databases and run FAISS retrieval over query lengths and
degradation conditions:

```bash
python test_fp.py --config config/grafp.yaml --test_config config/test_config.yaml \
    --test_dir data/fma_medium.json --encoder grafp \
    --query_lens 1,2,3,5,10 --n_query_db 2000
```

`eval.py::eval_faiss` reports top-1 hit rate (%) at segment and sequence level.
The thesis Table 5.2 protocol uses the fma-medium reference database with
queries degraded by additive noise (0-20 dB SNR) and noise + reverb.

## Results (thesis Table 5.2, summary)

Top-1 hit rate (%) on the fma-medium reference database (25K tracks), 1 s
queries:

| Method | Noise 0dB | Noise 10dB | Noise+Reverb 0dB | Noise+Reverb 10dB |
|---|---|---|---|---|
| NAF (CNN) | 50.7 | 73.7 | 20.8 | 55.5 |
| TAF + LSH (Transformer) | 66.6 | 87.6 | 44.8 | 73.8 |
| GraFPrint | **63.9** | **93.5** | **52.3** | **79.2** |

With 2 s queries GraFPrint reaches 85.7 / 98.6 (noise 0/10 dB) and 80.0 / 94.7
(noise+reverb 0/10 dB). The full table across query lengths (1-5 s), SNRs
(0-20 dB) and the fma-large (106K tracks) scaling experiment is in the
paper/thesis; regenerate with the evaluation command above.

## Citation

```bibtex
@inproceedings{bhattacharjee2025grafprint,
  title     = {GraFPrint: A GNN-based Approach for Audio Identification},
  author    = {Bhattacharjee, Aditya and Singh, Shubhr and Benetos, Emmanouil},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2025}
}
```
