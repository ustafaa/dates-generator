# Dates Generator -- Research-Backed Submission

Conditional date generation given (DOW, MON, LEAP, DEC) conditions, with four
research-backed generative models trained from scratch on a Google Colab GPU
runtime.

## Run on Google Colab (training, evaluation, full walkthrough)

1. Open `notebook.ipynb` on Google Colab (File -> Upload notebook, or pull
   from the private repo).
2. Upload the rest of the submission folder as a zip and extract it to
   `/content/submission`. The notebook's setup cell expects that path.
   (Alternative: clone the private repo with a PAT in a setup cell.)
3. Runtime -> Change runtime type -> **GPU (T4)**.
4. Run all cells. The notebook detects Colab, `!pip install -r requirements.txt`,
   trains all four models (~60-90 min total on T4), evaluates, runs the CFG
   ablation, and writes a `predictions.txt` from the example input.

## Run inference locally (matches assignment CLI)

    conda env create -f environment.yml
    conda activate dates-generator
    cd model
    python predict.py -i ../data/example_input.txt -o ../predictions.txt

Pick a different model with `--model {cvae,acgan,maskgit,mdlm}`. Default: `cvae`.

## Retrain from scratch

    python model/train.py --data data/data.txt --out model/weights \
        --models cvae acgan maskgit mdlm

## Run tests

    pytest

## Layout

- `model/predict.py` -- CLI entry point
- `model/train.py` -- training orchestrator
- `model/training.py` -- per-model trainers
- `model/tokenizer.py`, `model/data.py`, `model/metrics.py`, `model/evaluation.py`
- `model/models/{cvae,acgan,maskgit,mdlm}.py` -- the four model implementations
- `model/weights/{cvae,acgan,maskgit,mdlm}.pt` -- trained weights
- `model/weights_legacy/` -- previous submission's weights (kept for comparison)
- `tests/` -- pytest tests
- `notebook.ipynb` -- Colab walkthrough
- `report/` -- assignment write-up
- `scripts/run_final_eval.py`, `scripts/run_cfg_ablation.py`
- `results/` -- CSR tables, diversity, figures

## Models

| # | Model | Class | Citation |
|---|---|---|---|
| 1 | CVAE + CFG dropout | In-course | Sohn 2015; Ho & Salimans 2022 |
| 2 | AC-GAN hybrid MLE + WGAN-GP + projection-D | In-course (required GAN) | Odena 2017; Che 2017; Gulrajani 2017; Miyato/Koyama 2018 |
| 3 | MaskGIT | Out-of-course | Chang et al. CVPR 2022 |
| 4 | MDLM + CFG + joint head | Out-of-course | Sahoo et al. NeurIPS 2024 |

See `docs/superpowers/specs/2026-05-17-dates-generator-research-backed-redesign-design.md`
for the design rationale.

## Reproducibility

Seed 0 everywhere (torch, numpy, random). Dates range [1-1-1800, 31-12-2200]
per the assignment constraint.
