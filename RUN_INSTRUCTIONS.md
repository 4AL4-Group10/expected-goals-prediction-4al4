# Run instructions

Run these commands from the folder containing the submitted files. Python 3.12 is recommended.

Keep `data_processing.py`, `models.py`, `train.py`, `evaluate.py`, and `requirements.txt` together in that folder. The scripts expect `data/` beside this folder.

## 1. Install dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, create and activate the environment with `py -3.12 -m venv .venv` and `.\.venv\Scripts\Activate.ps1` instead.

## 2. Download the data

```bash
mkdir -p ../data
git clone --depth 1 https://github.com/hudl/open-data.git ../data/open-data
```

On Windows PowerShell, use `New-Item -ItemType Directory -Force ../data` instead of `mkdir -p ../data`. Skip cloning if `../data/open-data` already exists.

## 3. Preprocess, train, and evaluate

```bash
python data_processing.py
python train.py --learning-rate-range 0.0001 0.1 --batch-sizes 32 64 128 256 512 1024 --n-trials 20 --epochs 100
python evaluate.py
```

The training command searches 20 combinations of learning rate and batch size, trains each for 100 epochs, and selects the lowest validation loss. It uses the default random seed of 42. Run evaluation after training finishes.

Preprocessing writes `../data/processed/shots.csv`. Training saves models and results in `artifacts/`. Evaluation saves test results and plots in `eval_results/`.
