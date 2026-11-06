# Expected Goals (xG) Prediction in Soccer

Predict the probability that a shot results in a goal using StatsBomb shot
location, technique, body part, pressure, and defensive positioning.
The active implementation is in `progress-report/`.

## Project files

- `progress-report/data_processing.py`: combine raw match JSON files into cleaned shot records and construct model features.
- `progress-report/models.py`: manual NumPy logistic regression and majority/constant-probability baselines.
- `progress-report/train.py`: match-based splitting, random hyperparameter search, validation comparison, and saving.
- `requirements.txt`: Python dependencies.

The root-level `data_pipeline.py` and `train_alt.py` are alternative/older scripts;
use the commands below for the progress report.

## Setup

Run commands from the repository root. Python 3.12 was used for the recorded run.

### Ubuntu / WSL

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install -r requirements.txt
```

If virtual-environment creation reports missing `ensurepip`, install the matching
`python3.12-venv` package with your system package manager first.
For later sessions, activate the environment again before running scripts.

### Windows PowerShell

Using an installed Python 3.12 interpreter:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Windows and WSL use separate Python installations and virtual environments.
Use a distinct environment folder if working in both systems.

## Download and preprocess data

Download [StatsBomb Open Data](https://github.com/hudl/open-data) once:

```bash
git clone --depth 1 https://github.com/hudl/open-data.git data/open-data
```

Skip cloning if `data/open-data` already exists. Expected input layout:

```text
data/open-data/data/events/*.json
data/open-data/data/matches/<competition_id>/<season_id>.json
```

Run preprocessing:

```bash
python progress-report/data_processing.py
```

This writes `data/processed/shots.csv` and `pipeline_report.json`.
The recorded dataset contains 4,235 match files and 108,281 raw shots;
363 shootout shots and 7 off-pitch shots were removed, leaving 107,911 shots
and 11,838 goals. There are 1,065 retained shots without a freeze frame.

Training loads this cleaned CSV and calls `build_feature_matrix()` to create
40 numeric/flag/one-hot feature columns and the binary `is_goal` target.
Match/event IDs, outcome, and StatsBomb xG are excluded from model inputs.

## Train and validate

Default run:

```bash
python progress-report/train.py
```

Defaults: 10 random-search trials, 100 epochs per trial, learning rate sampled
uniformly in log space from 0.001 to 0.1, batch size chosen from 256 or 1024,
and seed 42. The implementation runs on the CPU, not the GPU.

The recorded 20-trial experiment used this WSL/Bash command:

```bash
python -u progress-report/train.py \
  --learning-rate-range 0.0001 0.1 \
  --batch-sizes 32 64 128 256 512 1024 \
  --n-trials 20 \
  --epochs 100 \
  2>&1 | tee training_run.log
```

For PowerShell, put the arguments on one line instead of using Bash's `\`.
Use `python progress-report/train.py --help` for available arguments.

Matches are split approximately 70/15/15 into training, validation, and test
sets. All shots from one match stay together. The recorded split contains
75,429 training shots, 16,148 validation shots, and 16,334 test shots.

Logistic regression learns training-only scaling, weights, and bias using
manual mini-batch gradient descent and binary cross-entropy. Validation uses
the model's same manual cross-entropy implementation. No class weighting or
regularization is applied. The lowest validation-loss trial is selected;
test shots remain reserved for final evaluation.

## Preliminary validation results

The recorded run used 20 trials and 100 epochs. Trial 20 was selected with
learning rate **0.011198164932559393** and batch size **32**.
Reported wall-clock runtime was approximately **3 minutes 16 seconds** on the
contributor's laptop; runtime varies by environment.

| Model/reference | Training cross-entropy | Validation cross-entropy |
| --- | ---: | ---: |
| Majority class | 3.025423 | 3.016689 |
| Constant training goal probability | 0.345455 | 0.344793 |
| Selected logistic regression | 0.269440 | 0.266354 |
| Supplied StatsBomb xG | ? | 0.264980 |

Lower loss is better. Logistic regression reduces validation loss by about
22.8% versus constant probability. Its loss is 0.001374 above the supplied
StatsBomb benchmark. These are validation results after model selection,
not final test results or evidence of equivalent performance. StatsBomb's
training-data overlap is unknown. The majority baseline predicts hard
probabilities, clipped to avoid log(0), which explains its large loss.
Blank hyperparameters for baselines mean they are not applicable.

## Saved outputs

After the search completes, `progress-report/artifacts/` contains:

- `logistic_regression.joblib`: selected fitted model, including scaling and loss history.
- `majority.joblib`, `constant_probability.joblib`: fitted baseline models.
- `search_results.csv`: all trial settings and training/validation losses, sorted by validation loss.
- `baseline_comparison.csv`: baseline, selected model, and StatsBomb comparisons.
- `splits.csv`: one train/validation/test assignment per match.
- `training_config.json`: selected settings and ordered feature names.
- `training_loss.png`: selected model's training cross-entropy by epoch (when Matplotlib is available).

Runs use the same output directory by default and overwrite matching outputs.
Use `--output progress-report/artifacts/experiment_name` to keep runs separate.

For evaluation, make `progress-report/` importable before loading the custom
models with `joblib`. Feed raw features from `build_feature_matrix()` in the
saved feature order to `model.predict_probability(X)`; scaling is internal.
`model.predict(X, threshold=0.5)` produces binary labels. Tune thresholds on
validation data only and reserve test data for final assessment.

Downloaded/processed data, virtual environments, logs, and generated artifacts
are excluded by `.gitignore`; rerun the commands to regenerate them.
The result summary above remains in version control.

## Data credit

Data source: StatsBomb Open Data. Follow the repository's
[attribution and usage terms](https://github.com/hudl/open-data#terms--conditions)
when publishing results, including the requested StatsBomb credit and logo.
