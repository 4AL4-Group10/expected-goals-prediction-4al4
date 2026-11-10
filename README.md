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

## Final Evaluation on Test Set

Before evaluating the test set, ensure pre processing is complete and the training 
stage is complete. The progress-report/artifacts directory must be populated
accordingly. To run the evaluation:

Default run:

```bash
python progress-report/evaluate.py
```

This tests the trained logistic regression model side by side with all of the 
baselines. The following tables summarize the results:

*Loss Function Results for Trained LR Model and each Baseline*
| Model/Reference | Test Cross-Entropy | Test Brier Score | Test ROC-AUC Score |
| --- | ---: | ---: | ---: |
| Majority class | 3.07199 | 0.111179 | 0.5 |
| Constant training goal probability | 0.348988 | 0.098821 | 0.5|
| Selected logistic regression | 0.270382 | 0.077240 | 0.816711|
| Supplied StatsBomb xG | 0.270069 | 0.076434 | 0.817325|

*Percentage Improvement in Loss for Trained LR Model vs each Baseline*
| Model/Reference | Test Cross-Entropy Improvement (%) | Test Brier Score Improvement (%) |
| --- | ---: | ---: |
| Majority class | 91.1985 | 30.5266 | 
| Constant training goal probability | 22.5239 | 21.8387 | 
| Supplied StatsBomb xG | -0.11598 | -1.05390 |

As can be observed by the cross entropy loss values, the trained LR model vastly outperforms 
the majority and constant probability baselines. It incurs slightly greater loss than the
Supplied StatsBomb xG model. This is reasonable since this model is highly optimized with 
XGBoost techniques instead of our trained logistic regression model. A similar observation
can be made from the percent improvements: the logistic regression model is performing
very slightly worse than the StatsBomb model. The ROC-AUC score indicates that the two 
constant probability baselines are essentially predicting probabilities randomly. The higher
values for this score suggest that for both the other models, 81% of the time they predict
a higher expected goal probability, it is actually turning into a goal. This figure is 
generally good, and since our model is close to the baseline, it further reaffirms it is 
performing well. There are no significant differences in the loss measured during the 
validation and test stages (both validation and test datasets were roughly equal in data 
points), further suggesting that the trained LR model is generalizing quite well.

From the generated calibration curves, the quantile graph suggests that most chances
are assigned probabilities between 0-0.3. Both models demonstrate relatively good 
calibration within this range, with their predictions remaining close to the ideal diagonal.
Therefore the prediction is very close to the observed goal frequency for this range. 
As the probability gets higher, both models become a lot more erratic in the uniform graph 
due to less training data being available for these chances, making the prediction a lot more 
sensitive to individual outcomes in the training data instead of providing a general result. This also 
tells us that there are a lot less extreme danger chances than low to medium 
danger chances in a real game.

Therefore, this trained LR model is quite a powerful model overall. When compared
to an industry standard tool like the StatsBomb xG predictor, it performs nearly as well. 
This is shown by its similar cross entropy loss, Brier Score, ROC AUC Score, and calibration. 

## Saved outputs

After the evaluation completes, `progress-report/eval_results/` contains:

- `test_set_model_evaluations.csv`: all evaluated scores and loss functions for the trained LR and baseline models.
- `lr_improvement_percentage_wrt_baseline.csv`: all improvement percentages for the trained LR model with respect to each baseline.
- `uniform_datapoints_calibration_curve.png`: calibration curves when evaluating the test set with the trained LR model and StatsBomb Baseline compared to a perfect predictor. Uses 25 bins, with bins spaced uniformly across probabilities from 0 to 1. 
- `quantile_datapoints_calibration_curve.png`: calibration curves when evaluating the test set with the trained LR model and StatsBomb Baseline compared to a perfect predictor. Uses 25 bins, with quantile bin spacing. This means the data points were sorted and assigned to evenly sized bins in ascending order.   

## Data credit

Data source: StatsBomb Open Data. Follow the repository's
[attribution and usage terms](https://github.com/hudl/open-data#terms--conditions)
when publishing results, including the requested StatsBomb credit and logo.
