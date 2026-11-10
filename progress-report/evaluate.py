import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from data_processing import OUT_DIR, build_feature_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import os

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

EVAL_RESULTS = Path(__file__).resolve().parent / "eval_results"

LR = "logistic_regression"
MAJORITY = "majority"
CONST_PROB = "constant_probability"
STATSBOMB = "statsbomb"

def recover_test_set():
    # Read processed dataset and recreate matching feature matrices
    df = pd.read_csv(Path(OUT_DIR) / "shots.csv")
    X, y = build_feature_matrix(df)

    # Based on exported splits, recreate the train/test/validation split
    splits = pd.read_csv(ARTIFACTS / "splits.csv")
    split_map = splits.drop_duplicates("match_id").set_index("match_id")["split"]

    # Get just test rows. Also get statsbomb xg to use as another baseline
    test_mask = df["match_id"].map(split_map).eq("test")
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]
    statsbomb_xg = df.loc[test_mask, "statsbomb_xg"]

    return X_test, y_test, statsbomb_xg

def load_model(model_name):
    model_path = ARTIFACTS / f"{model_name}.joblib"
    model = joblib.load(model_path)
    return model


#==========================================
#   CROSS ENTROPY HELPERS AND CALCULATOR  #
#==========================================
def statsbomb_cross_entropy(X_test, y_test,statsbomb_xg):
    return -np.mean(y_test * np.log(statsbomb_xg) + (1 - y_test) * np.log(1 - statsbomb_xg))

# Calculates eval model ce, and all baselines including the statsbomb loss as well
# Returns eval model cross entropy followed by map with all baseline cross entropies and the eval LR model too
# Exports all cross entropies into a csv
def calculate_model_cross_entropies(X_test, y_test, eval_model, baseline_models,statsbomb_xg):
    print("\n============================\n")
    print("Calculating cross entropy...\n")
    baseline_ce = {}
    eval_model_ce = eval_model.cross_entropy(X_test, y_test)
    baseline_ce[LR] = eval_model_ce
    print(f"{LR}: {eval_model_ce}")
    for name, baseline in baseline_models.items():
        baseline_ce_val = baseline.cross_entropy(X_test, y_test)
        baseline_ce[name]  = baseline_ce_val
        print(f"{name} : {baseline_ce_val}")
    statsbomb_ce_val = statsbomb_cross_entropy(X_test, y_test,statsbomb_xg)
    baseline_ce[STATSBOMB] = statsbomb_ce_val
    print(f"{STATSBOMB}: {statsbomb_ce_val}")
    return eval_model_ce, baseline_ce

# Get the cross entropy improvement percentage
# Exports improvement percentages wrt each baseline into a csv
def compare_cross_entropy(X_test, y_test, eval_model, baseline_models, statsbomb_xg):
    eval_model_ce, all_baseline_ce = calculate_model_cross_entropies(X_test, y_test, eval_model, baseline_models,statsbomb_xg)
    to_export = all_baseline_ce.copy()
    percent_improvements = {}
    print(f"\nImprovement (%) with LR wrt Baseline...\n")
    for name, baseline_ce in all_baseline_ce.items():
        percent_improvement = (baseline_ce - eval_model_ce)/baseline_ce
        percent_improvements[name] = percent_improvement
        print(f"Baseline ({name}): {percent_improvement*100}%")
    print("\n============================\n")
    return all_baseline_ce, percent_improvements

#==========================================
#   BRIER SCORE HELPERS AND CALCULATOR    #
#==========================================

def compare_brier_scores(X_test, y_test, eval_model, baseline_models,statsbomb_xg):
    eval_model_brier, brier_scores = calculate_model_brier_scores(X_test, y_test, eval_model, baseline_models,statsbomb_xg)
    percent_improvements = {}
    print(f"\nImprovement (%) with LR wrt Baseline...\n")
    for name, baseline_brier in brier_scores.items():
        percent_improvement = (baseline_brier - eval_model_brier) / baseline_brier
        percent_improvements[name] = percent_improvement
        print(f"Baseline ({name}): {percent_improvement * 100}%")
    print("\n============================\n")
    return brier_scores, percent_improvements

def calculate_model_brier_scores(X_test, y_test, eval_model, baseline_models,statsbomb_xg):
    print("\n============================\n")
    print("Calculating brier scores...\n")
    brier_scores = {}
    eval_model_brier = brier_score(X_test, y_test,eval_model)
    brier_scores[LR] = eval_model_brier
    print(f"{LR}: {eval_model_brier}")
    for name, baseline in baseline_models.items():
        brier_score_val = brier_score(X_test, y_test,baseline)
        brier_scores[name] = brier_score_val
        print(f"{name} : {brier_score_val}")
    statsbomb_brier_score_val = statsbomb_brier_score(X_test, y_test,statsbomb_xg)
    brier_scores[STATSBOMB] = statsbomb_brier_score_val
    print(f"{STATSBOMB} : {statsbomb_brier_score_val}")
    return eval_model_brier, brier_scores

def brier_score(X_test, y_test, model):
    y_pred = model.predict_probability(X_test)
    return np.mean((y_test - y_pred) ** 2)

def statsbomb_brier_score(X_test, y_test,statsbomb_xg):
    return np.mean((y_test - statsbomb_xg) ** 2)

#==========================================
#      MODEL AUC WITH SCIKIT-LEARN        #
#==========================================

def calculate_model_auc(X_test, y_test, eval_model, baseline_models, statsbomb_xg):
    print("\n============================\n")
    print("Calculating ROC-AUC scores...\n")
    auc_scores={}
    # Our LR Model
    lr_auc=roc_auc_score(y_test, eval_model.predict_probability(X_test))
    auc_scores[LR] = lr_auc
    print(f"{LR} : {lr_auc}")
    for name, model in baseline_models.items():
        baseline_auc = roc_auc_score(y_test, model.predict_probability(X_test))
        auc_scores[name] = baseline_auc
        print(f"{name} : {baseline_auc}")
    statsbomb_auc = roc_auc_score(y_test, statsbomb_xg)
    auc_scores[STATSBOMB] = statsbomb_auc
    print(f"{STATSBOMB} : {statsbomb_auc}")
    print("\n============================\n")
    return auc_scores

#==========================================
#   CALIBRATION CURVE WITH SCIKIT-LEARN   #
#==========================================

def plot_calibration_curve(X_test, y_test, lr_model, statsbomb_xg):
    y_pred_lr =  lr_model.predict_probability(X_test)

    # calibration curve data for 25 bins, uniform strategy - evenly spaced probability intervals
    actual_bin_goal_fraction_lr_uni, mean_pred_xg_lr_uni = calibration_curve_general(y_test, y_pred_lr)
    actual_bin_goal_fraction_statsbomb_uni, mean_pred_xg_statsbomb_uni = calibration_curve_general(y_test, statsbomb_xg)

    plt.figure(figsize=(8, 6))

    plt.plot(
        mean_pred_xg_lr_uni,
        actual_bin_goal_fraction_lr_uni,
        marker="o",
        label="Logistic Regression"
    )

    plt.plot(
        mean_pred_xg_statsbomb_uni,
        actual_bin_goal_fraction_statsbomb_uni,
        marker="o",
        label="Statsbomb Baseline"
    )

    # Perfect calibration reference line
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")

    plt.xlabel("Mean Predicted Probability (xG)")
    plt.ylabel("Observed Goal Frequency")
    plt.title("Calibration Curve (25 bins, Uniform Strategy)")
    plt.legend()
    plt.grid(True)
    plt.savefig(EVAL_RESULTS/"uniform_datapoints_calibration_curve.png", dpi=300, bbox_inches="tight")

    # calibration curve for 25 bins, quantile strategy - each bin has ~same number of data points
    actual_bin_goal_fraction_lr_qu, mean_pred_xg_lr_qu = calibration_curve_general(y_test, y_pred_lr,"quantile")
    actual_bin_goal_fraction_statsbomb_qu, mean_pred_xg_statsbomb_qu = calibration_curve_general(y_test, statsbomb_xg,"quantile")

    plt.figure(figsize=(8, 6))

    plt.plot(
        mean_pred_xg_lr_qu,
        actual_bin_goal_fraction_lr_qu,
        marker="o",
        label="Logistic Regression"
    )

    plt.plot(
        mean_pred_xg_statsbomb_qu,
        actual_bin_goal_fraction_statsbomb_qu,
        marker="o",
        label="Statsbomb Baseline"
    )

    # Perfect calibration reference line
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")

    plt.xlabel("Mean Predicted Probability (xG)")
    plt.ylabel("Observed Goal Frequency")
    plt.title("Calibration Curve (25 bins, Quantile Strategy)")
    plt.legend()
    plt.grid(True)
    plt.savefig(EVAL_RESULTS / "quantile_datapoints_calibration_curve.png", dpi=300, bbox_inches="tight")


def calibration_curve_general(y_test, y_pred, strategy="uniform"):
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred, n_bins=25,strategy=strategy)
    return fraction_of_positives, mean_predicted_value

#==========================================
#                   MAIN                  #
#==========================================

def main():

    # Make output directory for all evaluation artifacts
    os.makedirs(EVAL_RESULTS, exist_ok=True)

    # Recover the same test data from the train/test/validation split
    X_test, y_test, statsbomb_xg = recover_test_set()

    # Load the trained model and baselines
    logistic_regression = load_model(LR)
    majority = load_model(MAJORITY)
    constant_probability = load_model(CONST_PROB)
    models={MAJORITY: majority, CONST_PROB: constant_probability}

    # Perform all evaluations
    ce_results, ce_improvements = compare_cross_entropy( X_test, y_test, logistic_regression, models, statsbomb_xg)
    brier_results, brier_improvements = compare_brier_scores(X_test, y_test, logistic_regression, models, statsbomb_xg)
    auc_results = calculate_model_auc(X_test, y_test, logistic_regression, models, statsbomb_xg)
    plot_calibration_curve(X_test, y_test, logistic_regression, statsbomb_xg)

    # Export all loss data and improvements relative to baseline to csv files

    loss_df = pd.DataFrame({
        "Name": ce_results.keys(),
        "Cross Entropy Loss": ce_results.values(),
        "Brier Score": brier_results.values(),
        "ROC-AUC Score": auc_results.values()
    })

    loss_df.to_csv(os.path.join(EVAL_RESULTS, "test_set_model_evaluations.csv"),index=False)

    improvements_df = pd.DataFrame({
        "Baseline Name": ce_improvements.keys(),
        "Cross Entropy Improvement (%) wrt Baseline": ce_improvements.values(),
        "Brier Score Improvement (%) wrt Baseline": brier_improvements.values()
    })

    improvements_df.to_csv(os.path.join(EVAL_RESULTS, "lr_improvement_percentage_wrt_baseline.csv"),index=False)

if __name__ == "__main__":
    main()