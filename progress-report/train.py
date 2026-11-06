"""Train baselines and logistic regression; reserve test matches for evaluation."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_processing import FLAG_COLUMNS, CATEGORY_OPTIONS, NUMBER_COLUMNS, OUT_DIR, build_feature_matrix
from models import LogisticRegression, make_baselines

HERE = Path(__file__).resolve().parent

def load_data(path):
    df = pd.read_csv(path)
    required = set(NUMBER_COLUMNS + FLAG_COLUMNS + list(CATEGORY_OPTIONS) + ["match_id", "event_id", "is_goal", "statsbomb_xg"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    
    X, y = build_feature_matrix(df)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Features contain missing or infinite values; rerun preprocessing.")
    return df, X, y

def split_data(df, seed):
    # Split unique match IDs, then assign all their shots together.
    match_ids = np.sort(df.match_id.unique())
    train_matches, holdout_matches = train_test_split(match_ids, test_size=0.30, random_state=seed)
    validation_matches, test_matches = train_test_split(holdout_matches, test_size=0.50, random_state=seed)
    labels = pd.Series("train", index=df.index)
    labels.loc[df.match_id.isin(validation_matches)] = "validation"
    labels.loc[df.match_id.isin(test_matches)] = "test"

    indices = {name: np.flatnonzero(labels.eq(name)) for name in ["train", "validation", "test"]}
    return labels, indices

# 1. Prepare training and validation inputs (test data remains reserved).
def prepare_data(data_path, seed):
    df, X, y = load_data(data_path)
    labels, indices = split_data(df, seed)
    train, validation = indices["train"], indices["validation"]
    return df, X, y, labels, indices, (X.iloc[train], y.iloc[train]), (X.iloc[validation], y.iloc[validation])

# 2. Random search, following Tutorial 3's trial records and model dictionary. Find best model
def search_hyperparameters(X_train, y_train, X_val, y_val, n_trials, learning_rate_range, batch_sizes, epochs, seed):
    search_rng = np.random.default_rng(seed)
    trial_records, trial_models = [], {}
    low, high = np.log10(learning_rate_range)

    for trial_id in range(1, n_trials + 1):
        lr = float(10 ** search_rng.uniform(low, high))
        batch_size = int(search_rng.choice(batch_sizes))

        model = LogisticRegression(lr, epochs, batch_size, seed)
        model.fit(X_train, y_train)

        train_loss = model.cross_entropy(X_train, y_train)
        validation_loss = model.cross_entropy(X_val, y_val)

        trial_models[trial_id] = model
        trial_records.append({
            "trial_id": trial_id, 
            "learning_rate": lr,
            "batch_size": batch_size, 
            "epochs": epochs, 
            "seed": seed,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        })

        print(f"Trial {trial_id}/{n_trials}: lr={lr:.6g}, batch={batch_size}, "
              f"train={train_loss:.5f}, validation={validation_loss:.5f}", flush=True)

    search_results = pd.DataFrame(trial_records)
    leaderboard = search_results.sort_values("validation_loss").reset_index(drop=True)
    print(leaderboard.to_string(index=False))
    best_trial = leaderboard.iloc[0]
    best_trial_id = int(best_trial["trial_id"])
    best_search_model = trial_models[best_trial_id]
    print("Selected trial:", best_trial_id)
    return best_search_model, leaderboard

# 4. Fit reference baselines and compare the selected model on both splits.
def compare_baselines(best, X_train, y_train, X_val, y_val, statsbomb_xg):
    models = make_baselines()
    for model in models.values():
        model.fit(X_train, y_train)
    models["logistic_regression"] = best
    records = []
    for name, model in models.items():
        for split, X_split, y_split in [("train", X_train, y_train), ("validation", X_val, y_val)]:
            records.append({
                "model": name, "split": split,
                "learning_rate": best.learning_rate if name == "logistic_regression" else None,
                "batch_size": best.batch_size if name == "logistic_regression" else None,
                "cross_entropy": model.cross_entropy(X_split, y_split),
            })
    # External benchmark only; its training matches may overlap this dataset.
    p = np.clip(np.asarray(statsbomb_xg, dtype=float), 1e-12, 1 - 1e-12)
    target = np.asarray(y_val)
    records.append({
        "model": "statsbomb_xg", "split": "validation",
        "learning_rate": None, "batch_size": None,
        "cross_entropy": -np.mean(target * np.log(p) + (1 - target) * np.log(1 - p)),
    })
    return models, pd.DataFrame(records)

# 5. Save reproducible outputs for the evaluation/report handoff.
def save_results(output, models, leaderboard, df, labels, features, results):
    output.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, output / f"{name}.joblib")

    leaderboard.to_csv(output / "search_results.csv", index=False)
    results.to_csv(output / "baseline_comparison.csv", index=False)
    splits = df[["match_id"]].assign(split=labels).drop_duplicates()
    splits.to_csv(output / "splits.csv", index=False)

    best = models["logistic_regression"]
    config = {
        "learning_rate": best.learning_rate,
        "batch_size": best.batch_size,
        "epochs": best.epochs,
        "seed": best.seed,
        "features": list(features),
    }
    with open(output / "training_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


# 6. Plot the selected model's learning curve, as in Tutorial 3.
def plot_training_loss(model, output):
    try:
        import matplotlib
    except ModuleNotFoundError:
        print("Plot skipped: install matplotlib from requirements.txt.")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(model.loss_history) + 1), model.loss_history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training cross-entropy")
    ax.set_title("Logistic regression training loss")
    fig.tight_layout()
    fig.savefig(output / "training_loss.png", dpi=150)
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(OUT_DIR) / "shots.csv")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--learning-rate-range", type=float, nargs=2, default=[0.001, 0.1], metavar=("MIN", "MAX"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if (any(not np.isfinite(lr) or lr <= 0 for lr in args.learning_rate_range)
            or args.learning_rate_range[0] > args.learning_rate_range[1] or args.n_trials <= 0
            or any(size <= 0 for size in args.batch_sizes) or args.epochs <= 0):
        parser.error("Use positive trials, batch sizes, epochs, and an ordered positive learning-rate range.")
    if not args.data.is_file():
        parser.error(f"Dataset not found: {args.data}. Run data_processing.py or pass --data.")

    return args

def main():
    args = parse_args()
    df, X, y, labels, indices, training, validation = prepare_data(args.data, args.seed)
    X_train, y_train = training
    X_val, y_val = validation
    best, leaderboard = search_hyperparameters(X_train, y_train, X_val, y_val, args.n_trials, args.learning_rate_range, args.batch_sizes, args.epochs, args.seed)
    statsbomb_xg = df.iloc[indices["validation"]]["statsbomb_xg"]
    models, results = compare_baselines(best, X_train, y_train, X_val, y_val, statsbomb_xg)
    save_results(args.output, models, leaderboard, df, labels, X.columns, results)
    plot_training_loss(best, args.output)
    print(results.to_string(index=False))
    print(f"Selected lr={best.learning_rate}, batch={best.batch_size}; "
          f"outputs saved to {args.output.resolve()}")

if __name__ == "__main__":
    main()