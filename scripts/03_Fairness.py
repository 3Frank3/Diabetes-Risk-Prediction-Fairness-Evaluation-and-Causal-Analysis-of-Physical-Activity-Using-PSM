from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path("/Users/zansong/Desktop/203C/Project")
DATA_PATH = PROJECT_DIR / "archive-6" / "diabetes_binary_health_indicators_BRFSS2015.csv"
PLOTS_DIR = PROJECT_DIR / "outputs" / "plots"
TABLES_DIR = PROJECT_DIR / "outputs" / "tables"

BLUE = "#8FBCD4"
RED = "#E26D5A"
PURPLE = "#B07AA1"
GREEN = "#5B8E7D"
TEXT_GREY = "#444444"
GRID_GREY = "#D0D0D0"

TARGET = "Diabetes_binary"
FAIRNESS_FEATURES = ["Age", "Sex", "Income", "Education"]
GROUP_SPECS = [
    ("Sex", "Sex_fair", ["F", "M"]),
    ("Age", "Age_group_fair", ["18-44", "45-64", "65+"]),
    ("Income", "Income_group_fair", ["Low", "Middle", "High"]),
    ("Education", "Education_group_fair", ["HS or less", "Some college", "College grad"]),
]


def ensure_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["axes.edgecolor"] = GRID_GREY
    plt.rcParams["grid.color"] = GRID_GREY
    plt.rcParams["axes.labelcolor"] = TEXT_GREY
    plt.rcParams["text.color"] = TEXT_GREY
    plt.rcParams["axes.titlecolor"] = TEXT_GREY
    plt.rcParams["xtick.color"] = TEXT_GREY
    plt.rcParams["ytick.color"] = TEXT_GREY


def soft_red_cmap():
    return LinearSegmentedColormap.from_list("soft_red", ["#F8FBFD", "#F4CDC5", RED])


def soft_blue_cmap():
    return LinearSegmentedColormap.from_list("soft_blue", ["#F8FBFD", "#D9EAF7", BLUE])


def soft_diverging_cmap():
    return LinearSegmentedColormap.from_list("soft_diverging", [BLUE, "#F8FBFD", RED])


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def choose_model():
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        return "XGBoost", model
    except Exception:
        model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        return "RandomForestFallback", model


def build_fairness_frame(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    fairness_df = X_test[FAIRNESS_FEATURES].copy().reset_index(drop=True)
    fairness_df["y_true"] = y_test.reset_index(drop=True)
    fairness_df["y_pred"] = pd.Series(y_pred).reset_index(drop=True)
    fairness_df["y_prob"] = pd.Series(y_prob).reset_index(drop=True)

    fairness_df["Sex_fair"] = fairness_df["Sex"].map({0: "F", 1: "M"})
    fairness_df["Age_group_fair"] = fairness_df["Age"].map(
        lambda x: "18-44" if x in [1, 2, 3, 4, 5] else ("45-64" if x in [6, 7, 8, 9] else "65+")
    )
    fairness_df["Income_group_fair"] = fairness_df["Income"].map(
        lambda x: "Low" if x in [1, 2, 3, 4] else ("Middle" if x in [5, 6, 7] else "High")
    )
    fairness_df["Education_group_fair"] = fairness_df["Education"].map(
        lambda x: "HS or less" if x in [1, 2, 3, 4] else ("Some college" if x == 5 else "College grad")
    )
    return fairness_df


def subgroup_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    y_true = frame["y_true"]
    y_pred = frame["y_pred"]
    y_prob = frame["y_prob"]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, y_prob) if y_true.nunique() > 1 else np.nan
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prevalence = y_true.mean()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    return {
        "n": len(frame),
        "Prevalence": prevalence,
        "AUC": auc,
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "FPR": fpr,
        "FNR": fnr,
    }


def fairness_tables(fairness_df: pd.DataFrame):
    overall = subgroup_metrics(fairness_df)
    rows: List[Dict[str, object]] = [{"Group": "Overall", "Subgroup": "All", **overall}]

    for group_name, col, order in GROUP_SPECS:
        for subgroup in order:
            sub = fairness_df.loc[fairness_df[col] == subgroup].copy()
            rows.append({"Group": group_name, "Subgroup": subgroup, **subgroup_metrics(sub)})

    fairness_metrics = pd.DataFrame(rows)
    metric_cols = ["Prevalence", "AUC", "Recall", "Precision", "F1", "FPR", "FNR"]
    fairness_metrics[metric_cols] = fairness_metrics[metric_cols].round(3)

    prevalence_table = fairness_metrics[["Group", "Subgroup", "n", "Prevalence"]].copy()
    metrics_table = fairness_metrics[["Group", "Subgroup", "AUC", "Recall", "Precision", "F1", "FPR", "FNR"]].copy()

    overall_metrics = fairness_metrics.loc[
        fairness_metrics["Group"] == "Overall", ["AUC", "Recall", "Precision", "F1", "FPR", "FNR"]
    ].iloc[0]
    gap_table = fairness_metrics.loc[
        fairness_metrics["Group"] != "Overall", ["Group", "Subgroup", "AUC", "Recall", "Precision", "F1", "FPR", "FNR"]
    ].copy()
    for metric in ["AUC", "Recall", "Precision", "F1", "FPR", "FNR"]:
        gap_table[f"{metric}_gap"] = (gap_table[metric] - overall_metrics[metric]).round(3)

    return fairness_metrics, prevalence_table, metrics_table, gap_table


def export_tables(model_name: str, prevalence_table: pd.DataFrame, metrics_table: pd.DataFrame, gap_table: pd.DataFrame) -> None:
    pd.DataFrame([{"Model used": model_name}]).to_csv(TABLES_DIR / "fairness_run_metadata.csv", index=False)
    prevalence_table.to_csv(TABLES_DIR / "fairness_subgroup_prevalence.csv", index=False)
    metrics_table.to_csv(TABLES_DIR / "fairness_subgroup_metrics.csv", index=False)
    gap_table.to_csv(TABLES_DIR / "fairness_subgroup_gaps.csv", index=False)


def export_plots(fairness_metrics: pd.DataFrame) -> None:
    plot_df = fairness_metrics.loc[fairness_metrics["Group"] != "Overall"].copy()
    plot_df["Label"] = plot_df["Group"] + " | " + plot_df["Subgroup"]

    recall_fnr = plot_df.set_index("Label")[["Recall", "FNR"]]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.heatmap(
        recall_fnr,
        annot=True,
        fmt=".3f",
        cmap=soft_red_cmap(),
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 10},
        ax=ax,
    )
    ax.set_title("Fairness Heatmap: Recall and FNR")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Subgroup")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fairness_recall_fnr_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    auc_precision = plot_df.set_index("Label")[["AUC", "Precision"]]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.heatmap(
        auc_precision,
        annot=True,
        fmt=".3f",
        cmap=soft_blue_cmap(),
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 10},
        ax=ax,
    )
    ax.set_title("Fairness Heatmap: AUC and Precision")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Subgroup")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fairness_auc_precision_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    gap_df = plot_df[["Label"]].copy()
    overall = fairness_metrics.loc[fairness_metrics["Group"] == "Overall"].iloc[0]
    for metric in ["Recall", "FNR", "AUC", "Precision"]:
        gap_df[f"{metric}_gap"] = (plot_df[metric] - overall[metric]).round(3)
    gap_df = gap_df.set_index("Label")

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    sns.heatmap(
        gap_df,
        annot=True,
        fmt=".3f",
        cmap=soft_diverging_cmap(),
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 10},
        ax=ax,
    )
    ax.set_title("Fairness Gap Heatmap vs Overall Model")
    ax.set_xlabel("Gap metric")
    ax.set_ylabel("Subgroup")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fairness_gap_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    sns.barplot(
        data=plot_df,
        x="Label",
        y="FNR",
        hue="Group",
        palette=[BLUE, RED, PURPLE, GREEN],
        dodge=False,
        ax=ax,
    )
    ax.set_title("False Negative Rate by Subgroup")
    ax.set_xlabel("Subgroup")
    ax.set_ylabel("FNR")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Group", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fairness_fnr_barplot.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_fairness_pipeline() -> Dict[str, object]:
    ensure_dirs()
    set_style()

    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model_name, model = choose_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    fairness_df = build_fairness_frame(X_test, y_test, y_pred, y_prob)
    fairness_metrics, prevalence_table, metrics_table, gap_table = fairness_tables(fairness_df)

    export_tables(model_name, prevalence_table, metrics_table, gap_table)
    export_plots(fairness_metrics)

    return {
        "model_name": model_name,
        "fairness_df": fairness_df,
        "fairness_metrics": fairness_metrics,
        "prevalence_table": prevalence_table,
        "metrics_table": metrics_table,
        "gap_table": gap_table,
        "plot_paths": [
            PLOTS_DIR / "fairness_recall_fnr_heatmap.png",
            PLOTS_DIR / "fairness_auc_precision_heatmap.png",
            PLOTS_DIR / "fairness_gap_heatmap.png",
            PLOTS_DIR / "fairness_fnr_barplot.png",
        ],
        "table_paths": [
            TABLES_DIR / "fairness_run_metadata.csv",
            TABLES_DIR / "fairness_subgroup_prevalence.csv",
            TABLES_DIR / "fairness_subgroup_metrics.csv",
            TABLES_DIR / "fairness_subgroup_gaps.csv",
        ],
    }


def main() -> None:
    artifacts = run_fairness_pipeline()
    print(f"Model used: {artifacts['model_name']}")
    print("Saved tables:")
    for p in artifacts["table_paths"]:
        print(f"  - {p}")
    print("Saved plots:")
    for p in artifacts["plot_paths"]:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
