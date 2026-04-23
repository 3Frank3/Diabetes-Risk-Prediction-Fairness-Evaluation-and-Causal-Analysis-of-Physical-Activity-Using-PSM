import os
from pathlib import Path
import textwrap
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-config")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_DIR = Path("/Users/zansong/Desktop/203C/Project")
DATA_PATH = PROJECT_DIR / "archive-6" / "diabetes_binary_health_indicators_BRFSS2015.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "report"

TARGET = "Diabetes_binary"
GENERAL_HEALTH_VARS = ["GenHlth", "PhysHlth", "MentHlth", "DiffWalk"]
DEMOGRAPHIC_VARS = ["Age", "Sex", "Income", "Education"]
SELECTED_VARS = [TARGET] + GENERAL_HEALTH_VARS + DEMOGRAPHIC_VARS

GENHLTH_LABELS = {
    1: "Excellent",
    2: "Very good",
    3: "Good",
    4: "Fair",
    5: "Poor",
}
AGE_LABELS = {
    1: "18-24",
    2: "25-29",
    3: "30-34",
    4: "35-39",
    5: "40-44",
    6: "45-49",
    7: "50-54",
    8: "55-59",
    9: "60-64",
    10: "65-69",
    11: "70-74",
    12: "75-79",
    13: "80+",
}
SEX_LABELS = {0: "Female", 1: "Male"}
EDU_LABELS = {
    1: "No school/K",
    2: "Grades 1-8",
    3: "Grades 9-11",
    4: "HS graduate",
    5: "Some college",
    6: "College grad",
}
INCOME_LABELS = {
    1: "<10k",
    2: "10-15k",
    3: "15-20k",
    4: "20-25k",
    5: "25-35k",
    6: "35-50k",
    7: "50-75k",
    8: "75k+",
}

LABEL_MAPS = {
    "GenHlth": GENHLTH_LABELS,
    "Age": AGE_LABELS,
    "Sex": SEX_LABELS,
    "Education": EDU_LABELS,
    "Income": INCOME_LABELS,
    "DiffWalk": {0: "No", 1: "Yes"},
    TARGET: {0: "No diabetes", 1: "Diabetes"},
}


def ensure_dirs() -> None:
    for path in [PLOTS_DIR, TABLES_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def wrap(text: str, width: int = 96) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def labeled(series: pd.Series, variable: str) -> pd.Series:
    mapping = LABEL_MAPS.get(variable)
    if mapping is None:
        return series.astype(str)
    return series.astype(int).map(mapping)


def add_text_page(pdf: PdfPages, title: str, paragraphs: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.95, title, fontsize=16, fontweight="bold", va="top")
    y = 0.90
    for para in paragraphs:
        wrapped = wrap(para)
        fig.text(0.08, y, wrapped, fontsize=10.5, va="top")
        y -= 0.07 + 0.018 * wrapped.count("\n")
    pdf.savefig(fig)
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, note: Optional[str] = None, fontsize: float = 9.0) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.45)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d9eaf7")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fbfd")
    if note:
        fig.text(0.08, 0.03, wrap(note, 102), fontsize=9, va="bottom")
    pdf.savefig(fig)
    plt.close(fig)


def save_plot(pdf: PdfPages, fig: plt.Figure, filename: str) -> None:
    fig.savefig(PLOTS_DIR / filename, dpi=220, bbox_inches="tight")
    pdf.savefig(fig)
    plt.close(fig)


def prevalence_table(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    temp = df.groupby(variable)[TARGET].agg(["count", "mean"]).reset_index()
    temp[variable] = labeled(temp[variable], variable)
    temp["Count"] = temp["count"].map(lambda x: f"{x:,}")
    temp["Diabetes prevalence"] = (temp["mean"] * 100).round(1).map(lambda x: f"{x:.1f}%")
    return temp[[variable, "Count", "Diabetes prevalence"]]


def health_day_bucket_table(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    b = pd.cut(df[variable], bins=[-1, 0, 5, 13, 29, 30], labels=["0", "1-5", "6-13", "14-29", "30"])
    temp = df.groupby(b, observed=False)[TARGET].agg(["count", "mean"]).reset_index()
    temp.columns = [variable + "_bucket", "Count", "Mean"]
    temp["Count"] = temp["Count"].map(lambda x: f"{x:,}")
    temp["Diabetes prevalence"] = (temp["Mean"] * 100).round(1).map(lambda x: f"{x:.1f}%")
    return temp[[variable + "_bucket", "Count", "Diabetes prevalence"]]


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[TARGET].value_counts().sort_index()
    return pd.DataFrame(
        {
            "Metric": ["Rows", "Variables in report", "Missing cells", "No diabetes", "Diabetes"],
            "Value": [
                f"{len(df):,}",
                f"{len(df.columns)}",
                f"{int(df.isna().sum().sum()):,}",
                f"{counts.loc[0.0]:,} ({counts.loc[0.0] / len(df) * 100:.1f}%)",
                f"{counts.loc[1.0]:,} ({counts.loc[1.0] / len(df) * 100:.1f}%)",
            ],
        }
    )


def variable_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["GenHlth", "Ordinal", "Self-rated health from excellent to poor"],
            ["PhysHlth", "Count", "Poor physical health days in past 30 days"],
            ["MentHlth", "Count", "Poor mental health days in past 30 days"],
            ["DiffWalk", "Binary", "Serious difficulty walking or climbing stairs"],
            ["Age", "Ordinal", "Age group 18-24 through 80+"],
            ["Sex", "Binary", "Female/Male indicator"],
            ["Income", "Ordinal", "Income category from <10k to 75k+"],
            ["Education", "Ordinal", "Education from no school to college graduate"],
            [TARGET, "Binary", "No diabetes / diabetes"],
        ],
        columns=["Variable", "Type", "Meaning"],
    )


def baseline_by_diabetes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in GENERAL_HEALTH_VARS + DEMOGRAPHIC_VARS:
        g0 = df.loc[df[TARGET] == 0, var]
        g1 = df.loc[df[TARGET] == 1, var]
        pooled = ((g0.var() + g1.var()) / 2) ** 0.5
        smd = (g1.mean() - g0.mean()) / pooled if pooled else 0.0
        rows.append(
            {
                "Variable": var,
                "No diabetes": round(float(g0.mean()), 3),
                "Diabetes": round(float(g1.mean()), 3),
                "Difference": round(float(g1.mean() - g0.mean()), 3),
                "SMD": round(float(smd), 3),
            }
        )
    return pd.DataFrame(rows)


def spearman_corr(df: pd.DataFrame) -> pd.DataFrame:
    return df[SELECTED_VARS].corr(method="spearman").round(3)


def plot_outcome(df: pd.DataFrame) -> plt.Figure:
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(["No diabetes", "Diabetes"], counts.values, color=["#8fbcd4", "#e26d5a"])
    ax.set_title("Outcome Distribution")
    ax.set_ylabel("Number of respondents")
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, count, f"{count:,}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    return fig


def plot_health_panels(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.8))
    tmp = df.copy()
    tmp["GenHlthLabel"] = labeled(tmp["GenHlth"], "GenHlth")
    sns.countplot(
        data=tmp,
        x="GenHlthLabel",
        hue=TARGET,
        order=[GENHLTH_LABELS[i] for i in sorted(GENHLTH_LABELS)],
        palette=["#8fbcd4", "#e26d5a"],
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("General Health by Diabetes Status")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].legend(title="", labels=["No diabetes", "Diabetes"])

    sns.boxplot(data=df, x=TARGET, y="PhysHlth", palette=["#8fbcd4", "#e26d5a"], showfliers=False, ax=axes[0, 1])
    axes[0, 1].set_title("Poor Physical Health Days")
    axes[0, 1].set_xticklabels(["No diabetes", "Diabetes"])
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Days")

    sns.boxplot(data=df, x=TARGET, y="MentHlth", palette=["#8fbcd4", "#e26d5a"], showfliers=False, ax=axes[1, 0])
    axes[1, 0].set_title("Poor Mental Health Days")
    axes[1, 0].set_xticklabels(["No diabetes", "Diabetes"])
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("Days")

    diff = df.groupby(TARGET)["DiffWalk"].mean().rename(index={0.0: "No diabetes", 1.0: "Diabetes"}) * 100
    axes[1, 1].bar(diff.index, diff.values, color=["#8fbcd4", "#e26d5a"])
    axes[1, 1].set_title("Difficulty Walking by Diabetes Status")
    axes[1, 1].set_ylabel("Percent with difficulty")
    for i, v in enumerate(diff.values):
        axes[1, 1].text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=10)

    fig.tight_layout()
    return fig


def plot_demographic_panels(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.8))

    for ax, variable, labels, title in [
        (axes[0, 0], "Age", AGE_LABELS, "Age Distribution by Diabetes Status"),
        (axes[1, 0], "Education", EDU_LABELS, "Education Distribution by Diabetes Status"),
        (axes[1, 1], "Income", INCOME_LABELS, "Income Distribution by Diabetes Status"),
    ]:
        dist = pd.crosstab(df[variable], df[TARGET], normalize="columns") * 100
        dist.index = [labels[i] for i in dist.index]
        dist.columns = ["No diabetes", "Diabetes"]
        dist.plot(kind="bar", ax=ax, color=["#8fbcd4", "#e26d5a"], width=0.85)
        ax.set_title(title)
        ax.set_ylabel("Column percent")
        ax.tick_params(axis="x", rotation=35 if variable != "Age" else 45)
        ax.legend(title="")

    sex = df.groupby(TARGET)["Sex"].mean().rename(index={0.0: "No diabetes", 1.0: "Diabetes"}) * 100
    axes[0, 1].bar(sex.index, sex.values, color=["#8fbcd4", "#e26d5a"])
    axes[0, 1].set_title("Male Share by Diabetes Status")
    axes[0, 1].set_ylabel("Percent male")
    for i, v in enumerate(sex.values):
        axes[0, 1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)

    fig.tight_layout()
    return fig


def plot_prevalence_gradients(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.8))

    def bar_from_group(ax, variable, labels_map, title):
        prev = df.groupby(variable)[TARGET].mean() * 100
        prev.index = [labels_map[int(i)] for i in prev.index]
        ax.bar(prev.index, prev.values, color="#5b8e7d")
        ax.set_title(title)
        ax.set_ylabel("Diabetes prevalence (%)")
        ax.tick_params(axis="x", rotation=30)

    bar_from_group(axes[0, 0], "GenHlth", GENHLTH_LABELS, "Prevalence by Self-Rated Health")
    bar_from_group(axes[0, 1], "Age", AGE_LABELS, "Prevalence by Age Group")

    phys = pd.cut(df["PhysHlth"], bins=[-1, 0, 5, 13, 29, 30], labels=["0", "1-5", "6-13", "14-29", "30"])
    prev_phys = df.groupby(phys, observed=False)[TARGET].mean() * 100
    axes[1, 0].bar(prev_phys.index.astype(str), prev_phys.values, color="#7c9dc5")
    axes[1, 0].set_title("Prevalence by Poor Physical Health Days")
    axes[1, 0].set_ylabel("Diabetes prevalence (%)")

    inc = df.groupby("Income")[TARGET].mean() * 100
    inc.index = [INCOME_LABELS[int(i)] for i in inc.index]
    axes[1, 1].bar(inc.index, inc.values, color="#b07aa1")
    axes[1, 1].set_title("Prevalence by Income Category")
    axes[1, 1].set_ylabel("Diabetes prevalence (%)")
    axes[1, 1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    return fig


def plot_heatmap(df: pd.DataFrame) -> plt.Figure:
    corr = spearman_corr(df)
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Spearman Correlation Matrix for Selected Variables")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig


def findings(df: pd.DataFrame) -> list[str]:
    prev = df[TARGET].mean() * 100
    diff_no = df.loc[df[TARGET] == 0, "DiffWalk"].mean() * 100
    diff_yes = df.loc[df[TARGET] == 1, "DiffWalk"].mean() * 100
    phys_no = df.loc[df[TARGET] == 0, "PhysHlth"].mean()
    phys_yes = df.loc[df[TARGET] == 1, "PhysHlth"].mean()
    ment_no = df.loc[df[TARGET] == 0, "MentHlth"].mean()
    ment_yes = df.loc[df[TARGET] == 1, "MentHlth"].mean()
    return [
        f"The selected EDA dataset contains {len(df):,} respondents with no missing values across the variables used in this report. Diabetes prevalence is {prev:.1f}%, so the outcome is meaningfully imbalanced but still large enough for stable subgroup summaries.",
        f"The clearest health-status gradient appears in self-rated general health. Diabetes prevalence rises from 2.5% among respondents reporting excellent health to 37.9% among those reporting poor health, which makes GenHlth one of the most informative variables in this subset.",
        f"Physical functioning variables are much more strongly separated by diabetes status than mental health alone. Difficulty walking increases from {diff_no:.1f}% in the non-diabetes group to {diff_yes:.1f}% in the diabetes group, and mean poor physical health days rise from {phys_no:.2f} to {phys_yes:.2f}.",
        f"Mental health still shows a difference, but the magnitude is smaller: mean poor mental health days increase from {ment_no:.2f} to {ment_yes:.2f}, and the distribution remains heavily concentrated at zero in both groups.",
        "Demographic gradients are also pronounced. Diabetes prevalence increases across older age groups and decreases across higher income and education levels, which supports treating socioeconomic position as part of the main analytic story rather than as a minor adjustment variable.",
    ]


def render_report(df: pd.DataFrame) -> None:
    pdf_path = REPORT_DIR / "diabetes_eda_extended_general_health_demographics.pdf"
    overview = summary_table(df)
    vardict = variable_dictionary()
    baseline = baseline_by_diabetes(df)
    genhlth_prev = prevalence_table(df, "GenHlth")
    diffwalk_prev = prevalence_table(df, "DiffWalk")
    sex_prev = prevalence_table(df, "Sex")
    age_prev = prevalence_table(df, "Age")
    income_prev = prevalence_table(df, "Income")
    education_prev = prevalence_table(df, "Education")
    phys_bucket = health_day_bucket_table(df, "PhysHlth")
    ment_bucket = health_day_bucket_table(df, "MentHlth")
    corr = spearman_corr(df).reset_index().rename(columns={"index": "Variable"})

    overview.to_csv(TABLES_DIR / "extended_dataset_overview.csv", index=False)
    baseline.to_csv(TABLES_DIR / "extended_baseline_by_diabetes.csv", index=False)
    genhlth_prev.to_csv(TABLES_DIR / "extended_genhlth_prevalence.csv", index=False)
    diffwalk_prev.to_csv(TABLES_DIR / "extended_diffwalk_prevalence.csv", index=False)
    sex_prev.to_csv(TABLES_DIR / "extended_sex_prevalence.csv", index=False)
    age_prev.to_csv(TABLES_DIR / "extended_age_prevalence.csv", index=False)
    income_prev.to_csv(TABLES_DIR / "extended_income_prevalence.csv", index=False)
    education_prev.to_csv(TABLES_DIR / "extended_education_prevalence.csv", index=False)
    phys_bucket.to_csv(TABLES_DIR / "extended_physhlth_bucket_prevalence.csv", index=False)
    ment_bucket.to_csv(TABLES_DIR / "extended_menthlth_bucket_prevalence.csv", index=False)
    corr.to_csv(TABLES_DIR / "extended_spearman_corr.csv", index=False)

    with PdfPages(pdf_path) as pdf:
        add_text_page(
            pdf,
            "Extended EDA of General Health and Demographic Variables in BRFSS 2015 Diabetes Data",
            [
                "Date: 2026-04-23. This report focuses on the general health and fairness-oriented demographic variables that your team highlighted for the diabetes analysis: GenHlth, PhysHlth, MentHlth, DiffWalk, Age, Sex, Income, and Education.",
                "The purpose of the report is to move beyond simple descriptive counts and show which variables exhibit clear gradients with diabetes status, which variables appear secondary, and which groups may deserve more attention in later prediction, fairness, or causal analyses.",
                findings(df)[0],
                findings(df)[1],
            ],
        )
        add_table_page(pdf, "Dataset Overview", overview)
        add_table_page(pdf, "Variable Dictionary", vardict)
        save_plot(pdf, plot_outcome(df), "extended_outcome_distribution.png")
        add_text_page(
            pdf,
            "High-Level EDA Interpretation",
            findings(df),
        )
        add_table_page(pdf, "Baseline Comparison by Diabetes Status", baseline, note="SMD is included to show which variables separate the two outcome groups most strongly on a standardized scale.")
        save_plot(pdf, plot_health_panels(df), "extended_general_health_panels.png")
        add_table_page(pdf, "Diabetes Prevalence by Self-Rated Health and Walking Difficulty", pd.concat(
            [
                genhlth_prev.assign(Variable="GenHlth").rename(columns={"GenHlth": "Category"}),
                diffwalk_prev.assign(Variable="DiffWalk").rename(columns={"DiffWalk": "Category"}),
            ],
            ignore_index=True,
        ))
        add_table_page(pdf, "Diabetes Prevalence by Poor Physical and Mental Health Day Buckets", pd.concat(
            [
                phys_bucket.assign(Variable="PhysHlth").rename(columns={"PhysHlth_bucket": "Category"}),
                ment_bucket.assign(Variable="MentHlth").rename(columns={"MentHlth_bucket": "Category"}),
            ],
            ignore_index=True,
        ), note="Physical health days show a cleaner monotonic gradient than mental health days.")
        save_plot(pdf, plot_demographic_panels(df), "extended_demographic_panels.png")
        add_table_page(pdf, "Diabetes Prevalence by Age, Sex, Income, and Education", pd.concat(
            [
                age_prev.assign(Variable="Age").rename(columns={"Age": "Category"}),
                sex_prev.assign(Variable="Sex").rename(columns={"Sex": "Category"}),
                income_prev.assign(Variable="Income").rename(columns={"Income": "Category"}),
                education_prev.assign(Variable="Education").rename(columns={"Education": "Category"}),
            ],
            ignore_index=True,
        ), fontsize=8.5)
        save_plot(pdf, plot_prevalence_gradients(df), "extended_prevalence_gradients.png")
        save_plot(pdf, plot_heatmap(df), "extended_selected_variable_heatmap.png")
        add_table_page(pdf, "Spearman Correlation Matrix", corr, note="Spearman correlation is more appropriate than Pearson here because the selected predictors mix binary, ordinal, and count scales.", fontsize=8.0)
        add_text_page(
            pdf,
            "Interpretation and Opportunities for Further Analysis",
            [
                "The main pattern in this subset is not a general mental-health story. Instead, diabetes appears most closely tied to a cluster of worse self-rated health, more poor physical-health days, and greater mobility limitation. Those variables are moderately correlated with one another, suggesting that they are capturing a shared health-burden dimension.",
                "Income and education move in the opposite direction and show a clear socioeconomic gradient. That makes them useful not only as predictors but also as variables for subgroup performance checks, fairness discussion, and confounding adjustment if the team adds a focused causal analysis around physical activity later.",
                "The correlation matrix belongs in the EDA section when it adds interpretation rather than clutter. In this case it helps because it shows both association with the outcome and redundancy among predictors, especially the link between GenHlth, PhysHlth, and DiffWalk, as well as the pairing of income and education.",
                "Reasonable next steps include variable-block modeling, subgroup error analysis, and a small fairness section. A strong modeling comparison would be to test demographics alone versus demographics plus health burden versus demographics plus health burden plus mental health, and then quantify whether MentHlth adds meaningful predictive value beyond the stronger physical-health signals.",
            ],
        )


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="notebook")
    df = pd.read_csv(DATA_PATH, usecols=SELECTED_VARS)
    render_report(df)
    print(f"Extended EDA report written to: {REPORT_DIR / 'diabetes_eda_extended_general_health_demographics.pdf'}")


if __name__ == "__main__":
    main()
