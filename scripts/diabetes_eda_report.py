import os
from pathlib import Path
import textwrap
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-config")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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


def wrap(text: str, width: int = 95) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def add_text_page(pdf: PdfPages, title: str, paragraphs: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.95, title, fontsize=16, fontweight="bold", va="top")

    y = 0.90
    for para in paragraphs:
        fig.text(0.08, y, wrap(para), fontsize=10.5, va="top")
        y -= 0.09 + 0.018 * wrap(para).count("\n")

    pdf.savefig(fig)
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, note: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    rendered = df.copy()
    table = ax.table(
        cellText=rendered.values,
        colLabels=rendered.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.45)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d9eaf7")
        if row > 0 and row % 2 == 0:
            cell.set_facecolor("#f7fbff")

    if note:
        fig.text(0.08, 0.03, wrap(note, 100), fontsize=9, va="bottom")

    pdf.savefig(fig)
    plt.close(fig)


def save_and_attach_plot(pdf: PdfPages, fig: plt.Figure, filename: str) -> None:
    fig.savefig(PLOTS_DIR / filename, dpi=220, bbox_inches="tight")
    pdf.savefig(fig)
    plt.close(fig)


def labeled_series(series: pd.Series, variable: str) -> pd.Series:
    mapping = LABEL_MAPS.get(variable)
    if mapping is None:
        return series
    return series.map(mapping).fillna(series.astype(str))


def format_pct(x: float) -> str:
    return f"{x:.1f}%"


def overall_overview(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[TARGET].value_counts().sort_index()
    return pd.DataFrame(
        {
            "Metric": [
                "Rows",
                "Columns used in this EDA",
                "Missing cells",
                "No diabetes",
                "Diabetes",
            ],
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
            ["GenHlth", "Ordinal (1-5)", "Self-rated overall health; 1=excellent, 5=poor"],
            ["PhysHlth", "Count (0-30)", "Days of poor physical health in past 30 days"],
            ["MentHlth", "Count (0-30)", "Days of poor mental health in past 30 days"],
            ["DiffWalk", "Binary", "Serious difficulty walking or climbing stairs"],
            ["Age", "Ordinal (1-13)", "Age group from 18-24 through 80+"],
            ["Sex", "Binary", "0=female, 1=male"],
            ["Income", "Ordinal (1-8)", "Income category from <10k to 75k+"],
            ["Education", "Ordinal (1-6)", "Education level from no school to college graduate"],
            [TARGET, "Binary", "0=no diabetes, 1=diabetes"],
        ],
        columns=["Variable", "Type", "Description"],
    )


def univariate_numeric(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    rows = []
    for var in variables:
        s = df[var]
        rows.append(
            {
                "Variable": var,
                "Mean": round(float(s.mean()), 2),
                "SD": round(float(s.std()), 2),
                "Median": round(float(s.median()), 2),
                "IQR": f"{s.quantile(0.25):.1f}-{s.quantile(0.75):.1f}",
                "Pct zero": format_pct(float((s == 0).mean() * 100)),
                "Pct max (30)": format_pct(float((s == 30).mean() * 100)),
            }
        )
    return pd.DataFrame(rows)


def univariate_categorical(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    counts = df[variable].value_counts().sort_index()
    labels = [LABEL_MAPS.get(variable, {}).get(int(k), str(int(k))) for k in counts.index]
    out = pd.DataFrame(
        {
            variable: labels,
            "Count": counts.values,
            "Percent": (counts.values / len(df) * 100).round(1),
        }
    )
    out["Percent"] = out["Percent"].map(lambda x: f"{x:.1f}%")
    return out


def bivariate_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(TARGET)
    rows = []
    for var in GENERAL_HEALTH_VARS + DEMOGRAPHIC_VARS:
        if var in ["PhysHlth", "MentHlth"]:
            no_val = grouped[var].mean().loc[0.0]
            yes_val = grouped[var].mean().loc[1.0]
            metric = "Mean days"
        else:
            no_val = grouped[var].mean().loc[0.0]
            yes_val = grouped[var].mean().loc[1.0]
            metric = "Mean/proportion"

        rows.append(
            {
                "Variable": var,
                "Metric": metric,
                "No diabetes": round(float(no_val), 3),
                "Diabetes": round(float(yes_val), 3),
                "Difference": round(float(yes_val - no_val), 3),
            }
        )
    return pd.DataFrame(rows)


def prevalence_by_category(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    summary = df.groupby(variable)[TARGET].agg(["count", "mean"]).reset_index()
    summary[variable] = summary[variable].astype(int).map(lambda x: LABEL_MAPS.get(variable, {}).get(x, str(x)))
    summary["count"] = summary["count"].map(lambda x: f"{x:,}")
    summary["Diabetes prevalence"] = (summary["mean"] * 100).round(1).map(lambda x: f"{x:.1f}%")
    return summary[[variable, "count", "Diabetes prevalence"]]


def combined_prevalence_table(df: pd.DataFrame) -> pd.DataFrame:
    selected = {
        "GenHlth": ["Excellent", "Good", "Poor"],
        "DiffWalk": ["No", "Yes"],
        "Sex": ["Female", "Male"],
        "Age": ["18-24", "50-54", "65-69", "80+"],
        "Income": ["<10k", "35-50k", "75k+"],
    }
    blocks = []
    for variable, levels in selected.items():
        temp = prevalence_by_category(df, variable)
        temp = temp[temp[variable].isin(levels)].copy()
        temp.insert(0, "Variable", variable)
        temp = temp.rename(columns={variable: "Category", "count": "N"})
        blocks.append(temp)
    return pd.concat(blocks, ignore_index=True)


def group_distribution(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    ct = pd.crosstab(df[variable], df[TARGET], normalize="columns") * 100
    ct = ct.sort_index()
    ct.index = [LABEL_MAPS.get(variable, {}).get(int(k), str(int(k))) for k in ct.index]
    ct.columns = ["No diabetes", "Diabetes"]
    return ct.round(1).reset_index().rename(columns={"index": variable})


def make_outcome_plot(df: pd.DataFrame) -> plt.Figure:
    counts = df[TARGET].value_counts().sort_index()
    labels = ["No diabetes", "Diabetes"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, counts.values, color=["#8fbcd4", "#e26d5a"])
    ax.set_title("Outcome Distribution", fontsize=13)
    ax.set_ylabel("Number of respondents")
    ax.set_xlabel("")
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,}", ha="center", va="bottom", fontsize=10)
    return fig


def make_general_health_plot(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
    order = list(GENHLTH_LABELS.keys())
    plot_df = df.copy()
    plot_df["GenHlthLabel"] = labeled_series(plot_df["GenHlth"], "GenHlth")
    sns.countplot(
        data=plot_df,
        x="GenHlthLabel",
        hue=TARGET,
        order=[GENHLTH_LABELS[k] for k in order],
        palette=["#8fbcd4", "#e26d5a"],
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("General Health by Diabetes Status")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].legend(title="", labels=["No diabetes", "Diabetes"])

    sns.boxplot(data=df, x=TARGET, y="PhysHlth", palette=["#8fbcd4", "#e26d5a"], ax=axes[0, 1], showfliers=False)
    axes[0, 1].set_title("Poor Physical Health Days")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_xticklabels(["No diabetes", "Diabetes"])
    axes[0, 1].set_ylabel("Days in past 30")

    sns.boxplot(data=df, x=TARGET, y="MentHlth", palette=["#8fbcd4", "#e26d5a"], ax=axes[1, 0], showfliers=False)
    axes[1, 0].set_title("Poor Mental Health Days")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_xticklabels(["No diabetes", "Diabetes"])
    axes[1, 0].set_ylabel("Days in past 30")

    diff = df.groupby(TARGET)["DiffWalk"].mean().rename(index={0.0: "No diabetes", 1.0: "Diabetes"}) * 100
    axes[1, 1].bar(diff.index, diff.values, color=["#8fbcd4", "#e26d5a"])
    axes[1, 1].set_title("Difficulty Walking by Diabetes Status")
    axes[1, 1].set_ylabel("Percent with difficulty")
    axes[1, 1].set_xlabel("")
    for i, v in enumerate(diff.values):
        axes[1, 1].text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=10)

    fig.tight_layout()
    return fig


def make_demographic_plot(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.8))

    age_dist = pd.crosstab(df["Age"], df[TARGET], normalize="columns") * 100
    age_dist.index = [AGE_LABELS[k] for k in age_dist.index]
    age_dist.columns = ["No diabetes", "Diabetes"]
    age_dist.plot(kind="bar", ax=axes[0, 0], color=["#8fbcd4", "#e26d5a"], width=0.85)
    axes[0, 0].set_title("Age Distribution by Diabetes Status")
    axes[0, 0].set_ylabel("Column percent")
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].legend(title="")

    sex = df.groupby(TARGET)["Sex"].mean().rename(index={0.0: "No diabetes", 1.0: "Diabetes"}) * 100
    axes[0, 1].bar(sex.index, sex.values, color=["#8fbcd4", "#e26d5a"])
    axes[0, 1].set_title("Male Share by Diabetes Status")
    axes[0, 1].set_ylabel("Percent male")
    for i, v in enumerate(sex.values):
        axes[0, 1].text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=10)

    edu = pd.crosstab(df["Education"], df[TARGET], normalize="columns") * 100
    edu.index = [EDU_LABELS[k] for k in edu.index]
    edu.columns = ["No diabetes", "Diabetes"]
    edu.plot(kind="bar", ax=axes[1, 0], color=["#8fbcd4", "#e26d5a"], width=0.85)
    axes[1, 0].set_title("Education Distribution by Diabetes Status")
    axes[1, 0].set_ylabel("Column percent")
    axes[1, 0].tick_params(axis="x", rotation=35)
    axes[1, 0].legend(title="")

    inc = pd.crosstab(df["Income"], df[TARGET], normalize="columns") * 100
    inc.index = [INCOME_LABELS[k] for k in inc.index]
    inc.columns = ["No diabetes", "Diabetes"]
    inc.plot(kind="bar", ax=axes[1, 1], color=["#8fbcd4", "#e26d5a"], width=0.85)
    axes[1, 1].set_title("Income Distribution by Diabetes Status")
    axes[1, 1].set_ylabel("Column percent")
    axes[1, 1].tick_params(axis="x", rotation=35)
    axes[1, 1].legend(title="")

    fig.tight_layout()
    return fig


def make_heatmap(df: pd.DataFrame) -> plt.Figure:
    corr_vars = [TARGET] + GENERAL_HEALTH_VARS + DEMOGRAPHIC_VARS
    corr = df[corr_vars].corr()
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap for Selected Variables")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig


def key_findings(df: pd.DataFrame) -> list[str]:
    prev = df[TARGET].mean() * 100
    poor_general = ((df["GenHlth"] >= 4).mean()) * 100
    diff_no = df.loc[df[TARGET] == 0, "DiffWalk"].mean() * 100
    diff_yes = df.loc[df[TARGET] == 1, "DiffWalk"].mean() * 100
    phys_no = df.loc[df[TARGET] == 0, "PhysHlth"].mean()
    phys_yes = df.loc[df[TARGET] == 1, "PhysHlth"].mean()
    inc_no = df.loc[df[TARGET] == 0, "Income"].mean()
    inc_yes = df.loc[df[TARGET] == 1, "Income"].mean()
    age_no = df.loc[df[TARGET] == 0, "Age"].mean()
    age_yes = df.loc[df[TARGET] == 1, "Age"].mean()

    return [
        f"The working EDA dataset contains {len(df):,} respondents and no missing values across the selected variables. Diabetes prevalence in the binary file is {prev:.1f}%.",
        f"Overall health is already skewed toward the healthier categories, but {poor_general:.1f}% of respondents still rate their health as fair or poor.",
        f"Respondents with diabetes report substantially worse physical functioning: difficulty walking rises from {diff_no:.1f}% in the non-diabetes group to {diff_yes:.1f}% in the diabetes group, and mean poor physical health days increase from {phys_no:.2f} to {phys_yes:.2f}.",
        f"The diabetes group is older on average in the ordinal age coding ({age_yes:.2f} vs {age_no:.2f}) and also shifted downward on income ({inc_yes:.2f} vs {inc_no:.2f}), which supports including demographic and socioeconomic variables in later fairness and confounding analyses.",
        "Mental health is more right-skewed than normally distributed, with a majority of respondents reporting zero poor-mental-health days. Physical health shows the same pattern but with a heavier upper tail among respondents with diabetes.",
    ]


def render_report(df: pd.DataFrame) -> None:
    pdf_path = REPORT_DIR / "diabetes_eda_general_health_demographics.pdf"
    overview = overall_overview(df)
    vardict = variable_dictionary()
    numeric_summary = univariate_numeric(df, ["PhysHlth", "MentHlth"])
    bivar = bivariate_summary(df)
    prevalence_table = combined_prevalence_table(df)
    genhlth_dist = univariate_categorical(df, "GenHlth")
    age_dist = univariate_categorical(df, "Age")
    sex_dist = univariate_categorical(df, "Sex")
    income_dist = univariate_categorical(df, "Income")
    edu_dist = univariate_categorical(df, "Education")
    diffwalk_dist = univariate_categorical(df, "DiffWalk")
    genhlth_by_dm = group_distribution(df, "GenHlth")
    age_by_dm = group_distribution(df, "Age")
    income_by_dm = group_distribution(df, "Income")
    edu_by_dm = group_distribution(df, "Education")

    overview.to_csv(TABLES_DIR / "dataset_overview.csv", index=False)
    vardict.to_csv(TABLES_DIR / "variable_dictionary.csv", index=False)
    numeric_summary.to_csv(TABLES_DIR / "univariate_numeric_summary.csv", index=False)
    bivar.to_csv(TABLES_DIR / "bivariate_summary_by_diabetes.csv", index=False)
    prevalence_table.to_csv(TABLES_DIR / "diabetes_prevalence_selected_categories.csv", index=False)
    genhlth_dist.to_csv(TABLES_DIR / "genhlth_distribution.csv", index=False)
    age_dist.to_csv(TABLES_DIR / "age_distribution.csv", index=False)
    sex_dist.to_csv(TABLES_DIR / "sex_distribution.csv", index=False)
    income_dist.to_csv(TABLES_DIR / "income_distribution.csv", index=False)
    edu_dist.to_csv(TABLES_DIR / "education_distribution.csv", index=False)
    diffwalk_dist.to_csv(TABLES_DIR / "diffwalk_distribution.csv", index=False)
    genhlth_by_dm.to_csv(TABLES_DIR / "genhlth_by_diabetes.csv", index=False)
    age_by_dm.to_csv(TABLES_DIR / "age_by_diabetes.csv", index=False)
    income_by_dm.to_csv(TABLES_DIR / "income_by_diabetes.csv", index=False)
    edu_by_dm.to_csv(TABLES_DIR / "education_by_diabetes.csv", index=False)

    with PdfPages(pdf_path) as pdf:
        findings = key_findings(df)
        add_text_page(
            pdf,
            "EDA of General Health and Demographic Variables in BRFSS 2015 Diabetes Data",
            [
                "Date: 2026-04-23. This exploratory report focuses on the requested variable block: GenHlth, PhysHlth, MentHlth, DiffWalk, Age, Sex, Income, and Education. The binary diabetes file is used as the working dataset so that the EDA stays aligned with later prediction and causal analyses.",
                "The goal of this section is not to exhaust every possible plot, but to isolate the health-status and demographic patterns that are most useful for interpretation, subgroup analysis, and later model building.",
                findings[0],
                findings[2],
            ],
        )
        add_table_page(pdf, "Dataset Overview", overview, note="Supporting distribution tables are saved under outputs/tables/ and can be selectively pulled into the final 10-page team report.")
        add_table_page(pdf, "Selected Variable Dictionary", vardict, note="Age, education, and income are ordinal category codes rather than raw continuous measurements.")
        add_table_page(pdf, "Univariate Summary for Count Variables", numeric_summary, note="For categorical variables, full one-way distributions are exported as CSV files under outputs/tables/.")
        add_table_page(pdf, "Selected Category-Level Diabetes Prevalence", prevalence_table, note="This compact table keeps only the most interpretable category contrasts. Full category distributions are available in the CSV outputs.")
        save_and_attach_plot(pdf, make_outcome_plot(df), "outcome_distribution.png")
        save_and_attach_plot(pdf, make_general_health_plot(df), "general_health_panels.png")
        save_and_attach_plot(pdf, make_demographic_plot(df), "demographic_panels.png")
        add_table_page(pdf, "Bivariate Summary by Diabetes Status", bivar, note="Binary and ordinal coded variables are shown as group means/proportions to keep the table compact. Complementary distribution tables are saved separately under outputs/tables/.")
        add_text_page(
            pdf,
            "Interpretation for Modeling",
            [
                "The strongest signal in this subset comes from functional and self-rated health rather than from mental health alone. General health, poor physical health days, and walking difficulty all separate the diabetes and non-diabetes groups much more sharply than sex does.",
                "The category-level prevalence table is useful because it turns coded variables into interpretable comparisons. Diabetes prevalence increases as self-rated health worsens, rises steeply across older age groups, and declines across higher income categories, suggesting a broad social and health gradient rather than a single isolated risk factor.",
                "Mental health still deserves attention, but in this subset it behaves more like a secondary correlate than a primary separator. Its distribution is heavily concentrated at zero, and the between-group difference is modest relative to the gap seen in physical-health burden or mobility limitation.",
                "For the final report, the clearest EDA narrative is that these variables capture three linked dimensions: overall health burden, physical functioning, and demographic disadvantage. That framing will connect naturally to both the later prediction results and any fairness or confounding discussion built around age, income, education, and sex.",
            ],
        )


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="notebook")
    df = pd.read_csv(DATA_PATH, usecols=SELECTED_VARS).copy()
    render_report(df)
    print(f"EDA report written to: {REPORT_DIR / 'diabetes_eda_general_health_demographics.pdf'}")
    print(f"Plots written to: {PLOTS_DIR}")
    print(f"Tables written to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
