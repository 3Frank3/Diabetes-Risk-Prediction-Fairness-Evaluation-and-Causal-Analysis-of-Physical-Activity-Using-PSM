import os
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-config")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import nbformat as nbf
import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_DIR = Path("/Users/zansong/Desktop/203C/Project")
DATA_PATH = PROJECT_DIR / "archive-6" / "diabetes_binary_health_indicators_BRFSS2015.csv"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
NOTEBOOK_PATH = SCRIPTS_DIR / "01_EDA.ipynb"
OUTPUT_DIR = PROJECT_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "report"
TABLE_IMAGES_DIR = OUTPUT_DIR / "table_images"

TARGET = "Diabetes_binary"

CARDIOMETABOLIC_VARS = [
    "BMI",
    "HighBP",
    "HighChol",
    "CholCheck",
    "Stroke",
    "HeartDiseaseorAttack",
]
LIFESTYLE_VARS = ["Smoker", "PhysActivity"]
GENERAL_HEALTH_VARS = ["GenHlth", "PhysHlth", "MentHlth", "DiffWalk"]
DEMOGRAPHIC_VARS = ["Age", "Sex", "Income", "Education"]

SECTION_VARS = CARDIOMETABOLIC_VARS + LIFESTYLE_VARS + GENERAL_HEALTH_VARS + DEMOGRAPHIC_VARS + [TARGET]
ALL_VARS = [
    "Diabetes_binary",
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]

BLUE = "#8FBCD4"
RED = "#E26D5A"
HEADER_BLUE = "#D9EAF7"
GRID_GREY = "#D0D0D0"
TEXT_GREY = "#444444"
GREEN = "#5B8E7D"
PURPLE = "#B07AA1"

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
SEX_LABELS = {0: "F", 1: "M"}
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
EDUCATION_LABELS = {
    1: "No school/K",
    2: "Grades 1-8",
    3: "Grades 9-11",
    4: "HS graduate",
    5: "Some college",
    6: "College grad",
}
BINARY_LABELS = {0: "No", 1: "Yes"}
TARGET_LABELS = {0: "No diabetes", 1: "Diabetes"}

LABEL_MAPS = {
    "GenHlth": GENHLTH_LABELS,
    "Age": AGE_LABELS,
    "Sex": SEX_LABELS,
    "Income": INCOME_LABELS,
    "Education": EDUCATION_LABELS,
    "HighBP": BINARY_LABELS,
    "HighChol": BINARY_LABELS,
    "CholCheck": BINARY_LABELS,
    "Stroke": BINARY_LABELS,
    "HeartDiseaseorAttack": BINARY_LABELS,
    "Smoker": BINARY_LABELS,
    "PhysActivity": BINARY_LABELS,
    "DiffWalk": BINARY_LABELS,
    TARGET: TARGET_LABELS,
}


def ensure_dirs() -> None:
    for path in [PLOTS_DIR, TABLES_DIR, REPORT_DIR, TABLE_IMAGES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["axes.edgecolor"] = GRID_GREY
    plt.rcParams["grid.color"] = GRID_GREY
    plt.rcParams["axes.labelcolor"] = TEXT_GREY
    plt.rcParams["text.color"] = TEXT_GREY
    plt.rcParams["axes.titlecolor"] = TEXT_GREY
    plt.rcParams["xtick.color"] = TEXT_GREY
    plt.rcParams["ytick.color"] = TEXT_GREY


def wrap(text: str, width: int = 96) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def labeled(series: pd.Series, variable: str) -> pd.Series:
    mapping = LABEL_MAPS.get(variable)
    if mapping is None:
        return series.astype(str)
    return series.astype(int).map(mapping)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, usecols=ALL_VARS).copy()
    df["BMI_group"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
        include_lowest=True,
    )
    df["PhysHlthPct30"] = df["PhysHlth"] / 30.0 * 100.0
    df["MentHlthPct30"] = df["MentHlth"] / 30.0 * 100.0
    return df


def save_csv(df: pd.DataFrame, slug: str) -> Path:
    path = TABLES_DIR / f"{slug}.csv"
    df.to_csv(path, index=False)
    return path


def make_table_figure(df: pd.DataFrame, title: str, note: Optional[str] = None, figsize=(10.5, 0)) -> plt.Figure:
    rows = len(df)
    height = max(2.6, min(11.5, 1.4 + 0.42 * (rows + 1)))
    width = figsize[0]
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.0 if rows < 18 else 8.0)
    table.scale(1, 1.45)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(HEADER_BLUE)
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F8FBFD")
    if note:
        fig.text(0.06, 0.03, wrap(note, 102), fontsize=8.8, va="bottom")
    return fig


def export_table(df: pd.DataFrame, slug: str, title: str, pdf: Optional[PdfPages] = None, note: Optional[str] = None) -> Dict[str, object]:
    csv_path = save_csv(df, slug)
    fig = make_table_figure(df, title, note=note)
    png_path = TABLE_IMAGES_DIR / f"{slug}.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)
    return {"df": df, "csv": csv_path, "png": png_path, "title": title}


def export_plot(fig: plt.Figure, slug: str, pdf: Optional[PdfPages] = None) -> Path:
    path = PLOTS_DIR / f"{slug}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)
    return path


def add_text_page(pdf: PdfPages, title: str, paragraphs: List[str]) -> None:
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


def prevalence_table(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    temp = df.groupby(variable, observed=False)[TARGET].agg(["count", "mean"]).reset_index()
    if variable == "BMI_group":
        temp[variable] = temp[variable].astype(str)
    else:
        temp[variable] = labeled(temp[variable], variable)
    temp["Count"] = temp["count"].map(lambda x: f"{x:,}")
    temp["Diabetes prevalence"] = (temp["mean"] * 100).round(1).map(lambda x: f"{x:.1f}%")
    return temp[[variable, "Count", "Diabetes prevalence"]]


def baseline_table(df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    rows = []
    for var in variables:
        g0 = df.loc[df[TARGET] == 0, var]
        g1 = df.loc[df[TARGET] == 1, var]
        pooled = ((g0.var() + g1.var()) / 2) ** 0.5
        diff = g1.mean() - g0.mean()
        smd = diff / pooled if pooled else 0.0
        rows.append(
            {
                "Variable": var,
                "No diabetes": round(float(g0.mean()), 3),
                "Diabetes": round(float(g1.mean()), 3),
                "Difference": round(float(diff), 3),
                "SMD": round(float(smd), 3),
            }
        )
    return pd.DataFrame(rows)


def bucket_prevalence_table(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    b = pd.cut(df[variable], bins=[-1, 0, 5, 13, 29, 30], labels=["0", "1-5", "6-13", "14-29", "30"])
    temp = df.groupby(b, observed=False)[TARGET].agg(["count", "mean"]).reset_index()
    temp.columns = ["Category", "count", "mean"]
    temp["Count"] = temp["count"].map(lambda x: f"{x:,}")
    temp["Diabetes prevalence"] = (temp["mean"] * 100).round(1).map(lambda x: f"{x:.1f}%")
    temp.insert(0, "Variable", variable)
    return temp[["Variable", "Category", "Count", "Diabetes prevalence"]]


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[TARGET].value_counts().sort_index()
    return pd.DataFrame(
        {
            "Metric": ["Rows", "Columns in all-variable dataset", "Missing cells", "No diabetes", "Diabetes"],
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
    rows = [
        ["BMI", "Continuous", "Body Mass Index"],
        ["BMI_group", "Categorical", "BMI grouped as Underweight / Normal / Overweight / Obese"],
        ["HighBP", "Binary", "High blood pressure"],
        ["HighChol", "Binary", "High cholesterol"],
        ["CholCheck", "Binary", "Cholesterol check in past 5 years"],
        ["Stroke", "Binary", "History of stroke"],
        ["HeartDiseaseorAttack", "Binary", "History of coronary heart disease or heart attack"],
        ["Smoker", "Binary", "Smoked at least 100 cigarettes in lifetime"],
        ["PhysActivity", "Binary", "Physical activity in past 30 days, excluding job"],
        ["GenHlth", "Ordinal", "Self-rated health from excellent to poor"],
        ["PhysHlth", "Count", "Poor physical health days in past 30 days"],
        ["MentHlth", "Count", "Poor mental health days in past 30 days"],
        ["PhysHlthPct30", "Derived percent", "Percent of 30 days reported as poor physical health"],
        ["MentHlthPct30", "Derived percent", "Percent of 30 days reported as poor mental health"],
        ["DiffWalk", "Binary", "Serious difficulty walking or climbing stairs"],
        ["Age", "Ordinal", "Age group 18-24 through 80+"],
        ["Sex", "Binary", "F / M indicator"],
        ["Income", "Ordinal", "Income category from <10k to 75k+"],
        ["Education", "Ordinal", "Education level from no school to college graduate"],
        [TARGET, "Binary", "No diabetes / diabetes"],
    ]
    return pd.DataFrame(rows, columns=["Variable", "Type", "Meaning"])


def corr_selected(df: pd.DataFrame) -> pd.DataFrame:
    vars_ = CARDIOMETABOLIC_VARS + LIFESTYLE_VARS + GENERAL_HEALTH_VARS + DEMOGRAPHIC_VARS + [TARGET]
    return df[vars_].corr(method="spearman").round(3)


def corr_all(df: pd.DataFrame) -> pd.DataFrame:
    return df[ALL_VARS].corr(method="spearman").round(3)


def plot_outcome_distribution(df: pd.DataFrame) -> plt.Figure:
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(["No diabetes", "Diabetes"], counts.values, color=[BLUE, RED])
    ax.set_title("Outcome Distribution")
    ax.set_ylabel("Number of respondents")
    ax.set_xlabel("")
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, count, f"{count:,}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    return fig


def plot_bmi_prevalence(df: pd.DataFrame) -> plt.Figure:
    prev = df.groupby("BMI_group", observed=False)[TARGET].mean() * 100
    order = ["Underweight", "Normal", "Overweight", "Obese"]
    prev = prev.reindex(order)
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    bars = ax.bar(prev.index.astype(str), prev.values, color=RED)
    ax.set_title("Diabetes Prevalence by BMI Category")
    ax.set_xlabel("BMI Category")
    ax.set_ylabel("Diabetes prevalence (%)")
    ax.tick_params(axis="x", rotation=0)
    for bar, val in zip(bars, prev.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.4, f"{val:.1f}%", ha="center", fontsize=10)
    fig.tight_layout()
    return fig


def plot_binary_prevalence_heatmap(df: pd.DataFrame, variables: List[str], title: str, cmap: str = "Reds") -> plt.Figure:
    heatmap_data = pd.DataFrame(
        {var: pd.crosstab(df[var], df[TARGET], normalize="index")[1] for var in variables}
    ).T * 100
    heatmap_data.columns = ["No", "Yes"]
    heatmap_data.index = variables
    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.52 * len(variables) + 1.5)))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap=cmap, ax=ax, cbar=True)
    ax.set_title(title)
    ax.set_xlabel("Group value")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def prevalence_series(df: pd.DataFrame, variable: str) -> pd.Series:
    if variable in {"PhysHlth", "MentHlth"}:
        grouped = pd.cut(df[variable], bins=[-1, 0, 5, 13, 29, 30], labels=["0", "1-5", "6-13", "14-29", "30"])
        prev = df.groupby(grouped, observed=False)[TARGET].mean() * 100
        prev.index = prev.index.astype(str)
        return prev
    if variable == "BMI_group":
        prev = df.groupby(variable, observed=False)[TARGET].mean() * 100
        prev.index = prev.index.astype(str)
        return prev
    prev = df.groupby(variable, observed=False)[TARGET].mean() * 100
    if variable in LABEL_MAPS:
        prev.index = [LABEL_MAPS[variable][int(i)] for i in prev.index]
    else:
        prev.index = prev.index.astype(str)
    return prev


def plot_category_heatmap_panels(df: pd.DataFrame, variables: List[str], titles: Dict[str, str], panel_title: str) -> plt.Figure:
    nrows = 2 if len(variables) > 1 else 1
    ncols = 2 if len(variables) > 1 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 6.8 if len(variables) > 1 else 3.0))
    axes = [axes] if len(variables) == 1 else list(axes.flatten())
    for ax, variable in zip(axes, variables):
        prev = prevalence_series(df, variable)
        hm = pd.DataFrame([prev.values], index=[""], columns=prev.index)
        sns.heatmap(
            hm,
            annot=True,
            fmt=".1f",
            cmap="Reds",
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
            annot_kws={"size": 11},
        )
        ax.set_title(titles[variable], fontsize=12, pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20, labelsize=9)
        ax.tick_params(axis="y", length=0, labelleft=False)
    for ax in axes[len(variables):]:
        ax.axis("off")
    fig.suptitle(panel_title, fontsize=16, y=0.97)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.07, right=0.98, hspace=0.75, wspace=0.28)
    return fig


def plot_binary_prevalence_bar(df: pd.DataFrame, variables: List[str], title: str, palette: List[str]) -> plt.Figure:
    rows = []
    for var in variables:
        ct = pd.crosstab(df[var], df[TARGET], normalize="index")
        for group in [0, 1]:
            rows.append(
                {
                    "Variable": var,
                    "Group": LABEL_MAPS.get(var, BINARY_LABELS).get(group, str(group)),
                    "Prevalence": ct.loc[group, 1] * 100 if group in ct.index else 0.0,
                }
            )
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.0, max(4.2, 0.52 * len(variables) + 2)))
    sns.barplot(data=plot_df, x="Variable", y="Prevalence", hue="Group", palette=palette, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Diabetes prevalence (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="")
    fig.tight_layout()
    return fig


def plot_genhlth_distribution(df: pd.DataFrame) -> plt.Figure:
    ct = pd.crosstab(df["GenHlth"], df[TARGET], normalize="columns") * 100
    ct.index = [GENHLTH_LABELS[int(i)] for i in ct.index]
    ct.columns = ["No diabetes", "Diabetes"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ct.plot(kind="bar", color=[BLUE, RED], width=0.85, ax=ax)
    ax.set_title("General Health Distribution by Diabetes Status")
    ax.set_xlabel("")
    ax.set_ylabel("Column percent")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="")
    fig.tight_layout()
    return fig


def plot_health_burden_panels(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.8))
    sns.boxplot(data=df, x=TARGET, y="PhysHlth", palette=[BLUE, RED], showfliers=False, ax=axes[0, 0])
    axes[0, 0].set_title("Poor Physical Health Days")
    axes[0, 0].set_xticklabels(["No diabetes", "Diabetes"])
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Days")

    sns.boxplot(data=df, x=TARGET, y="MentHlth", palette=[BLUE, RED], showfliers=False, ax=axes[0, 1])
    axes[0, 1].set_title("Poor Mental Health Days")
    axes[0, 1].set_xticklabels(["No diabetes", "Diabetes"])
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Days")

    sns.boxplot(data=df, x=TARGET, y="PhysHlthPct30", palette=[BLUE, RED], showfliers=False, ax=axes[1, 0])
    axes[1, 0].set_title("Poor Physical Health as % of 30 Days")
    axes[1, 0].set_xticklabels(["No diabetes", "Diabetes"])
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("Percent of 30 days")

    sns.boxplot(data=df, x=TARGET, y="MentHlthPct30", palette=[BLUE, RED], showfliers=False, ax=axes[1, 1])
    axes[1, 1].set_title("Poor Mental Health as % of 30 Days")
    axes[1, 1].set_xticklabels(["No diabetes", "Diabetes"])
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Percent of 30 days")

    fig.tight_layout()
    return fig


def plot_diffwalk_distribution(df: pd.DataFrame) -> plt.Figure:
    ct = pd.crosstab(df["DiffWalk"], df[TARGET], normalize="columns") * 100
    ct.index = [BINARY_LABELS[int(i)] for i in ct.index]
    ct.columns = ["No diabetes", "Diabetes"]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ct.plot(kind="bar", color=[BLUE, RED], width=0.8, ax=ax)
    ax.set_title("Walking Difficulty Distribution by Diabetes Status")
    ax.set_xlabel("")
    ax.set_ylabel("Column percent")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="")
    fig.tight_layout()
    return fig


def plot_health_bucket_prevalence(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, var, color, title in [
        (axes[0], "PhysHlth", BLUE, "Diabetes Prevalence by Poor Physical Health Days"),
        (axes[1], "MentHlth", RED, "Diabetes Prevalence by Poor Mental Health Days"),
    ]:
        b = pd.cut(df[var], bins=[-1, 0, 5, 13, 29, 30], labels=["0", "1-5", "6-13", "14-29", "30"])
        prev = df.groupby(b, observed=False)[TARGET].mean() * 100
        bars = ax.bar(prev.index.astype(str), prev.values, color=color)
        ax.set_title(title)
        ax.set_ylabel("Diabetes prevalence (%)")
        ax.set_xlabel("Days bucket")
        for bar, val in zip(bars, prev.values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_demographic_distribution_panels(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.8))
    for ax, variable, labels, title in [
        (axes[0, 0], "Age", AGE_LABELS, "Age Distribution by Diabetes Status"),
        (axes[1, 0], "Education", EDUCATION_LABELS, "Education Distribution by Diabetes Status"),
        (axes[1, 1], "Income", INCOME_LABELS, "Income Distribution by Diabetes Status"),
    ]:
        ct = pd.crosstab(df[variable], df[TARGET], normalize="columns") * 100
        ct.index = [labels[int(i)] for i in ct.index]
        ct.columns = ["No diabetes", "Diabetes"]
        ct.plot(kind="bar", color=[BLUE, RED], width=0.85, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Column percent")
        ax.tick_params(axis="x", rotation=35 if variable != "Age" else 45)
        ax.legend(title="")

    sex_ct = pd.crosstab(df["Sex"], df[TARGET], normalize="columns") * 100
    sex_ct.index = [SEX_LABELS[int(i)] for i in sex_ct.index]
    sex_ct.columns = ["No diabetes", "Diabetes"]
    sex_ct.plot(kind="bar", color=[BLUE, RED], width=0.8, ax=axes[0, 1])
    axes[0, 1].set_title("Sex Distribution by Diabetes Status")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Column percent")
    axes[0, 1].tick_params(axis="x", rotation=0)
    axes[0, 1].legend(title="")
    fig.tight_layout()
    return fig


def plot_prevalence_gradients(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.8))

    def bar_group(ax, variable, labels_map, title, color):
        prev = df.groupby(variable, observed=False)[TARGET].mean() * 100
        prev.index = [labels_map[int(i)] for i in prev.index]
        bars = ax.bar(prev.index, prev.values, color=color)
        ax.set_title(title)
        ax.set_ylabel("Diabetes prevalence (%)")
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, prev.values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}%", ha="center", fontsize=8.5)

    bar_group(axes[0, 0], "GenHlth", GENHLTH_LABELS, "Prevalence by Self-Rated Health", RED)
    bar_group(axes[0, 1], "Age", AGE_LABELS, "Prevalence by Age Group", BLUE)
    bar_group(axes[1, 0], "Income", INCOME_LABELS, "Prevalence by Income Category", PURPLE)
    bar_group(axes[1, 1], "Education", EDUCATION_LABELS, "Prevalence by Education Level", GREEN)

    fig.tight_layout()
    return fig


def plot_selected_heatmap(df: pd.DataFrame) -> plt.Figure:
    corr = corr_selected(df)
    fig, ax = plt.subplots(figsize=(11.5, 9.2))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("Spearman Correlation Matrix for Section Variables")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    return fig


def plot_all_heatmap(df: pd.DataFrame) -> plt.Figure:
    corr = corr_all(df)
    fig, ax = plt.subplots(figsize=(15.5, 13.2))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.35,
        cbar_kws={"shrink": 0.75},
        ax=ax,
        annot_kws={"size": 6.5},
    )
    ax.set_title("Spearman Correlation Matrix for All Variables")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig


def findings(df: pd.DataFrame) -> List[str]:
    prev = df[TARGET].mean() * 100
    diff_no = df.loc[df[TARGET] == 0, "DiffWalk"].mean() * 100
    diff_yes = df.loc[df[TARGET] == 1, "DiffWalk"].mean() * 100
    phys_no = df.loc[df[TARGET] == 0, "PhysHlth"].mean()
    phys_yes = df.loc[df[TARGET] == 1, "PhysHlth"].mean()
    ment_no = df.loc[df[TARGET] == 0, "MentHlth"].mean()
    ment_yes = df.loc[df[TARGET] == 1, "MentHlth"].mean()
    return [
        f"The analysis dataset contains {len(df):,} respondents with no missing values across the exported section variables. Diabetes prevalence is {prev:.1f}%, so the outcome is imbalanced but large enough for stable subgroup summaries.",
        "The cardiometabolic and lifestyle variables preserved from the upper-half notebook remain in the analysis, but are now restyled to match the later general-health and demographic sections. BMI retains a category-based prevalence view because that is clinically easier to interpret than raw BMI alone.",
        f"The clearest general-health signal comes from self-rated health and physical burden. Walking difficulty rises from {diff_no:.1f}% in the non-diabetes group to {diff_yes:.1f}% in the diabetes group, and mean poor physical health days rise from {phys_no:.2f} to {phys_yes:.2f}.",
        f"Mental health still differs by outcome, but more weakly: mean poor mental health days increase from {ment_no:.2f} to {ment_yes:.2f}. Showing both raw days and percent-of-30-day versions improves readability without changing that conclusion.",
        "Demographic gradients are also pronounced. Diabetes prevalence increases across older age groups and decreases across higher income and education levels, making those variables natural candidates for subgroup performance checks and fairness-oriented discussion later in the project.",
    ]


def variable_dictionary_section() -> pd.DataFrame:
    rows = []
    for var, meaning, typ in [
        ("BMI", "Body Mass Index", "Continuous"),
        ("HighBP", "High blood pressure", "Binary"),
        ("HighChol", "High cholesterol", "Binary"),
        ("CholCheck", "Cholesterol check in past 5 years", "Binary"),
        ("Stroke", "History of stroke", "Binary"),
        ("HeartDiseaseorAttack", "History of coronary heart disease or heart attack", "Binary"),
        ("Smoker", "Smoked at least 100 cigarettes in lifetime", "Binary"),
        ("PhysActivity", "Physical activity in past 30 days, excluding job", "Binary"),
        ("GenHlth", "Self-rated health from excellent to poor", "Ordinal"),
        ("PhysHlth", "Poor physical health days in past 30 days", "Count"),
        ("MentHlth", "Poor mental health days in past 30 days", "Count"),
        ("DiffWalk", "Serious difficulty walking or climbing stairs", "Binary"),
        ("Age", "Age group 18-24 through 80+", "Ordinal"),
        ("Sex", "F/M indicator", "Binary"),
        ("Income", "Income category from <10k to 75k+", "Ordinal"),
        ("Education", "Education level from no school to college graduate", "Ordinal"),
        (TARGET, "No diabetes / diabetes", "Binary"),
    ]:
        rows.append({"Variable": var, "Type": typ, "Meaning": meaning})
    return pd.DataFrame(rows)


def export_overview(df: pd.DataFrame, pdf: Optional[PdfPages] = None) -> Dict[str, object]:
    return {
        "summary": export_table(summary_table(df), "eda_dataset_overview", "Dataset Overview", pdf=pdf),
        "dictionary": export_table(variable_dictionary_section(), "eda_variable_dictionary", "Variable Dictionary", pdf=pdf),
        "outcome_plot": export_plot(plot_outcome_distribution(df), "eda_outcome_distribution", pdf=pdf),
    }


def export_cardiometabolic_section(df: pd.DataFrame, pdf: Optional[PdfPages] = None) -> Dict[str, object]:
    baseline = baseline_table(df, CARDIOMETABOLIC_VARS)
    prevalence = pd.concat(
        [
            prevalence_table(df, "BMI_group").rename(columns={"BMI_group": "Category"}).assign(Variable="BMI_group"),
            *[
                prevalence_table(df, var).rename(columns={var: "Category"}).assign(Variable=var)
                for var in CARDIOMETABOLIC_VARS[1:]
            ],
        ],
        ignore_index=True,
    )[["Variable", "Category", "Count", "Diabetes prevalence"]]
    return {
        "tables": {
            "baseline": export_table(
                baseline,
                "cardiometabolic_baseline_by_diabetes",
                "Cardiometabolic Baseline Comparison by Diabetes Status",
                pdf=pdf,
            ),
            "prevalence": export_table(
                prevalence,
                "cardiometabolic_prevalence_table",
                "Cardiometabolic Diabetes Prevalence by Category",
                pdf=pdf,
            ),
        },
        "plots": {
            "bmi_prevalence": export_plot(plot_bmi_prevalence(df), "cardiometabolic_bmi_prevalence", pdf=pdf),
            "binary_heatmap": export_plot(
                plot_binary_prevalence_heatmap(
                    df,
                    CARDIOMETABOLIC_VARS[1:],
                    "Cardiometabolic Binary Variable Prevalence Heatmap",
                    cmap="Reds",
                ),
                "cardiometabolic_binary_prevalence_heatmap",
                pdf=pdf,
            ),
            "binary_bar": export_plot(
                plot_binary_prevalence_bar(
                    df,
                    CARDIOMETABOLIC_VARS[1:],
                    "Cardiometabolic Binary Variable Prevalence",
                    palette=[BLUE, RED],
                ),
                "cardiometabolic_binary_prevalence_bar",
                pdf=pdf,
            ),
        },
    }


def export_lifestyle_section(df: pd.DataFrame, pdf: Optional[PdfPages] = None) -> Dict[str, object]:
    baseline = baseline_table(df, LIFESTYLE_VARS)
    prevalence = pd.concat(
        [
            prevalence_table(df, var).rename(columns={var: "Category"}).assign(Variable=var)
            for var in LIFESTYLE_VARS
        ],
        ignore_index=True,
    )[["Variable", "Category", "Count", "Diabetes prevalence"]]
    return {
        "tables": {
            "baseline": export_table(
                baseline,
                "lifestyle_baseline_by_diabetes",
                "Lifestyle Baseline Comparison by Diabetes Status",
                pdf=pdf,
            ),
            "prevalence": export_table(
                prevalence,
                "lifestyle_prevalence_table",
                "Lifestyle Diabetes Prevalence by Category",
                pdf=pdf,
            ),
        },
        "plots": {
            "heatmap": export_plot(
                plot_binary_prevalence_heatmap(df, LIFESTYLE_VARS, "Lifestyle Variable Prevalence Heatmap", cmap="Blues"),
                "lifestyle_prevalence_heatmap",
                pdf=pdf,
            ),
            "bar": export_plot(
                plot_binary_prevalence_bar(df, LIFESTYLE_VARS, "Lifestyle Variable Prevalence", palette=[BLUE, RED]),
                "lifestyle_prevalence_bar",
                pdf=pdf,
            ),
        },
    }


def export_general_health_section(df: pd.DataFrame, pdf: Optional[PdfPages] = None) -> Dict[str, object]:
    baseline = baseline_table(df, GENERAL_HEALTH_VARS + ["PhysHlthPct30", "MentHlthPct30"])
    prevalence = pd.concat(
        [
            prevalence_table(df, "GenHlth").rename(columns={"GenHlth": "Category"}).assign(Variable="GenHlth"),
            prevalence_table(df, "DiffWalk").rename(columns={"DiffWalk": "Category"}).assign(Variable="DiffWalk"),
            bucket_prevalence_table(df, "PhysHlth"),
            bucket_prevalence_table(df, "MentHlth"),
        ],
        ignore_index=True,
    )[["Variable", "Category", "Count", "Diabetes prevalence"]]
    return {
        "tables": {
            "baseline": export_table(
                baseline,
                "general_health_baseline_by_diabetes",
                "General Health Baseline Comparison by Diabetes Status",
                pdf=pdf,
            ),
            "prevalence": export_table(
                prevalence,
                "general_health_prevalence_table",
                "General Health Diabetes Prevalence by Category",
                pdf=pdf,
            ),
        },
        "plots": {
            "genhlth_distribution": export_plot(plot_genhlth_distribution(df), "general_health_genhlth_distribution", pdf=pdf),
            "health_burden_panels": export_plot(plot_health_burden_panels(df), "general_health_burden_panels", pdf=pdf),
            "diffwalk_distribution": export_plot(plot_diffwalk_distribution(df), "general_health_diffwalk_distribution", pdf=pdf),
            "bucket_prevalence": export_plot(plot_health_bucket_prevalence(df), "general_health_bucket_prevalence", pdf=pdf),
            "prevalence_heatmap": export_plot(
                plot_category_heatmap_panels(
                    df,
                    ["GenHlth", "PhysHlth", "MentHlth", "DiffWalk"],
                    {
                        "GenHlth": "GenHlth",
                        "PhysHlth": "PhysHlth days bucket",
                        "MentHlth": "MentHlth days bucket",
                        "DiffWalk": "DiffWalk",
                    },
                    "General Health Prevalence Heatmaps",
                ),
                "general_health_prevalence_heatmaps",
                pdf=pdf,
            ),
        },
    }


def export_demographic_section(df: pd.DataFrame, pdf: Optional[PdfPages] = None) -> Dict[str, object]:
    baseline = baseline_table(df, DEMOGRAPHIC_VARS)
    prevalence = pd.concat(
        [
            prevalence_table(df, "Age").rename(columns={"Age": "Category"}).assign(Variable="Age"),
            prevalence_table(df, "Sex").rename(columns={"Sex": "Category"}).assign(Variable="Sex"),
            prevalence_table(df, "Income").rename(columns={"Income": "Category"}).assign(Variable="Income"),
            prevalence_table(df, "Education").rename(columns={"Education": "Category"}).assign(Variable="Education"),
        ],
        ignore_index=True,
    )[["Variable", "Category", "Count", "Diabetes prevalence"]]
    return {
        "tables": {
            "baseline": export_table(
                baseline,
                "demographics_baseline_by_diabetes",
                "Demographic / Fairness Baseline Comparison by Diabetes Status",
                pdf=pdf,
            ),
            "prevalence": export_table(
                prevalence,
                "demographics_prevalence_table",
                "Demographic / Fairness Diabetes Prevalence by Category",
                pdf=pdf,
            ),
        },
        "plots": {
            "distribution_panels": export_plot(
                plot_demographic_distribution_panels(df),
                "demographics_distribution_panels",
                pdf=pdf,
            ),
            "prevalence_gradients": export_plot(
                plot_prevalence_gradients(df),
                "demographics_prevalence_gradients",
                pdf=pdf,
            ),
            "prevalence_heatmap": export_plot(
                plot_category_heatmap_panels(
                    df,
                    ["Age", "Sex", "Income", "Education"],
                    {
                        "Age": "Age",
                        "Sex": "Sex",
                        "Income": "Income",
                        "Education": "Education",
                    },
                    "Demographic / Fairness Prevalence Heatmaps",
                ),
                "demographics_prevalence_heatmaps",
                pdf=pdf,
            ),
        },
    }


def export_relationships_section(df: pd.DataFrame, pdf: Optional[PdfPages] = None) -> Dict[str, object]:
    selected = corr_selected(df).reset_index().rename(columns={"index": "Variable"})
    allc = corr_all(df).reset_index().rename(columns={"index": "Variable"})
    return {
        "tables": {
            "selected_corr": export_table(
                selected,
                "relationships_selected_spearman_corr",
                "Spearman Correlation Matrix for Section Variables",
                pdf=None,
            ),
            "all_corr": export_table(
                allc,
                "relationships_all_variable_spearman_corr",
                "Spearman Correlation Matrix for All Variables",
                pdf=None,
            ),
        },
        "plots": {
            "selected_heatmap": export_plot(plot_selected_heatmap(df), "relationships_selected_heatmap", pdf=pdf),
            "all_heatmap": export_plot(plot_all_heatmap(df), "relationships_all_heatmap", pdf=pdf),
        },
    }


def build_pdf_report(df: pd.DataFrame) -> Path:
    pdf_path = REPORT_DIR / "diabetes_eda_extended_general_health_demographics.pdf"
    with PdfPages(pdf_path) as pdf:
        add_text_page(
            pdf,
            "Unified EDA",
            [
                "Updated formatting across all sections.",
                "Added prevalence heatmaps for general health and demographics.",
                "Correlation heatmaps retained; correlation tables exported separately as CSV and PNG.",
            ],
        )
        export_overview(df, pdf=pdf)
        export_cardiometabolic_section(df, pdf=pdf)
        export_lifestyle_section(df, pdf=pdf)
        export_general_health_section(df, pdf=pdf)
        export_demographic_section(df, pdf=pdf)
        export_relationships_section(df, pdf=pdf)
    return pdf_path


def generate_all_outputs(df: pd.DataFrame) -> Dict[str, object]:
    results = {
        "overview": export_overview(df),
        "cardiometabolic": export_cardiometabolic_section(df),
        "lifestyle": export_lifestyle_section(df),
        "general_health": export_general_health_section(df),
        "demographics": export_demographic_section(df),
        "relationships": export_relationships_section(df),
    }
    results["report_pdf"] = build_pdf_report(df)
    return results


def notebook_cells() -> List[dict]:
    return [
        nbf.v4.new_markdown_cell("# 01_EDA"),
        nbf.v4.new_markdown_cell("## Load Data"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n"
            "import diabetes_eda_extended_report as eda\n"
            "\n"
            "eda.ensure_dirs()\n"
            "eda.set_style()\n"
            "df = eda.load_data()\n"
            "print('Shape:', df.shape)\n"
            "print('Columns:', list(df.columns))"
        ),
        nbf.v4.new_markdown_cell("## Create Analysis DataFrame"),
        nbf.v4.new_code_cell(
            "section_df = df[eda.SECTION_VARS + ['BMI_group', 'PhysHlthPct30', 'MentHlthPct30']].copy()\n"
            "display(section_df.head())\n"
            "display(eda.summary_table(section_df[[c for c in section_df.columns if c in df.columns]]))"
        ),
        nbf.v4.new_markdown_cell("## Shared Style / Helper Functions"),
        nbf.v4.new_code_cell(
            "# Shared settings for exported figures and tables are defined in the companion Python script."
        ),
        nbf.v4.new_markdown_cell("## Cardiometabolic"),
        nbf.v4.new_code_cell(
            "cardio = eda.export_cardiometabolic_section(df)\n"
            "display(cardio['tables']['baseline']['df'])\n"
            "display(cardio['tables']['prevalence']['df'].head(10))\n"
            "display(Image(filename=str(cardio['plots']['bmi_prevalence'])))\n"
            "display(Image(filename=str(cardio['plots']['binary_heatmap'])))\n"
            "display(Image(filename=str(cardio['plots']['binary_bar'])))"
        ),
        nbf.v4.new_markdown_cell("## Lifestyle"),
        nbf.v4.new_code_cell(
            "lifestyle = eda.export_lifestyle_section(df)\n"
            "display(lifestyle['tables']['baseline']['df'])\n"
            "display(lifestyle['tables']['prevalence']['df'])\n"
            "display(Image(filename=str(lifestyle['plots']['heatmap'])))\n"
            "display(Image(filename=str(lifestyle['plots']['bar'])))"
        ),
        nbf.v4.new_markdown_cell("## General Health"),
        nbf.v4.new_code_cell(
            "health = eda.export_general_health_section(df)\n"
            "display(health['tables']['baseline']['df'])\n"
            "display(health['tables']['prevalence']['df'].head(12))\n"
            "display(Image(filename=str(health['plots']['genhlth_distribution'])))\n"
            "display(Image(filename=str(health['plots']['health_burden_panels'])))\n"
            "display(Image(filename=str(health['plots']['diffwalk_distribution'])))\n"
            "display(Image(filename=str(health['plots']['bucket_prevalence'])))\n"
            "display(Image(filename=str(health['plots']['prevalence_heatmap'])))"
        ),
        nbf.v4.new_markdown_cell("## Demographics / Fairness"),
        nbf.v4.new_code_cell(
            "demo = eda.export_demographic_section(df)\n"
            "display(demo['tables']['baseline']['df'])\n"
            "display(demo['tables']['prevalence']['df'].head(12))\n"
            "display(Image(filename=str(demo['plots']['distribution_panels'])))\n"
            "display(Image(filename=str(demo['plots']['prevalence_gradients'])))\n"
            "display(Image(filename=str(demo['plots']['prevalence_heatmap'])))"
        ),
        nbf.v4.new_markdown_cell("## Correlation / Relationships"),
        nbf.v4.new_code_cell(
            "rel = eda.export_relationships_section(df)\n"
            "display(rel['tables']['selected_corr']['df'].head())\n"
            "display(rel['tables']['all_corr']['df'].head())\n"
            "display(Image(filename=str(rel['plots']['selected_heatmap'])))\n"
            "display(Image(filename=str(rel['plots']['all_heatmap'])))"
        ),
        nbf.v4.new_markdown_cell("## Build Full Artifact Set"),
        nbf.v4.new_code_cell(
            "artifacts = eda.generate_all_outputs(df)\n"
            "print('PDF report:', artifacts['report_pdf'])\n"
            "print('All plots saved to:', eda.PLOTS_DIR)\n"
            "print('All tables saved to:', eda.TABLES_DIR)\n"
            "print('Table image files saved to:', eda.TABLE_IMAGES_DIR)"
        ),
    ]


def write_notebook(path: Path = NOTEBOOK_PATH) -> Path:
    nb = nbf.v4.new_notebook()
    nb["cells"] = notebook_cells()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.9"}
    path.write_text(nbf.writes(nb))
    return path


def main() -> None:
    ensure_dirs()
    set_style()
    df = load_data()
    generate_all_outputs(df)
    write_notebook()
    print(f"Unified EDA report written to: {REPORT_DIR / 'diabetes_eda_extended_general_health_demographics.pdf'}")
    print(f"Notebook written to: {NOTEBOOK_PATH}")
    print(f"Plot files written to: {PLOTS_DIR}")
    print(f"Table CSV files written to: {TABLES_DIR}")
    print(f"Table PNG files written to: {TABLE_IMAGES_DIR}")


if __name__ == "__main__":
    main()
