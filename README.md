# Diabetes Risk Prediction, Fairness Evaluation, and Causal Analysis of Physical Activity

This repository analyzes diabetes risk using the CDC Diabetes Health Indicators dataset. The project combines exploratory data analysis, supervised machine learning, subgroup/fairness-oriented summaries, and propensity score matching (PSM) to study the relationship between physical activity and diabetes.

The main outcome is `Diabetes_binary`, and the main causal treatment variable is `PhysActivity`.

## Project Goals

- Describe diabetes prevalence across cardiometabolic, lifestyle, general-health, and demographic variables.
- Build and compare machine learning models for diabetes risk prediction.
- Examine demographic gradients that are relevant for fairness discussion, especially age, sex, education, and income.
- Estimate the association between physical activity and diabetes after balancing observed confounders with propensity score matching.

## Dataset

The notebooks use the UCI Machine Learning Repository dataset `CDC Diabetes Health Indicators` (`uci_id=891`), which is derived from CDC BRFSS health survey data.

- Rows: 253,680 respondents
- Features: 21 health, lifestyle, and demographic predictors
- Target: `Diabetes_binary`
- Missing values: none in the UCI version used by the notebooks
- Sensitive or fairness-relevant variables: `Sex`, `Age`, `Education`, `Income`

The dataset is fetched directly in the notebooks with `ucimlrepo`. The EDA report script currently expects a local CSV path named `diabetes_binary_health_indicators_BRFSS2015.csv`; update `PROJECT_DIR` and `DATA_PATH` in `scripts/diabetes_eda_extended_report.py` before regenerating those outputs locally.

## Repository Structure

```text
.
|-- dataset.ipynb                         # Dataset loading and initial inspection
|-- scripts/
|   |-- 01_EDA.ipynb                      # Generated EDA notebook
|   |-- 01_EDA.py                         # Entry point for EDA report generation
|   `-- diabetes_eda_extended_report.py   # EDA tables, figures, and PDF report generator
|-- 02_Modeling.ipynb                     # Predictive modeling workflow
|-- PSM.ipynb                             # Propensity score matching causal analysis
|-- PSM.html                              # Rendered PSM notebook export
|-- outputs/
|   |-- plots/                            # EDA figures
|   |-- table_images/                     # Rendered summary tables
|   `-- report/                           # PDF EDA report
|-- literature review/                    # Supporting papers and references
|-- full_logistic_regression_table.png    # PSM logistic regression output
|-- logistic_regression_report_table.png  # PSM odds-ratio summary table
`-- LICENSE
```

## Methods

### Exploratory Data Analysis

The EDA pipeline summarizes diabetes prevalence and baseline differences across:

- Cardiometabolic factors: `BMI`, `HighBP`, `HighChol`, `Stroke`, `HeartDiseaseorAttack`
- Lifestyle factors: `Smoker`, `PhysActivity`, diet-related indicators
- General health: `GenHlth`, `PhysHlth`, `MentHlth`, `DiffWalk`
- Demographics and fairness variables: `Age`, `Sex`, `Education`, `Income`

Generated outputs are available in:

- `outputs/plots/`
- `outputs/table_images/`
- `outputs/report/diabetes_eda_extended_general_health_demographics.pdf`

### Machine Learning

`02_Modeling.ipynb` compares three classifiers:

| Model | Accuracy | ROC AUC | PR AUC | Recall, Diabetes=1 | Precision, Diabetes=1 | F1, Diabetes=1 | Brier Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.732 | 0.822 | 0.399 | 0.762 | 0.311 | 0.442 | 0.177 |
| Random Forest | 0.713 | 0.798 | 0.362 | 0.750 | 0.293 | 0.422 | 0.104 |
| XGBoost | 0.738 | 0.829 | 0.428 | 0.772 | 0.319 | 0.451 | 0.097 |

XGBoost performed best overall in this run, with the highest ROC AUC, PR AUC, recall, F1 score, and calibration score among the compared models.

### Propensity Score Matching

`PSM.ipynb` estimates the effect of physical activity using:

- Logistic regression propensity scores
- 1:1 nearest-neighbor matching with replacement
- A caliper of `0.01`
- Standardized mean difference diagnostics
- Post-matching diabetes prevalence comparison
- Logistic regression on the matched sample

The matching diagnostics show substantial covariate balance improvement, with all checked covariates reaching `|SMD| < 0.1` after matching.

Key PSM results:

- Matched diabetes prevalence among physically active respondents: 11.61%
- Matched diabetes prevalence among inactive respondents: 12.11%
- Matched risk difference: -0.50 percentage points
- Estimated odds ratio for physical activity: 0.953
- 95% CI: 0.935 to 0.972
- p-value: `<0.001`

Interpretation: after matching on observed confounders, physical activity remains associated with statistically significant but modestly lower diabetes odds.

## Selected Outputs

| Output | Description |
| --- | --- |
| `outputs/report/diabetes_eda_extended_general_health_demographics.pdf` | Consolidated EDA report |
| `outputs/plots/outcome_distribution.png` | Diabetes outcome distribution |
| `outputs/plots/demographics_prevalence_heatmaps.png` | Diabetes prevalence by demographic groups |
| `outputs/plots/general_health_burden_panels.png` | General-health burden by diabetes status |
| `outputs/plots/relationships_all_heatmap.png` | Spearman correlation heatmap |
| `logistic_regression_report_table.png` | PSM odds-ratio summary |
| `full_logistic_regression_table.png` | Full logistic regression output |

## How to Run

Create an environment with Python 3.10 or newer, then install the main dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost ucimlrepo nbformat jupyter
```

Run the notebooks in this order:

1. `dataset.ipynb`
2. `scripts/01_EDA.ipynb`
3. `02_Modeling.ipynb`
4. `PSM.ipynb`

To regenerate the scripted EDA report, first edit `PROJECT_DIR` and `DATA_PATH` in `scripts/diabetes_eda_extended_report.py`, then run:

```bash
python scripts/01_EDA.py
```

## Important Notes

- The predictive models estimate diabetes risk; they do not prove causality.
- The PSM analysis improves balance on observed variables only. Unmeasured confounding may remain.
- The dataset is observational survey data, so results should be interpreted as population-level associations rather than clinical recommendations.
- Demographic variables are included to support subgroup analysis and fairness discussion, not to encourage discriminatory decision-making.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
