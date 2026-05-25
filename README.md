# Econometric Diagnostics & Regression Modeling in MATLAB

This repository contains a comprehensive collection of MATLAB scripts and econometric procedures developed for descriptive analytics and statistical model verification tasks.

The project focuses on regression diagnostics, autoregressive modeling, econometric assumption testing, and model optimization using real-world datasets.

## Key Features

- **Autoregressive Time-Series Modeling**
  - Flexible AR model construction with customizable lag structures
  - Residual autocorrelation verification using Q-tests and LM tests

- **Regression Diagnostics**
  - Homoscedasticity verification using:
    - White Test
    - Breusch–Pagan Test
  - Multicollinearity analysis using:
    - Variance Inflation Factor (VIF)
    - Condition Index (CI)

- **Model Stability Analysis**
  - CUSUM stability testing for parameter consistency over time

- **Model Optimization**
  - Forward stepwise regression based on t-statistics and significance testing
  - Variable selection under 5% significance level criteria

- **Statistical Inference**
  - Hypothesis testing for econometric model assumptions
  - Residual diagnostics and robustness evaluation

---

## Tech Stack & Methodology

1. Environment: **MATLAB**
2. Input data:
   - Real-world econometric datasets in **CSV** format
   - Electricity price and consumption datasets
   - Wage determination datasets
3. Methodology:
   - Ordinary Least Squares (OLS)
   - Time-series econometrics
   - Diagnostic testing
   - Forward stepwise regression

---

# Repository Contents

1. [code&data](code&data)
   - Contains all MATLAB scripts and helper functions used throughout the project as well as all necessary datasets
   - Main execution script:
     - [third_report.m](code&data/third_report.m)

2. [report.pdf](report.pdf)
   - Full report containing:
     - econometric analysis,
     - statistical test interpretation,
     - model verification,
     - optimization results,
     - conclusions and discussion.

---

## 🛠 Functions Included
- `OLS.m`: Computes regression coefficients and their standard errors
