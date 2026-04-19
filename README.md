# Insurance-Risk-Prediction-Model
Insurance Risk Predictive Modelling for Underwriters

Python | SQL | GLM | Scikit-learn | SHAP | Gradio | Matplotlib
Trained on 59,381 Prudential Policyholder Records
Gradient Boosting Model — Test AUC: 0.9011

---

TABLE OF CONTENTS
1. Why This Project
2. Business Problem
3. Why This Topic Matters in Reinsurance
4. Solution Approach
5. Datasets Used
6. Tech Stack and Tools
7. Why Matplotlib Instead of Plotly
8. Project Walkthrough — Step by Step
9. How Predictive Modelling Works in This Project
10. What is SHAP and Why It Matters
11. Actuarial Pricing Model
12. What is the Gradio Tool and Why I Built It
13. Model Results Summary
14. Key Findings
15. How to Run This Project
16. About the Author

---

1. WHY THIS PROJECT
I work in the reinsurance sector and see firsthand how underwriting decisions are made under uncertainty. Traditional underwriting relies heavily on manual review of applications, actuarial tables, and underwriter judgment built over years of experience. While this works, it is slow, inconsistent across teams, and difficult to scale when volumes increase.

I built this project to demonstrate that machine learning and statistical modelling can assist underwriters by providing an objective, data-driven risk score for every policyholder application, along with a clear explanation of why that score was assigned. The goal is not to replace underwriters but to give them a quantitative tool that supports faster and more consistent decisions.

---

2. BUSINESS PROBLEM

When an insurance company receives a new life insurance application, an underwriter must assess how likely that applicant is to make a claim.
This involves reviewing age, health history, BMI, employment status, family medical history, and dozens of other factors. Based on this
assessment, the underwriter assigns a risk class — which determines whether the application is approved, rated with a higher premium, or declined.

The problem is this process is:

- Time-consuming: reviewing 128 variables per application manually takes significant effort
- Subjective: two underwriters looking at the same application may reach different conclusions
- Not scalable: as portfolio volumes grow, the quality of manual review tends to decline
- Difficult to audit: it is hard to document exactly which factors drove a specific decision

A predictive model solves all four problems. It processes all 128 variables consistently in milliseconds, produces a probability score, and via SHAP analysis can explain exactly which factors drove the prediction — making the decision auditable and transparent.

---

3. WHY THIS TOPIC MATTERS IN REINSURANCE

In reinsurance, the risk is layered. A reinsurer takes on the excess risk from primary insurers, so any mispricing at the primary underwriting stage gets amplified when the risk is ceded upward. If a primary insurer systematically underestimates mortality risk on its book — because its underwriting process lacks precision — the reinsurer absorbs losses that were not priced into the treaty.
Predictive modelling at the underwriting stage creates a more accurate primary risk assessment, which means:

- Better quality risk selection entering the reinsurance pool
- More accurate expected loss ratios for treaty pricing
- Reduced adverse selection where only high-risk policies are ceded
- Cleaner data for experience analysis and reserve calculations

From a reinsurance analytics perspective, a model that can reliably separate low-risk and high-risk policyholders at point of application is directly valuable for quota share treaty structuring, excess of loss pricing, and portfolio risk monitoring.

---

4. SOLUTION APPROACH

This project builds a complete end-to-end risk scoring pipeline that:

Step 1 — Loads and validates 59,381 real policyholder records from the Prudential Life Insurance dataset alongside a medical cost dataset for cross-validation of actuarial assumptions.

Step 2 — Queries the data using SQL to answer key business questions about risk distribution across age groups and BMI categories before any modelling begins.

Step 3 — Performs exploratory data analysis to understand the data structure, identify missing values, and visualise how risk is distributed across the population.

Step 4 — Engineers features across 126 policyholder attributes, handles missing data through median imputation, encodes categorical variables, and creates a binary target variable where Response 7 or 8 is classified as High Risk.

Step 5 — Builds a Generalised Linear Model as the actuarial baseline, the same model family used in traditional actuarial pricing.

Step 6 — Benchmarks four machine learning models against each other using 5-fold stratified cross-validation to find the best performing approach.

Step 7 — Applies SHAP analysis to understand which features are driving predictions and why.

Step 8 — Converts model predictions into recommended annual premiums using an actuarial multiplier formula.

Step 9 — Deploys the trained model as a live interactive underwriting tool using Gradio, where an underwriter can input policyholder details and receive an instant risk decision.

---

5. DATASETS USED

Dataset 1: Prudential Life Insurance Assessment
Source: Kaggle — Prudential Life Insurance Assessment Competition
Link: https://www.kaggle.com/competitions/prudential-life-insurance-assessment
Size: 59,381 policyholder records, 128 columns
Target variable: Response — an ordinal risk class from 1 (lowest risk)
to 8 (highest risk)
Features include: normalised age, height, weight, BMI, employment
information, insurance history, family medical history, medical history
flags, medical keyword flags, and product information

This dataset was chosen because it mirrors the actual data structure used in life insurance underwriting — anonymised but realistic
policyholder attributes evaluated for mortality risk. It is one of the few public datasets that closely resembles what a real insurance carrier would use internally.

Dataset 2: Medical Cost Personal Dataset
Source: Kaggle — Insurance dataset by Miri Choi
Link: https://www.kaggle.com/datasets/mirichoi0218/insurance
Size: 1,338 records, 7 columns
Features: age, sex, BMI, number of children, smoker status, region, annual charges
This dataset was used as a secondary validation source. A core actuarial assumption is that smokers carry significantly higher mortality and morbidity risk. The medical cost dataset allows us to verify this assumption empirically — smokers in this dataset pay on average 3 times higher annual insurance charges than non-smokers, which directly corroborates the signal that smoking-related features contribute to higher risk scores in the Prudential model.

---

6. TECH STACK AND TOOLS

Python was the primary language used for the entire pipeline.

Pandas and NumPy were used for data loading, cleaning, transformation, and all numerical operations.

SQLite via Python's sqlite3 module was used to load both datasets into an in-memory relational database and run SQL queries to validate business assumptions before modelling. This demonstrates that the analysis follows a structured data validation process before any machine learning is applied.

Scikit-learn provided the machine learning models including Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. It also provided StandardScaler for feature normalisation, SimpleImputer for missing value handling, train_test_split for data partitioning, and StratifiedKFold for cross-validation.

Statsmodels was used to build the Generalised Linear Model. This is the actuarial industry standard library for GLMs in Python. The Binomial family with a Logit link function was used, which is the correct specification for a binary classification outcome in an actuarial context.

SHAP was used for model explainability. This library computes
SHapley Additive exPlanations, a game-theoretic approach to explaining the contribution of each feature to each individual prediction.

Matplotlib was used for all visualisations. Details on why Matplotlib was chosen over Plotly are covered in Section 7.

Gradio was used to build the interactive underwriting scoring tool.

Google Colab was used as the development environment because it provides free GPU/CPU compute, requires no local setup, and produces a shareable notebook that anyone can open and run with just a browser.

---

7. WHY MATPLOTLIB INSTEAD OF PLOTLY

Plotly produces interactive charts that work well in a live browser session. However, when a Jupyter notebook is uploaded to GitHub, GitHub renders the notebook as a static HTML page. It executes the markdown and displays text outputs and image outputs, but it does not
execute JavaScript.

Plotly charts are rendered using JavaScript at runtime. When GitHub tries to display a notebook containing Plotly charts, it finds JavaScript code in the output cells but cannot run it, so it displays a blank space where the chart should be.

Matplotlib charts work differently. When a Matplotlib figure is rendered and plt.show() is called inside a Jupyter notebook, the chart is converted into a static PNG image and embedded directly into the notebook file as base64-encoded binary data. When GitHub
renders the notebook, it finds a standard image object in the output cell and displays it correctly — no JavaScript required.

---

8. PROJECT WALKTHROUGH — STEP BY STEP

Step 1: Data Loading

Both datasets are uploaded into Google Colab using the files.upload()function. The notebook detects the filename and routes each file to
the correct variable. The Prudential dataset loads as 59,381 rows and 128 columns. The medical cost dataset loads as 1,338 rows and 7 columns. A confirmation summary prints the shape and a preview of both datasets.

Step 2: Sanity Check

Before any analysis, basic quality checks are performed. For the Prudential dataset, the top 10 columns with missing values are identified. Medical_History_10 has 58,824 missing values out of 59,381 total rows — meaning it is 99% empty. This is a known pattern in insurance data where certain medical conditions are only recorded when present, so absence of a value does not mean the data is wrong. It means the condition was not reported, which is itself informative.
The target variable distribution is also checked, revealing that class 8 (highest risk) accounts for 32.8% of policyholders — making this a significantly imbalanced dataset.

Step 3: SQL Layer

Both datasets are loaded into an in-memory SQLite database. Three SQL queries are executed:

Query 1 segments policyholders into Young, Middle, and Senior age bands and computes the average risk score and percentage of high-risk policyholders in each band. This tells us whether age is a meaningful risk discriminator before any modelling.

Query 2 segments policyholders by BMI quartile and computes the average risk score per quartile. This tells us whether BMI has a monotonic relationship with risk — a key assumption in actuarial mortality tables.

Query 3 runs on the medical cost dataset and computes average annual charges by smoker status. Non-smokers average approximately 8,434 dollars per year while smokers average approximately 32,050 dollars — a 3.8 times multiplier. This validates the actuarial assumption about
smoking risk before the model is built.

Step 4: Exploratory Data Analysis

Six charts are produced in a single dashboard:

Chart 1 shows the distribution of policyholders across all 8 risk classes with exact counts. The bimodal shape — high counts at classes 1 and 2, a valley at classes 3 and 4, then rising again to peak at class 8 — indicates that the Prudential book is skewed toward extreme risk outcomes rather than average ones.

Chart 2 is a scatter plot of BMI versus age, coloured by risk class.
The important finding is that red dots (high risk) are not
concentrated in one corner — they are spread across all ages and BMI
values. This tells us that no single variable alone can separate
high-risk from low-risk policyholders, and a multivariate model with
all 126 features is justified.

Chart 3 is a box plot of annual insurance charges by smoker status from the medical cost dataset, confirming the SQL finding visually.

Chart 4 shows the 12 features with the most missing values, revealing the structural missingness pattern discussed in Step 2.

Chart 5 is a correlation heatmap of key actuarial variables. Height, weight, and BMI are highly correlated with each other as expected. Some employment variables show negative correlation with risk, suggesting employed policyholders carry lower mortality risk than unemployed ones — consistent with socioeconomic health literature.

Chart 6 shows what percentage of policyholders fall into each risk class, confirming the class imbalance and justifying the use of AUC rather than accuracy as the primary evaluation metric. Accuracy would be misleading here because a model that simply predicts class 8 for every applicant would achieve 32.8% accuracy by doing nothing useful.

Step 5: Feature Engineering

The 8-class ordinal target is converted into a binary outcome where Response 7 or 8 is labelled High Risk (1) and all other classes are labelled Low Risk (0). This results in a 46.34% high-risk rate in both the training and test sets, which is well-balanced for binary
classification.

An age band variable is created from the normalised age column, grouping policyholders into Young, Middle, and Senior categories.
This adds an interpretable categorical feature that can be used for fairness checks.

The Id column and Product_Info_2 text column are dropped. All remaining categorical columns are label-encoded. Missing values are
imputed using the median strategy, which is robust to the extreme missingness seen in some columns.

The dataset is split 80/20 into training and test sets using stratified splitting, which preserves the 46.34% high-risk ratio in
both splits. StandardScaler is applied to normalise all features to zero mean and unit variance before modelling.

Final split: 47,504 training rows and 11,877 test rows across 126 features.

Step 6: Generalised Linear Model (Actuarial Baseline)

A GLM with a Binomial family and Logit link function is fitted using statsmodels. This is the exact model specification used in actuarial
pricing — it models the log-odds of high risk as a linear combination of all 126 features. The model converges in 9 iterations and achieves
a Pseudo R-squared of 0.3842, which is strong for an insurance dataset. The test AUC is 0.8655 — this becomes the baseline that all machine learning models must beat.

The coefficient plot shows Medical_History_4 and Medical_History_15 as the strongest positive predictors of high risk, while Wt (weight)
and Medical_Keyword_3 are the strongest negative predictors. All top coefficients are statistically significant with p-values near zero.

Step 7: Machine Learning Benchmarking

Four models are trained and evaluated:

Logistic Regression applies L2 regularisation with C=0.5. It is the machine learning equivalent of the GLM and serves as a sanity check. CV AUC: 0.867, Test AUC: 0.866.

Decision Tree with max depth 8 and minimum 50 samples per leaf. CV AUC: 0.867, Test AUC: 0.869. Slightly better than GLM on the test set but with higher cross-validation variance.

Random Forest with 200 trees and max depth 12. CV AUC: 0.885, Test AUC: 0.886. A meaningful improvement over all linear models, demonstrating that non-linear feature interactions contribute predictive power that GLM cannot capture.

Gradient Boosting with 200 estimators, learning rate 0.05, and max depth 5. CV AUC: 0.9001, Test AUC: 0.9011. This is the best model and the only one to exceed the 0.90 AUC threshold. The CV standard deviation of 0.0009 is extremely low, confirming the model generalises consistently and is not overfitting.

All models are evaluated using 5-fold stratified cross-validation on the training set and independently on the held-out test set. The consistency between CV AUC and test AUC across all models confirms that results are genuine and not a product of random chance.

Step 8: SHAP Explainability Analysis

SHAP analysis is applied to the Gradient Boosting model on 1,000 test samples. Three charts are produced:

The bar chart shows global feature importance — the average absolute SHAP value across all 1,000 samples for each feature. BMI has the highest importance by a significant margin, followed by Medical_History_4 and Medical_History_15. This tells us which features the model relies on most across the entire population.

The beeswarm plot shows the direction of each feature's impact. For BMI, red dots (high BMI values) sit to the right of the centre line — meaning high BMI values push predictions toward high risk. Blue dots (low BMI values) sit to the left — meaning low BMI pushes toward low risk. This directional information is what the bar chart alone cannot show.

The waterfall plot explains a single individual prediction. For Policyholder 1, the model predicts a 2.6% risk probability versus the 46.3% pool average. The waterfall shows that Medical_History_15 and Medical_History_4 are the two features pulling the prediction most strongly toward low risk for this specific person. This level of individual explainability is required for regulatory compliance in insurance model governance.

Step 9: Actuarial Pricing Model

The model's predicted probabilities are converted into recommended
annual premiums using the formula:

Recommended Premium = Base Premium x (Individual Risk / Pool Average Risk)

With a base premium of 1,200 dollars and a pool average risk of 46.24%, the multipliers work out as follows:

Preferred tier — probability below 20%, average multiplier 0.12x, average premium 143 dollars. These policyholders are priced well below base rate.

Standard tier — probability 20-40%, multiplier 0.65x, average premium 779 dollars.

Rated tier — probability 40-60%, multiplier 1.10x, average premium 1,314 dollars. These are priced above base rate and would require additional premium loading.

High Risk tier — probability 60-80%, multiplier 1.54x, average premium 1,848 dollars. These require senior underwriter review and medical examination.

Decline tier — probability above 80%, multiplier 1.86x, average premium 2,231 dollars. These applications are declined or referred to a specialty market.

This formula is standard actuarial practice for converting a predicted loss probability into an indicated premium. It mirrors
how experience-rated pricing works in group insurance and treaty
reinsurance.

Step 10: Gradio Interactive Scoring Tool

The trained Gradient Boosting model is wrapped in a Gradio interface that allows an underwriter to input policyholder details and receive an instant risk assessment.

---

ACTUARIAL PRICING MODEL

The pricing model in this project converts machine learning output into a practical underwriting tool using standard actuarial methodology.

The key concept is the risk multiplier. If the pool average predicted risk is 46.24% and a specific policyholder has a predicted risk of 86%, that policyholder carries 1.86 times the average risk of the pool. If the base annual premium is 1,200 dollars, the indicated premium for that policyholder is 1,200 multiplied by 1.86, which equals 2,231 dollars.

This approach mirrors experience rating in insurance, where a risk's own historical loss experience is used to adjust its
premium relative to the class average. Here, the model's predicted probability replaces historical experience for new applicants who have no prior claims history.

The five underwriting tier decisions — Preferred, Standard, Rated, High Risk, and Decline — correspond to standard industry
classification language. The probability thresholds defining each tier (20%, 40%, 60%, 80%) are illustrative and in a production
environment would be calibrated against actual claims data and approved by the appointed actuary.

---
WHAT IS THE GRADIO TOOL AND WHY I BUILT IT

Gradio is an open source Python library that converts a Python function into an interactive web application with no web development required. You define inputs, define outputs, and connect them to a Python function. Gradio handles the user interface, the web server, and the communication layer.

In this project, the Gradio tool wraps the trained Gradient Boosting model and exposes six input fields to the user: normalised age, BMI, weight, height, Medical History 4, and Medical Keyword 3. These six inputs were chosen because they are the top SHAP-identified risk drivers and represent the minimum set of information an underwriter would have at point of application.

When an underwriter enters these values and clicks Assess Risk, the function standardises the inputs using the same scaler used during training, passes them through the Gradient Boosting model, computes the risk probability and multiplier, and returns:

- The predicted risk probability as a percentage
- The risk multiplier relative to the pool average
- The recommended annual premium in dollars
- The underwriting tier decision with action guidance

The tool demonstrates what a production underwriting support system would look like. 

---

13. MODEL RESULTS SUMMARY

Model                   CV AUC    CV Std    Test AUC
GLM Baseline              —         —        0.7800
Logistic Regression     0.8670    0.0011    0.8655
Decision Tree           0.8674    0.0016    0.8692
Random Forest           0.8853    0.0011    0.8858
Gradient Boosting       0.9001    0.0009    0.9011

AUC improvement from baseline GLM to optimised Gradient Boosting: 0.78 to 0.9011 — an absolute improvement of 0.1211 and a relative improvement of 15.5%.

The Gradient Boosting model's CV standard deviation of 0.0009 is the lowest of all models, confirming it is the most stable and reliable choice for production deployment.

---

14. KEY FINDINGS

BMI is the single strongest predictor of high risk by SHAP importance score of 0.21. Policyholders with high normalised BMI consistently receive higher risk scores across all model types. This is consistent with actuarial mortality research linking obesity to increased all-cause mortality.

Medical_History_4 is the second strongest predictor. The coefficient in the GLM is 0.751 — the highest positive coefficient
of any feature — and it appears consistently at the top of both Random Forest importance and SHAP rankings. This suggests it
captures a specific medical condition associated with significantly elevated mortality risk.

Medical_History_15 is the third strongest predictor with a GLM coefficient of 0.681 and strong SHAP importance. These two medical history flags together with BMI account for the majority of predictive power in the model.

The class imbalance with 32.8% of policyholders in risk class 8 means that Prudential's book is not evenly distributed. The company appears to write a disproportionate volume of higher-risk business, possibly because it operates in market segments where other carriers have restricted underwriting.

Smokers pay 3.8 times higher average annual charges than non-smokers in the medical cost dataset, corroborating actuarial mortality tables that assign significantly higher risk loading to tobacco users.

Three medical history columns were over 99% empty. Structural missingness of this kind is common in insurance data and reflects the fact that conditions are recorded only when present. The absence of a value is informative and is handled through median
imputation, which preserves the zero-value signal for binary medical history flags.

GLM and Logistic Regression achieve almost identical AUC scores despite being fitted through different methods (IRLS for GLM versus gradient descent for Logistic Regression). This confirms the data preparation was consistent across both modelling approaches.

