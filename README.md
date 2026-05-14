# 🏦 Loan Default Prediction System
> Machine Learning system to predict loan repayment risk using real banking data — built with Python, 8 ML models, and a full visual dashboard.

---

## 📌 Project Overview

This project builds an end-to-end **Loan Default Prediction pipeline** for a Finance & Banking use case.  
It takes real loan application data, trains and compares **8 machine learning models**, and generates a **professional 2-page visual dashboard** with risk insights.

**Best Model: AdaBoost → 87.8% Accuracy | AUC = 0.881**

---

## 📂 Dataset

| File | Description |
|---|---|
| `train_u6lujuX_CVtuZ9i.csv` | 614 real loan applications with known outcomes (Loan_Status = Y/N) |
| `test_Y3wMUE5_7gLdaTN.csv` | 367 new applications — model predicts these |

**Features in data:** Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area

---

## ⚙️ How It Works

### Step 1 — Feature Engineering
7 new smart features created from raw columns:

| New Feature | What It Means |
|---|---|
| `TotalIncome` | Applicant + Co-applicant income combined |
| `EMI` | Monthly repayment = LoanAmount ÷ Term |
| `Balance_Income` | Money left after EMI — negative = danger signal |
| `LoanAmount_log` | Log-scaled loan amount (reduces skew) |
| `TotalIncome_log` | Log-scaled income |
| `Debt_to_Income` | Loan ÷ Income — higher = more risk |
| `LTI` | Loan-to-Income ratio — key risk signal |

Also engineered: Income Brackets (Low/Mid/High), Employment Type labels, and a **5-level Risk Tier** combining Credit History + LTI.

---

### Step 2 — Data Cleaning
- Missing categorical values → filled with **mode**
- Missing numerical values → filled with **median**
- Label Encoding for categorical columns (Male→1, Female→0, etc.)
- Final median imputer for any remaining nulls
- All features **standardized** (mean=0, std=1) for SVM and Logistic Regression

---

### Step 3 — 8 ML Models Trained & Compared

| Model | What It Does |
|---|---|
| Logistic Regression | Linear probability classifier — fast, interpretable |
| Decision Tree | Flowchart of yes/no decisions |
| Random Forest | 200 decision trees voting together |
| Gradient Boosting | Trees built sequentially, each correcting previous errors |
| AdaBoost ⭐ | Focuses harder on difficult/misclassified cases — **BEST** |
| SVM | Finds optimal boundary between approved/default |
| KNN | Classifies based on 7 nearest similar applicants |
| Naive Bayes | Probabilistic classifier using Bayes theorem |

Each model evaluated on: **Accuracy, AUC, F1, Precision, Recall, 5-Fold Cross Validation**

---

### Step 4 — Predictions on Test Data
AdaBoost runs on all 367 test applicants and outputs:
- `Predicted_Status` — "Approved" or "Default"
- `Default_Probability` — exact probability score (0 to 1)

Saved to `loan_predictions.csv`

---

### Step 5 — Visual Dashboard (2 PNG Pages)

**Page 1 — Portfolio Risk Insights:**
- KPI tiles: total applications, approval rate, default rate, median loan
- Approval vs Default by income bracket
- Default rate by Employment, Property Area, Marital Status, Gender, Education
- Loan-to-Income ratio distribution (approved vs defaulted)
- Full heatmap — Income Bracket × Risk Tier showing default % in every combination

**Page 2 — ML Intelligence:**
- All 8 models compared side by side
- ROC curves for every model
- Top 14 feature importances (colour-coded by category)
- Confusion matrix for best model
- CV score boxplots showing model stability
- Predicted probability distribution
- Auto-generated key insights panel

---

## 📤 Output Files

| Output | Description |
|---|---|
| `loan_predictions.csv` | 367 test applicants with predicted status + default probability |
| `loan_dashboard_page1.png` | Portfolio risk intelligence dashboard |
| `loan_dashboard_page2.png` | ML model performance dashboard |

---

## 🔑 Key Finding

> **Credit History is everything.**  
> It accounts for **33.9%** of the model's decision.  
> Applicants *without* credit history default **92%** of the time vs only **20%** with history.  
> After credit history, the next biggest risk signals are **income adequacy** and **LTI ratio**.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| ML Models | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter Notebook |

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/loan-default-prediction.git
cd loan-default-prediction

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 3. Place the dataset files in the project folder

# 4. Run the notebook or script
jupyter notebook loan_default_prediction.ipynb
```

---

## 👩‍💻 Author

**Samiksha S Ojha**  
B.E. Computer Science Engineering (Data Science) — Lokmanya Tilak College of Engineering  
[LinkedIn](https://www.linkedin.com/in/samiksha-ojha-84b83a293/) • [GitHub](https://github.com/samikshaojha0-ux)

