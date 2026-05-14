"""
=============================================================
  LOAN DEFAULT PREDICTION — AI/ML Pipeline + Dashboard
  Finance & Banking Application
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score
)

# ─────────────────────────────────────────────
# COLOUR PALETTE  (Dark finance theme)
# ─────────────────────────────────────────────
BG        = "#0d1117"
SURFACE   = "#161b22"
CARD      = "#1c2333"
ACCENT1   = "#58a6ff"   # Electric blue
ACCENT2   = "#3fb950"   # Mint green  (approved)
ACCENT3   = "#f78166"   # Coral red   (default)
ACCENT4   = "#d2a8ff"   # Lavender    (neutral)
ACCENT5   = "#ffa657"   # Amber
TEXT_PRI  = "#e6edf3"
TEXT_SEC  = "#8b949e"
GRID_CLR  = "#21262d"

PALETTE   = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, ACCENT5,
             "#56d364", "#ff7b72", "#79c0ff", "#e3b341"]

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("📂  Loading datasets …")
train_df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
test_df  = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")

print(f"    Train : {train_df.shape}   |   Test : {test_df.shape}")

# ─────────────────────────────────────────────
# 2. EDA SNAPSHOTS  (collected for dashboard)
# ─────────────────────────────────────────────
eda = {}

# Target distribution
target_counts = train_df["Loan_Status"].value_counts()
eda["target_counts"] = target_counts

# Missing values
eda["train_missing"] = train_df.isnull().sum()[train_df.isnull().sum() > 0]

# Numerical stats
eda["num_stats"] = train_df.describe()

# ─────────────────────────────────────────────
# 3. PRE-PROCESSING
# ─────────────────────────────────────────────
print("\n🔧  Pre-processing …")

def preprocess(df, is_train=True):
    df = df.copy()

    # Drop Loan_ID
    df.drop(columns=["Loan_ID"], inplace=True, errors="ignore")

    # ── Impute categoricals
    cat_cols = ["Gender", "Married", "Dependents", "Self_Employed",
                "Credit_History", "Loan_Amount_Term"]
    for col in cat_cols:
        if col in df.columns:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)

    # ── Impute numericals
    num_cols = ["LoanAmount", "ApplicantIncome", "CoapplicantIncome"]
    for col in num_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    # ── Feature engineering
    df["TotalIncome"]       = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["EMI"]               = df["LoanAmount"] / df["Loan_Amount_Term"]
    df["Balance_Income"]    = df["TotalIncome"] - (df["EMI"] * 1000)
    df["LoanAmount_log"]    = np.log1p(df["LoanAmount"])
    df["TotalIncome_log"]   = np.log1p(df["TotalIncome"])
    df["Debt_to_Income"]    = df["LoanAmount"] / (df["TotalIncome"] + 1)
    df["Dependents"]        = df["Dependents"].replace("3+", 3).astype(float)

    # ── Encode
    le = LabelEncoder()
    encode_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]
    for col in encode_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    if is_train and "Loan_Status" in df.columns:
        df["Loan_Status"] = le.fit_transform(df["Loan_Status"])   # Y→1, N→0

    return df

train_proc = preprocess(train_df, is_train=True)
test_proc  = preprocess(test_df,  is_train=False)

FEATURES = [c for c in train_proc.columns if c != "Loan_Status"]
X = train_proc[FEATURES]
y = train_proc["Loan_Status"]
X_test_final = test_proc[FEATURES]

# Align columns
for col in X.columns:
    if col not in X_test_final.columns:
        X_test_final[col] = 0
X_test_final = X_test_final[X.columns]

print(f"    Features : {len(FEATURES)}")
print(f"    Class balance → 0(default):{(y==0).sum()}  1(approved):{(y==1).sum()}")

# Final safeguard imputation
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=FEATURES)
X_test_final = pd.DataFrame(imputer.transform(X_test_final), columns=FEATURES)

# ─────────────────────────────────────────────
# 4. TRAIN / VALIDATE SPLIT
# ─────────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_tr_s  = scaler.fit_transform(X_tr)
X_val_s = scaler.transform(X_val)

# ─────────────────────────────────────────────
# 5. MODELS
# ─────────────────────────────────────────────
print("\n🤖  Training models …")

models = {
    "Logistic Regression"    : LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"          : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"          : RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "Gradient Boosting"      : GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
    "AdaBoost"               : AdaBoostClassifier(n_estimators=100, random_state=42),
    "SVM"                    : SVC(probability=True, kernel="rbf", random_state=42),
    "KNN"                    : KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes"            : GaussianNB(),
}

results = {}
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_tr_s, y_tr)
    preds      = model.predict(X_val_s)
    proba      = model.predict_proba(X_val_s)[:, 1]
    cv_scores  = cross_val_score(model, X_tr_s, y_tr, cv=cv, scoring="accuracy")

    results[name] = {
        "model"     : model,
        "accuracy"  : accuracy_score(y_val, preds),
        "f1"        : f1_score(y_val, preds),
        "precision" : precision_score(y_val, preds),
        "recall"    : recall_score(y_val, preds),
        "roc_auc"   : roc_auc_score(y_val, proba),
        "cv_mean"   : cv_scores.mean(),
        "cv_std"    : cv_scores.std(),
        "preds"     : preds,
        "proba"     : proba,
        "cm"        : confusion_matrix(y_val, preds),
    }
    print(f"    {name:<25} Acc={results[name]['accuracy']:.3f}  "
          f"AUC={results[name]['roc_auc']:.3f}  CV={results[name]['cv_mean']:.3f}±{results[name]['cv_std']:.3f}")

# Best model
best_name = max(results, key=lambda n: results[n]["roc_auc"])
best      = results[best_name]
print(f"\n🏆  Best model : {best_name}  (AUC={best['roc_auc']:.4f})")

# Feature importance (Random Forest)
rf_model = results["Random Forest"]["model"]
feat_imp  = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=False)

# ─────────────────────────────────────────────
# 6. PREDICT ON HOLDOUT TEST SET
# ─────────────────────────────────────────────
X_test_s   = scaler.transform(X_test_final)
test_preds = best["model"].predict(X_test_s)
test_proba = best["model"].predict_proba(X_test_s)[:, 1]

test_df["Predicted_Status"]      = ["Approved" if p == 1 else "Default" for p in test_preds]
test_df["Default_Probability"]   = test_proba.round(4)
test_df.to_csv("c:/Users/Anupam/Downloads/files/loan_predictions.csv", index=False)
print("💾  Predictions saved → loan_predictions.csv")

# ─────────────────────────────────────────────
# 7. DASHBOARD
# ─────────────────────────────────────────────
print("\n📊  Building dashboard …")

plt.rcParams.update({
    "figure.facecolor"  : BG,
    "axes.facecolor"    : CARD,
    "axes.edgecolor"    : GRID_CLR,
    "axes.labelcolor"   : TEXT_PRI,
    "xtick.color"       : TEXT_SEC,
    "ytick.color"       : TEXT_SEC,
    "text.color"        : TEXT_PRI,
    "grid.color"        : GRID_CLR,
    "grid.linewidth"    : 0.6,
    "font.family"       : "DejaVu Sans",
    "axes.titlesize"    : 11,
    "axes.titleweight"  : "bold",
    "axes.titlecolor"   : TEXT_PRI,
    "legend.facecolor"  : SURFACE,
    "legend.edgecolor"  : GRID_CLR,
    "legend.labelcolor" : TEXT_PRI,
})

fig = plt.figure(figsize=(26, 36), facecolor=BG)
fig.suptitle(
    "LOAN DEFAULT PREDICTION — AI/ML INTELLIGENCE DASHBOARD",
    fontsize=22, fontweight="bold", color=ACCENT1,
    y=0.995, fontfamily="DejaVu Sans"
)

gs = gridspec.GridSpec(
    6, 4, figure=fig,
    hspace=0.52, wspace=0.38,
    left=0.05, right=0.97, top=0.975, bottom=0.03
)

# ── helper: card background
def card_bg(ax):
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_color(GRID_CLR)

# ════════════════════════════════════════
# ROW 0 — KPI TILES
# ════════════════════════════════════════
kpis = [
    ("Total Applications", f"{len(train_df):,}", ACCENT1),
    ("Approval Rate",      f"{target_counts.get('Y',0)/len(train_df)*100:.1f}%", ACCENT2),
    ("Default Rate",       f"{target_counts.get('N',0)/len(train_df)*100:.1f}%", ACCENT3),
    (f"Best Model AUC\n({best_name.split()[0]}…)", f"{best['roc_auc']:.4f}", ACCENT4),
]
for i, (title, val, clr) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_color(clr); sp.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.65, val, ha="center", va="center",
            transform=ax.transAxes, fontsize=26, fontweight="bold", color=clr)
    ax.text(0.5, 0.22, title, ha="center", va="center",
            transform=ax.transAxes, fontsize=9.5, color=TEXT_SEC)

# ════════════════════════════════════════
# ROW 1 — Loan Status Donut | Gender Bar | Property Area Pie
# ════════════════════════════════════════

# Donut — Loan Status
ax1 = fig.add_subplot(gs[1, 0])
card_bg(ax1)
sizes  = [target_counts.get("Y", 0), target_counts.get("N", 0)]
colors = [ACCENT2, ACCENT3]
wedges, texts, autotexts = ax1.pie(
    sizes, labels=["Approved", "Default"],
    autopct="%1.1f%%", startangle=90,
    colors=colors,
    pctdistance=0.75,
    wedgeprops=dict(width=0.5, edgecolor=CARD, linewidth=2)
)
for at in autotexts: at.set_color(BG); at.set_fontweight("bold"); at.set_fontsize(9)
for t  in texts:     t.set_color(TEXT_PRI)
ax1.set_title("Loan Status Distribution", pad=10)

# Bar — Gender vs Approval
ax2 = fig.add_subplot(gs[1, 1])
card_bg(ax2)
gender_status = train_df.groupby(["Gender", "Loan_Status"]).size().unstack(fill_value=0)
x = np.arange(len(gender_status))
w = 0.35
ax2.bar(x - w/2, gender_status.get("Y", 0), w, color=ACCENT2, label="Approved", alpha=0.9)
ax2.bar(x + w/2, gender_status.get("N", 0), w, color=ACCENT3, label="Default",  alpha=0.9)
ax2.set_xticks(x); ax2.set_xticklabels(gender_status.index)
ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.4)
ax2.set_title("Gender vs Loan Outcome")
ax2.set_ylabel("Count", color=TEXT_SEC)

# Pie — Property Area
ax3 = fig.add_subplot(gs[1, 2])
card_bg(ax3)
prop_counts = train_df["Property_Area"].value_counts()
ax3.pie(prop_counts, labels=prop_counts.index, autopct="%1.1f%%",
        colors=[ACCENT1, ACCENT4, ACCENT5],
        wedgeprops=dict(edgecolor=CARD, linewidth=1.5),
        pctdistance=0.8)
ax3.set_title("Loan Applications by Property Area")

# Bar — Education vs Approval
ax4 = fig.add_subplot(gs[1, 3])
card_bg(ax4)
edu_status = train_df.groupby(["Education", "Loan_Status"]).size().unstack(fill_value=0)
x = np.arange(len(edu_status))
ax4.bar(x - w/2, edu_status.get("Y", 0), w, color=ACCENT2, label="Approved", alpha=0.9)
ax4.bar(x + w/2, edu_status.get("N", 0), w, color=ACCENT3, label="Default",  alpha=0.9)
ax4.set_xticks(x); ax4.set_xticklabels(edu_status.index, rotation=10, fontsize=8)
ax4.legend(fontsize=8); ax4.grid(axis="y", alpha=0.4)
ax4.set_title("Education vs Loan Outcome")
ax4.set_ylabel("Count", color=TEXT_SEC)

# ════════════════════════════════════════
# ROW 2 — Income Dist | Loan Amount Dist | Credit History | Dependents
# ════════════════════════════════════════

# Income KDE
ax5 = fig.add_subplot(gs[2, 0])
card_bg(ax5)
for status, clr, lbl in [("Y", ACCENT2, "Approved"), ("N", ACCENT3, "Default")]:
    sub = train_df[train_df["Loan_Status"] == status]["ApplicantIncome"]
    sub.plot.kde(ax=ax5, color=clr, label=lbl, linewidth=2)
    ax5.fill_between(
        np.linspace(sub.min(), sub.max(), 200),
        0,
        [ax5.lines[-1].get_ydata()[np.abs(np.linspace(sub.min(), sub.max(), 200) - v).argmin()]
         for v in np.linspace(sub.min(), sub.max(), 200)],
        color=clr, alpha=0.12
    )
ax5.set_xlim(0, 30000)
ax5.set_title("Applicant Income Distribution"); ax5.legend(fontsize=8)
ax5.set_xlabel("Income", color=TEXT_SEC); ax5.grid(alpha=0.3)

# Loan Amount Hist
ax6 = fig.add_subplot(gs[2, 1])
card_bg(ax6)
approved = train_df[train_df["Loan_Status"] == "Y"]["LoanAmount"].dropna()
defaulted = train_df[train_df["Loan_Status"] == "N"]["LoanAmount"].dropna()
ax6.hist(approved,  bins=25, color=ACCENT2, alpha=0.7, label="Approved", density=True)
ax6.hist(defaulted, bins=25, color=ACCENT3, alpha=0.7, label="Default",  density=True)
ax6.set_title("Loan Amount Distribution"); ax6.legend(fontsize=8)
ax6.set_xlabel("Loan Amount (₹ thousands)", color=TEXT_SEC); ax6.grid(alpha=0.3)

# Credit History
ax7 = fig.add_subplot(gs[2, 2])
card_bg(ax7)
ch = train_df.groupby(["Credit_History", "Loan_Status"]).size().unstack(fill_value=0)
x = np.arange(len(ch))
ax7.bar(x - w/2, ch.get("Y", 0), w, color=ACCENT2, label="Approved", alpha=0.9)
ax7.bar(x + w/2, ch.get("N", 0), w, color=ACCENT3, label="Default",  alpha=0.9)
ax7.set_xticks(x); ax7.set_xticklabels(["No History", "Has History"])
ax7.legend(fontsize=8); ax7.grid(axis="y", alpha=0.4)
ax7.set_title("Credit History vs Outcome")

# Dependents
ax8 = fig.add_subplot(gs[2, 3])
card_bg(ax8)
dep = train_df.groupby(["Dependents", "Loan_Status"]).size().unstack(fill_value=0)
x = np.arange(len(dep))
ax8.bar(x - w/2, dep.get("Y", 0), w, color=ACCENT2, label="Approved", alpha=0.9)
ax8.bar(x + w/2, dep.get("N", 0), w, color=ACCENT3, label="Default",  alpha=0.9)
ax8.set_xticks(x); ax8.set_xticklabels(dep.index)
ax8.legend(fontsize=8); ax8.grid(axis="y", alpha=0.4)
ax8.set_title("Dependents vs Loan Outcome")

# ════════════════════════════════════════
# ROW 3 — Correlation Heatmap | Model Comparison Bar
# ════════════════════════════════════════

ax9 = fig.add_subplot(gs[3, :2])
card_bg(ax9)
num_df = train_proc.select_dtypes(include=np.number)
corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(
    corr, mask=mask, cmap=cmap, center=0,
    linewidths=0.5, linecolor=BG,
    annot=True, fmt=".2f", annot_kws={"size": 7, "color": TEXT_PRI},
    ax=ax9, cbar_kws={"shrink": 0.7}
)
ax9.set_title("Feature Correlation Heatmap", pad=10)
ax9.tick_params(axis="x", rotation=45, labelsize=7)
ax9.tick_params(axis="y", rotation=0, labelsize=7)

# Model Comparison
ax10 = fig.add_subplot(gs[3, 2:])
card_bg(ax10)
model_names  = list(results.keys())
accs   = [results[n]["accuracy"]  for n in model_names]
aucs   = [results[n]["roc_auc"]   for n in model_names]
f1s    = [results[n]["f1"]        for n in model_names]

x  = np.arange(len(model_names))
w2 = 0.25
ax10.bar(x - w2,   accs, w2, color=ACCENT1, label="Accuracy", alpha=0.9)
ax10.bar(x,        aucs, w2, color=ACCENT2, label="ROC-AUC",  alpha=0.9)
ax10.bar(x + w2,   f1s,  w2, color=ACCENT4, label="F1-Score", alpha=0.9)
ax10.set_xticks(x)
short_names = [n.replace(" ", "\n") for n in model_names]
ax10.set_xticklabels(short_names, fontsize=7)
ax10.set_ylim(0.5, 1.0)
ax10.legend(fontsize=8); ax10.grid(axis="y", alpha=0.4)
ax10.set_title("Model Performance Comparison (Validation Set)")
ax10.axhline(0.8, color=ACCENT5, linestyle="--", linewidth=1, alpha=0.5)

# ════════════════════════════════════════
# ROW 4 — ROC Curves | Feature Importance | Confusion Matrix (best)
# ════════════════════════════════════════

# ROC
ax11 = fig.add_subplot(gs[4, 0:2])
card_bg(ax11)
ax11.plot([0, 1], [0, 1], "w--", linewidth=1, alpha=0.4)
for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_val, res["proba"])
    lw = 2.5 if name == best_name else 1.2
    ax11.plot(fpr, tpr, linewidth=lw,
              color=PALETTE[i % len(PALETTE)],
              label=f"{name} ({res['roc_auc']:.3f})",
              alpha=0.9 if name == best_name else 0.65)
ax11.set_xlabel("False Positive Rate", color=TEXT_SEC)
ax11.set_ylabel("True Positive Rate",  color=TEXT_SEC)
ax11.set_title("ROC Curves — All Models")
ax11.legend(fontsize=7, loc="lower right")
ax11.grid(alpha=0.3)

# Feature Importance (Top 12)
ax12 = fig.add_subplot(gs[4, 2])
card_bg(ax12)
top_feat = feat_imp.head(12)
colors_fi = [ACCENT1 if i < 3 else ACCENT4 for i in range(len(top_feat))]
bars = ax12.barh(range(len(top_feat)), top_feat.values, color=colors_fi, alpha=0.9)
ax12.set_yticks(range(len(top_feat)))
ax12.set_yticklabels(top_feat.index, fontsize=8)
ax12.invert_yaxis()
ax12.set_xlabel("Importance", color=TEXT_SEC)
ax12.set_title("Top Feature Importances\n(Random Forest)")
ax12.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, top_feat.values):
    ax12.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
              f"{val:.3f}", va="center", fontsize=7, color=TEXT_SEC)

# Confusion Matrix (Best Model)
ax13 = fig.add_subplot(gs[4, 3])
card_bg(ax13)
cm = best["cm"]
im = ax13.imshow(cm, cmap="Blues", aspect="auto")
ax13.set_xticks([0, 1]); ax13.set_yticks([0, 1])
ax13.set_xticklabels(["Default (0)", "Approved (1)"], fontsize=8)
ax13.set_yticklabels(["Default (0)", "Approved (1)"], fontsize=8)
ax13.set_xlabel("Predicted", color=TEXT_SEC)
ax13.set_ylabel("Actual",    color=TEXT_SEC)
ax13.set_title(f"Confusion Matrix\n({best_name})")
for i in range(2):
    for j in range(2):
        clr = "white" if cm[i, j] > cm.max()/2 else TEXT_PRI
        ax13.text(j, i, str(cm[i, j]), ha="center", va="center",
                  fontsize=18, fontweight="bold", color=clr)

# ════════════════════════════════════════
# ROW 5 — CV Box Plot | Default Prob Dist | Metrics Table
# ════════════════════════════════════════

# CV Score Box
ax14 = fig.add_subplot(gs[5, 0])
card_bg(ax14)
cv_data  = []
cv_labels= []
for name, res in results.items():
    cv_s = cross_val_score(res["model"], X_tr_s, y_tr, cv=cv, scoring="accuracy")
    cv_data.append(cv_s)
    cv_labels.append(name.split()[0])
bp = ax14.boxplot(cv_data, patch_artist=True,
                  medianprops=dict(color=ACCENT5, linewidth=2))
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(PALETTE[i % len(PALETTE)])
    patch.set_alpha(0.7)
ax14.set_xticklabels(cv_labels, rotation=45, fontsize=7)
ax14.set_ylabel("CV Accuracy", color=TEXT_SEC)
ax14.set_title("5-Fold CV Score Distribution")
ax14.grid(axis="y", alpha=0.3)

# Default Probability Histogram
ax15 = fig.add_subplot(gs[5, 1])
card_bg(ax15)
true_labels = y_val.values
proba_vals  = best["proba"]
ax15.hist(proba_vals[true_labels == 1], bins=20, color=ACCENT2,
          alpha=0.7, label="Actual Approved", density=True)
ax15.hist(proba_vals[true_labels == 0], bins=20, color=ACCENT3,
          alpha=0.7, label="Actual Default",  density=True)
ax15.axvline(0.5, color=ACCENT5, linestyle="--", linewidth=1.5, label="Threshold 0.5")
ax15.set_xlabel("Predicted Probability", color=TEXT_SEC)
ax15.set_ylabel("Density", color=TEXT_SEC)
ax15.set_title(f"Predicted Probability Distribution\n({best_name})")
ax15.legend(fontsize=8); ax15.grid(alpha=0.3)

# Metrics Summary Table
ax16 = fig.add_subplot(gs[5, 2:])
card_bg(ax16); ax16.axis("off")
metrics_data = []
for name, res in results.items():
    metrics_data.append([
        name,
        f"{res['accuracy']:.3f}",
        f"{res['roc_auc']:.3f}",
        f"{res['f1']:.3f}",
        f"{res['precision']:.3f}",
        f"{res['recall']:.3f}",
        f"{res['cv_mean']:.3f}±{res['cv_std']:.3f}",
    ])

col_labels = ["Model", "Accuracy", "ROC-AUC", "F1", "Precision", "Recall", "CV Score"]

table = ax16.table(
    cellText=metrics_data,
    colLabels=col_labels,
    cellLoc="center", loc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.6)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor(ACCENT1)
        cell.set_text_props(color=BG, fontweight="bold")
    elif metrics_data[row-1][0] == best_name if row > 0 else False:
        cell.set_facecolor("#1e3a5f")
        cell.set_text_props(color=ACCENT1)
    else:
        cell.set_facecolor(CARD if row % 2 == 0 else SURFACE)
        cell.set_text_props(color=TEXT_PRI)
    cell.set_edgecolor(GRID_CLR)

ax16.set_title("Complete Model Metrics Summary", pad=14)

# ── Footer
fig.text(0.5, 0.005,
         f"🏆  Best Model: {best_name}  |  AUC: {best['roc_auc']:.4f}  |  "
         f"Accuracy: {best['accuracy']:.4f}  |  F1: {best['f1']:.4f}  |  "
         f"Predictions saved → loan_predictions.csv",
         ha="center", fontsize=9.5, color=ACCENT5, fontstyle="italic")

out_path = "c:/Users/Anupam/Downloads/files/loan_default_dashboard.png"
plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"\n✅  Dashboard saved → {out_path}")

# ─────────────────────────────────────────────
# 8. CLASSIFICATION REPORT
# ─────────────────────────────────────────────
print("\n" + "═"*55)
print(f"  CLASSIFICATION REPORT — {best_name}")
print("═"*55)
print(classification_report(y_val, best["preds"],
                             target_names=["Default (0)", "Approved (1)"]))

print("\n📋  Feature Importances (Top 10):")
print(feat_imp.head(10).to_string())
print("\n✅  All done!")
