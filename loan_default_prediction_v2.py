"""
=============================================================
  LOAN DEFAULT PREDICTION — Professional Insights Dashboard
  Finance & Banking AI/ML Application  v2.0
=============================================================
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, Wedge
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score
)

# ══════════════════════════════════════════════════════════
#  DESIGN SYSTEM  — Bloomberg Terminal meets Modern Fintech
# ══════════════════════════════════════════════════════════
BG       = "#080c14"
PANEL    = "#0e1520"
CARD     = "#121c2e"
BORDER   = "#1e2d47"
ACCENT   = "#00d4ff"    # Cyan  — primary
GREEN    = "#00e676"    # Approved
RED      = "#ff3d57"    # Default / Risk
GOLD     = "#ffca28"    # Warning / highlight
PURPLE   = "#b388ff"    # Neutral
ORANGE   = "#ff6d00"    # Mid risk
T1       = "#e8f4fd"    # Primary text
T2       = "#7aa3c8"    # Secondary text
T3       = "#3d5a78"    # Muted text

RISK_CMAP   = ["#00e676","#69f0ae","#ffca28","#ff6d00","#ff3d57"]
MODEL_COLORS= [ACCENT, GREEN, RED, GOLD, PURPLE, ORANGE, "#40c4ff","#ea80fc"]

plt.rcParams.update({
    "figure.facecolor" : BG,    "axes.facecolor"   : CARD,
    "axes.edgecolor"   : BORDER,"axes.labelcolor"  : T2,
    "xtick.color"      : T2,    "ytick.color"      : T2,
    "text.color"       : T1,    "grid.color"       : BORDER,
    "grid.linewidth"   : 0.5,   "font.family"      : "DejaVu Sans",
    "axes.titlesize"   : 10,    "axes.titleweight"  : "bold",
    "axes.titlecolor"  : T1,    "legend.facecolor"  : PANEL,
    "legend.edgecolor" : BORDER,"legend.labelcolor" : T2,
    "axes.spines.top"  : False, "axes.spines.right" : False,
})

def styled_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    if title:   ax.set_title(title, color=T1, fontsize=10, fontweight="bold", pad=8)
    if xlabel:  ax.set_xlabel(xlabel, color=T2, fontsize=8)
    if ylabel:  ax.set_ylabel(ylabel, color=T2, fontsize=8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.tick_params(labelsize=7.5)

# ══════════════════════════════════════════════════════════
#  1. LOAD & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════
print("📂  Loading data …")
train = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
test  = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")

raw = train.copy()   # keep original for EDA

# ── Derived insight columns (before encoding)
train["TotalIncome"]    = train["ApplicantIncome"] + train["CoapplicantIncome"].fillna(0)
train["HasCoapplicant"] = (train["CoapplicantIncome"] > 0).astype(int)

# Income brackets  → proxy for profession/lifestyle tier
def income_bracket(inc):
    if   inc <  2500: return "Low\n(<2.5k)"
    elif inc <  5000: return "Lower-Mid\n(2.5-5k)"
    elif inc < 10000: return "Mid\n(5-10k)"
    elif inc < 20000: return "Upper-Mid\n(10-20k)"
    else:             return "High\n(>20k)"

train["IncomeBracket"] = train["ApplicantIncome"].apply(income_bracket)
income_order = ["Low\n(<2.5k)","Lower-Mid\n(2.5-5k)","Mid\n(5-10k)","Upper-Mid\n(10-20k)","High\n(>20k)"]

# Loan-to-Income ratio
train["LTI"] = train["LoanAmount"] / (train["TotalIncome"] / 1000 + 1)

# Employment type label
train["Employment"] = train["Self_Employed"].map({"Yes":"Self-Employed","No":"Salaried"}).fillna("Unknown")

# Risk tier from Credit History + LTI
def risk_tier(row):
    ch  = row.get("Credit_History", 1)
    lti = row.get("LTI", 5)
    if   ch == 1 and lti < 3:  return "Low Risk"
    elif ch == 1 and lti < 6:  return "Moderate"
    elif ch == 1:               return "Elevated"
    elif ch == 0 and lti < 4:  return "High Risk"
    else:                       return "Very High"

train["RiskTier"] = train.apply(risk_tier, axis=1)
risk_order = ["Low Risk","Moderate","Elevated","High Risk","Very High"]

# Dependents clean
train["Dependents"] = train["Dependents"].replace("3+","3+")

# Default flag
train["Default"] = (train["Loan_Status"] == "N").astype(int)
train["Approved"] = (train["Loan_Status"] == "Y").astype(int)

# ══════════════════════════════════════════════════════════
#  2. ML PIPELINE
# ══════════════════════════════════════════════════════════
print("🤖  Training models …")

def ml_preprocess(df, is_train=True):
    d = df.copy()
    d.drop(columns=["Loan_ID"], inplace=True, errors="ignore")
    for col in ["Gender","Married","Dependents","Self_Employed","Credit_History","Loan_Amount_Term"]:
        if col in d.columns: d[col].fillna(d[col].mode()[0], inplace=True)
    for col in ["LoanAmount","ApplicantIncome","CoapplicantIncome"]:
        if col in d.columns: d[col].fillna(d[col].median(), inplace=True)
    d["TotalIncome"]     = d["ApplicantIncome"] + d["CoapplicantIncome"]
    d["EMI"]             = d["LoanAmount"] / d["Loan_Amount_Term"]
    d["Balance_Income"]  = d["TotalIncome"] - d["EMI"] * 1000
    d["LoanAmount_log"]  = np.log1p(d["LoanAmount"])
    d["TotalIncome_log"] = np.log1p(d["TotalIncome"])
    d["Debt_to_Income"]  = d["LoanAmount"] / (d["TotalIncome"] + 1)
    d["LTI"]             = d["LoanAmount"] / (d["TotalIncome"] / 1000 + 1)
    d["HasCoapplicant"]  = (d["CoapplicantIncome"] > 0).astype(int)
    d["Dependents"]      = d["Dependents"].replace("3+", 3)
    le = LabelEncoder()
    for col in ["Gender","Married","Education","Self_Employed","Property_Area"]:
        if col in d.columns: d[col] = le.fit_transform(d[col].astype(str))
    if is_train and "Loan_Status" in d.columns:
        d["Loan_Status"] = le.fit_transform(d["Loan_Status"])
    # drop EDA columns if present
    for c in ["IncomeBracket","Employment","RiskTier","Default","Approved","HasCoapplicant"]:
        d.drop(columns=[c], inplace=True, errors="ignore")
    return d

tr_proc  = ml_preprocess(train, is_train=True)
te_proc  = ml_preprocess(test,  is_train=False)
FEAT     = [c for c in tr_proc.columns if c != "Loan_Status"]
X        = tr_proc[FEAT]; y = tr_proc["Loan_Status"]
X_test_f = te_proc[[c for c in FEAT if c in te_proc.columns]]
for c in FEAT:
    if c not in X_test_f.columns: X_test_f[c] = 0
X_test_f = X_test_f[FEAT]

imp = SimpleImputer(strategy="median")
X        = pd.DataFrame(imp.fit_transform(X), columns=FEAT)
X_test_f = pd.DataFrame(imp.transform(X_test_f), columns=FEAT)

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler   = StandardScaler()
Xtr_s    = scaler.fit_transform(X_tr)
Xval_s   = scaler.transform(X_val)

MODELS = {
    "Logistic Reg."    : LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"    : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"    : RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "Grad. Boosting"   : GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
    "AdaBoost"         : AdaBoostClassifier(n_estimators=100, random_state=42),
    "SVM"              : SVC(probability=True, kernel="rbf", random_state=42),
    "KNN"              : KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes"      : GaussianNB(),
}

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
RES = {}
for name, mdl in MODELS.items():
    mdl.fit(Xtr_s, y_tr)
    preds = mdl.predict(Xval_s)
    proba = mdl.predict_proba(Xval_s)[:,1]
    cvs   = cross_val_score(mdl, Xtr_s, y_tr, cv=cv5, scoring="accuracy")
    RES[name] = dict(
        model=mdl, preds=preds, proba=proba,
        accuracy=accuracy_score(y_val,preds),
        f1=f1_score(y_val,preds),
        precision=precision_score(y_val,preds),
        recall=recall_score(y_val,preds),
        roc_auc=roc_auc_score(y_val,proba),
        cv_mean=cvs.mean(), cv_std=cvs.std(),
        cm=confusion_matrix(y_val,preds),
    )
    print(f"  {name:<18} Acc={RES[name]['accuracy']:.3f}  AUC={RES[name]['roc_auc']:.3f}")

BEST_N = max(RES, key=lambda n: RES[n]["roc_auc"])
BEST   = RES[BEST_N]
print(f"\n🏆  Best: {BEST_N}  AUC={BEST['roc_auc']:.4f}")

rf   = RES["Random Forest"]["model"]
FIMP = pd.Series(rf.feature_importances_, index=FEAT).sort_values(ascending=False)

# Predictions on test set
Xte_s     = scaler.transform(X_test_f)
te_preds  = BEST["model"].predict(Xte_s)
te_proba  = BEST["model"].predict_proba(Xte_s)[:,1]
test["Predicted_Status"]    = ["Approved" if p==1 else "Default" for p in te_preds]
test["Default_Probability"] = te_proba.round(4)
test.to_csv("loan_predictions.csv", index=False)

# ══════════════════════════════════════════════════════════
#  3. COMPUTE INSIGHT DATA
# ══════════════════════════════════════════════════════════
def default_rate(df, col):
    g = df.groupby(col)["Loan_Status"].value_counts(normalize=True).unstack(fill_value=0)
    return (g.get("N", pd.Series(dtype=float)) * 100).round(1)

# Default rates by key segments
dr_income   = default_rate(train, "IncomeBracket")
dr_employ   = default_rate(train, "Employment")
dr_edu      = default_rate(train, "Education")
dr_area     = default_rate(train, "Property_Area")
dr_married  = default_rate(train, "Married")
dr_gender   = default_rate(train, "Gender")
dr_dep      = default_rate(train, "Dependents")
dr_credit   = default_rate(train, "Credit_History")
dr_risk     = default_rate(train, "RiskTier")

# Approval rates by income bracket
app_by_income = train.groupby(["IncomeBracket","Loan_Status"]).size().unstack(fill_value=0)

# LTI distribution by outcome
lti_Y = train[train["Loan_Status"]=="Y"]["LTI"].dropna()
lti_N = train[train["Loan_Status"]=="N"]["LTI"].dropna()

# Loan amount by employment
la_employ = train.groupby(["Employment","Loan_Status"])["LoanAmount"].mean().unstack(fill_value=0)

# ══════════════════════════════════════════════════════════
#  4. BUILD DASHBOARD  (3 pages / panels)
# ══════════════════════════════════════════════════════════
print("\n📊  Rendering dashboard …")

# ─── Helper draw functions ────────────────────────────────

def kpi_box(ax, value, label, color, sub=""):
    ax.set_facecolor(PANEL); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(color); sp.set_linewidth(2)
    ax.text(0.5, 0.60, value, ha="center", va="center",
            transform=ax.transAxes, fontsize=28, fontweight="bold", color=color,
            path_effects=[pe.withStroke(linewidth=4, foreground=BG)])
    ax.text(0.5, 0.25, label, ha="center", va="center",
            transform=ax.transAxes, fontsize=8.5, color=T2)
    if sub:
        ax.text(0.5, 0.08, sub, ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color=T3)

def risk_bar(ax, series, title, color_fn=None):
    vals   = series.sort_values(ascending=True)
    colors = [RED if v > 40 else GOLD if v > 20 else GREEN for v in vals]
    if color_fn: colors = color_fn(vals)
    bars = ax.barh(range(len(vals)), vals, color=colors, height=0.6,
                   edgecolor=BG, linewidth=0.5)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(vals.index, fontsize=8)
    ax.set_xlabel("Default Rate (%)", color=T2, fontsize=8)
    ax.set_title(title, color=T1, fontsize=10, fontweight="bold", pad=8)
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=8, color=T1, fontweight="bold")
    ax.set_xlim(0, max(vals)*1.25 + 5)

# ════════════════════════════════════════════════════
#  PAGE 1 — EXECUTIVE SUMMARY + PORTFOLIO INSIGHTS
# ════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(24, 32), facecolor=BG)

# Header band
fig1.text(0.02, 0.978, "LOAN DEFAULT PREDICTION", fontsize=26, fontweight="bold",
          color=ACCENT, va="top",
          path_effects=[pe.withStroke(linewidth=6, foreground=BG)])
fig1.text(0.02, 0.963, "Executive Summary & Portfolio Risk Intelligence Dashboard — Page 1 of 2",
          fontsize=11, color=T2, va="top")
fig1.text(0.98, 0.970, f"Best Model: {BEST_N}  |  AUC: {BEST['roc_auc']:.4f}  |  Accuracy: {BEST['accuracy']:.2%}",
          fontsize=10, color=GOLD, va="top", ha="right")

gs1 = gridspec.GridSpec(7, 4, figure=fig1,
                         hspace=0.55, wspace=0.38,
                         left=0.05, right=0.97, top=0.955, bottom=0.03)

# ── ROW 0: KPIs ──────────────────────────────────────────
total      = len(train)
approved   = (train["Loan_Status"]=="Y").sum()
defaulted  = (train["Loan_Status"]=="N").sum()
avg_loan   = train["LoanAmount"].median()
avg_income = train["ApplicantIncome"].median()

kpis = [
    (f"{total:,}",          "Total Applications",     ACCENT,  "Training Dataset"),
    (f"{approved/total:.1%}","Approval Rate",          GREEN,   f"{approved} approved"),
    (f"{defaulted/total:.1%}","Default Rate",          RED,     f"{defaulted} defaults"),
    (f"₹{avg_loan:.0f}K",   "Median Loan Amount",     GOLD,    f"Median income ₹{avg_income/1000:.1f}K"),
]
for i,(val,lbl,clr,sub) in enumerate(kpis):
    ax = fig1.add_subplot(gs1[0,i])
    kpi_box(ax, val, lbl, clr, sub)

# ── ROW 1-2: Donut + Income Distribution + Employment ────
# Donut
ax_donut = fig1.add_subplot(gs1[1:3, 0])
ax_donut.set_facecolor(CARD)
for sp in ax_donut.spines.values(): sp.set_color(BORDER)
sizes  = [approved, defaulted]
clrs   = [GREEN, RED]
wedges, texts, autos = ax_donut.pie(
    sizes, colors=clrs, autopct="%1.1f%%", startangle=90,
    pctdistance=0.72, wedgeprops=dict(width=0.52, edgecolor=BG, linewidth=3),
    textprops=dict(color=T1, fontsize=9))
for at in autos: at.set_fontweight("bold"); at.set_color(BG); at.set_fontsize(10)
ax_donut.text(0,0, f"{total}\nApps", ha="center", va="center",
              fontsize=11, color=T1, fontweight="bold")
ax_donut.set_title("Portfolio Overview", color=T1, fontsize=10, fontweight="bold")
ax_donut.legend(["Approved","Default"], loc="lower center", fontsize=8,
                ncol=2, bbox_to_anchor=(0.5,-0.08))

# Income bracket stacked bar
ax_inc = fig1.add_subplot(gs1[1:3, 1:3])
styled_ax(ax_inc, "Approval vs Default by Income Bracket",
          "Income Bracket (Monthly, ₹)", "Number of Applications")
grp = train.groupby(["IncomeBracket","Loan_Status"]).size().unstack(fill_value=0)
grp = grp.reindex([b for b in income_order if b in grp.index])
x   = np.arange(len(grp))
w   = 0.35
b1  = ax_inc.bar(x-w/2, grp.get("Y",0), w, color=GREEN,  alpha=0.85, label="Approved", edgecolor=BG)
b2  = ax_inc.bar(x+w/2, grp.get("N",0), w, color=RED,    alpha=0.85, label="Default",  edgecolor=BG)
ax_inc.set_xticks(x); ax_inc.set_xticklabels(grp.index, fontsize=7.5)
ax_inc.legend(fontsize=8)
# Annotate default rate above each pair
for xi, idx in zip(x, grp.index):
    tot = grp.loc[idx].sum()
    nd  = grp.loc[idx].get("N",0)
    if tot > 0:
        ax_inc.text(xi, max(grp.loc[idx])+2, f"{nd/tot:.0%} def.",
                    ha="center", fontsize=7, color=RED)

# Employment type
ax_emp = fig1.add_subplot(gs1[1:3, 3])
styled_ax(ax_emp, "Default Rate\nby Employment Type", "", "Default Rate (%)")
emp_dr = train.groupby("Employment").apply(
    lambda d: (d["Loan_Status"]=="N").mean()*100).round(1)
colors_e = [RED if v>30 else GOLD for v in emp_dr.values]
bars = ax_emp.bar(emp_dr.index, emp_dr.values, color=colors_e,
                  width=0.5, edgecolor=BG, linewidth=0.5)
for bar, val in zip(bars, emp_dr.values):
    ax_emp.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
                color=T1, fontweight="bold")
ax_emp.set_ylim(0, emp_dr.max()*1.4)
ax_emp.grid(axis="y", alpha=0.2, linestyle="--")

# ── ROW 3: Default rate bars ──────────────────────────────
# By Risk Tier
ax_rt = fig1.add_subplot(gs1[3, :2])
risk_bar(ax_rt,
         dr_risk.reindex([r for r in risk_order if r in dr_risk.index]),
         "⚠  Default Rate by Risk Tier (Credit History × Loan-to-Income)")

# By Property Area
ax_area = fig1.add_subplot(gs1[3, 2])
styled_ax(ax_area, "Default Rate by\nProperty Area", "", "Default Rate (%)")
area_dr = train.groupby("Property_Area").apply(
    lambda d: (d["Loan_Status"]=="N").mean()*100).sort_values(ascending=False)
clrs_a  = [RED if v>35 else GOLD if v>25 else GREEN for v in area_dr]
bars    = ax_area.bar(area_dr.index, area_dr.values, color=clrs_a,
                      width=0.5, edgecolor=BG)
for bar, val in zip(bars, area_dr.values):
    ax_area.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{val:.1f}%", ha="center", fontsize=9, color=T1, fontweight="bold")
ax_area.set_ylim(0, area_dr.max()*1.35)

# By Married
ax_mar = fig1.add_subplot(gs1[3, 3])
styled_ax(ax_mar, "Default Rate by\nMarital Status", "", "Default Rate (%)")
mar_dr = train.groupby("Married").apply(
    lambda d: (d["Loan_Status"]=="N").mean()*100).sort_values(ascending=False)
clrs_m = [RED if v>35 else GOLD if v>25 else GREEN for v in mar_dr]
bars   = ax_mar.bar(mar_dr.index, mar_dr.values, color=clrs_m, width=0.5, edgecolor=BG)
for bar, val in zip(bars, mar_dr.values):
    ax_mar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{val:.1f}%", ha="center", fontsize=9, color=T1, fontweight="bold")
ax_mar.set_ylim(0, mar_dr.max()*1.35)

# ── ROW 4: LTI + Loan Amount + Dependents ────────────────
# LTI Distribution
ax_lti = fig1.add_subplot(gs1[4, :2])
styled_ax(ax_lti, "Loan-to-Income Ratio Distribution by Outcome",
          "Loan-to-Income Ratio", "Density")
ax_lti.hist(lti_Y.clip(0,20), bins=30, color=GREEN, alpha=0.65,
            density=True, label="Approved", edgecolor=BG)
ax_lti.hist(lti_N.clip(0,20), bins=30, color=RED,   alpha=0.65,
            density=True, label="Default",  edgecolor=BG)
med_Y = lti_Y.median(); med_N = lti_N.median()
ax_lti.axvline(med_Y, color=GREEN, linestyle="--", linewidth=1.5,
               label=f"Med Approved={med_Y:.1f}")
ax_lti.axvline(med_N, color=RED,   linestyle="--", linewidth=1.5,
               label=f"Med Default={med_N:.1f}")
ax_lti.legend(fontsize=7.5); ax_lti.set_xlim(0,20)
ax_lti.text(0.97, 0.92,
            "Higher LTI → Greater\ndefault probability",
            transform=ax_lti.transAxes, ha="right", fontsize=7.5,
            color=GOLD, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL, edgecolor=GOLD, alpha=0.8))

# Loan Amount violin-style (using box)
ax_la = fig1.add_subplot(gs1[4, 2])
styled_ax(ax_la, "Loan Amount\nby Outcome (₹K)", "", "Loan Amount (₹K)")
bp = ax_la.boxplot(
    [train[train["Loan_Status"]=="Y"]["LoanAmount"].dropna(),
     train[train["Loan_Status"]=="N"]["LoanAmount"].dropna()],
    labels=["Approved","Default"], patch_artist=True,
    medianprops=dict(color=GOLD, linewidth=2),
    whiskerprops=dict(color=T2), capprops=dict(color=T2),
    flierprops=dict(marker=".", color=T3, alpha=0.4, markersize=3))
bp["boxes"][0].set_facecolor(GREEN+"44")
bp["boxes"][1].set_facecolor(RED+"44")
for patch, clr in zip(bp["boxes"],[GREEN,RED]):
    patch.set_edgecolor(clr); patch.set_linewidth(1.5)

# Dependents
ax_dep = fig1.add_subplot(gs1[4, 3])
styled_ax(ax_dep, "Default Rate by\nNo. of Dependents", "Dependents","Default Rate (%)")
dep_dr = train.groupby("Dependents").apply(
    lambda d: (d["Loan_Status"]=="N").mean()*100).sort_index()
clrs_d = [RED if v>35 else GOLD if v>25 else GREEN for v in dep_dr]
bars   = ax_dep.bar(dep_dr.index.astype(str), dep_dr.values,
                    color=clrs_d, width=0.5, edgecolor=BG)
for bar, val in zip(bars, dep_dr.values):
    ax_dep.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{val:.0f}%", ha="center", fontsize=9, color=T1, fontweight="bold")
ax_dep.set_ylim(0, dep_dr.max()*1.4)

# ── ROW 5: Gender + Education + Credit History ───────────
ax_gen = fig1.add_subplot(gs1[5, 0])
styled_ax(ax_gen, "Default Rate\nby Gender","","Default Rate (%)")
gen_dr = train.groupby("Gender").apply(
    lambda d: (d["Loan_Status"]=="N").mean()*100).sort_values(ascending=False)
clrs_g = [RED if v>35 else GOLD if v>20 else GREEN for v in gen_dr]
bars   = ax_gen.bar(gen_dr.index, gen_dr.values, color=clrs_g, width=0.5, edgecolor=BG)
for bar, val in zip(bars, gen_dr.values):
    ax_gen.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{val:.1f}%", ha="center", fontsize=9, color=T1, fontweight="bold")
ax_gen.set_ylim(0, gen_dr.max()*1.4)

ax_edu = fig1.add_subplot(gs1[5, 1])
styled_ax(ax_edu, "Default Rate\nby Education","","Default Rate (%)")
edu_dr = train.groupby("Education").apply(
    lambda d: (d["Loan_Status"]=="N").mean()*100).sort_values(ascending=False)
clrs_e = [RED if v>35 else GOLD if v>20 else GREEN for v in edu_dr]
bars   = ax_edu.bar(edu_dr.index, edu_dr.values, color=clrs_e, width=0.5, edgecolor=BG)
for bar, val in zip(bars, edu_dr.values):
    ax_edu.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{val:.1f}%", ha="center", fontsize=9, color=T1, fontweight="bold")
ax_edu.set_ylim(0, edu_dr.max()*1.4)

ax_cr = fig1.add_subplot(gs1[5, 2:])
styled_ax(ax_cr, "💳  Credit History Impact on Default Rate — The #1 Predictor",
          "Credit History","Default Rate (%)")
cr_dr = train.groupby("Credit_History").apply(
    lambda d: (d["Loan_Status"]=="N").mean()*100)
labels = {0.0:"No Credit\nHistory", 1.0:"Has Credit\nHistory"}
clrs_c = [RED, GREEN]
x_pos  = [0, 1]
bars   = ax_cr.bar(x_pos, [cr_dr.get(0.0,0), cr_dr.get(1.0,0)],
                   color=clrs_c, width=0.4, edgecolor=BG)
ax_cr.set_xticks(x_pos); ax_cr.set_xticklabels(list(labels.values()), fontsize=9)
for bar, val in zip(bars, [cr_dr.get(0.0,0), cr_dr.get(1.0,0)]):
    ax_cr.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
               f"{val:.1f}%", ha="center", fontsize=14, color=T1, fontweight="bold")
ax_cr.set_ylim(0, 100)
diff = cr_dr.get(0.0,0) - cr_dr.get(1.0,0)
ax_cr.text(0.5, 0.82, f"Applicants with no credit history default\n{diff:.0f}% MORE than those with history",
           transform=ax_cr.transAxes, ha="center", fontsize=9, color=GOLD, style="italic",
           bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL, edgecolor=GOLD, alpha=0.9))

# ── ROW 6: Heatmap (Income × Risk) ───────────────────────
ax_heat = fig1.add_subplot(gs1[6, :])
styled_ax(ax_heat, "Default Rate Heatmap — Income Bracket × Risk Tier (%)")
pivot = train.groupby(["IncomeBracket","RiskTier"]).apply(
    lambda d: (d["Loan_Status"]=="N").mean()*100).unstack(fill_value=np.nan)
pivot = pivot.reindex([b for b in income_order if b in pivot.index])
pivot = pivot.reindex([r for r in risk_order if r in pivot.columns], axis=1)
sns.heatmap(pivot, ax=ax_heat, cmap="RdYlGn_r", annot=True, fmt=".0f",
            linewidths=0.5, linecolor=BG,
            annot_kws={"size":9, "weight":"bold"},
            cbar_kws={"shrink":0.6, "label":"Default Rate (%)"},
            vmin=0, vmax=100)
ax_heat.set_xlabel("Risk Tier", color=T2, fontsize=9)
ax_heat.set_ylabel("Income Bracket", color=T2, fontsize=9)
ax_heat.tick_params(labelsize=8)

# Footer
fig1.text(0.5, 0.012, "🟢 Green < 20% default  |  🟡 Amber 20–35%  |  🔴 Red > 35% default  |  Page 1/2",
          ha="center", fontsize=9, color=T3)

fig1.savefig("loan_dashboard_page1.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("✅  Page 1 saved")

# ════════════════════════════════════════════════════
#  PAGE 2 — ML MODEL PERFORMANCE & DECISION INSIGHTS
# ════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(24, 28), facecolor=BG)

fig2.text(0.02, 0.978, "ML MODEL PERFORMANCE", fontsize=26, fontweight="bold",
          color=ACCENT, va="top",
          path_effects=[pe.withStroke(linewidth=6, foreground=BG)])
fig2.text(0.02, 0.963, "Algorithm Comparison, Feature Intelligence & Decision Analytics — Page 2 of 2",
          fontsize=11, color=T2, va="top")

gs2 = gridspec.GridSpec(5, 4, figure=fig2,
                         hspace=0.52, wspace=0.38,
                         left=0.05, right=0.97, top=0.955, bottom=0.03)

# ── ROW 0: Model KPIs ────────────────────────────────────
model_kpis = [
    (f"{BEST['accuracy']:.2%}",  f"Best Accuracy\n({BEST_N})",    ACCENT),
    (f"{BEST['roc_auc']:.4f}",   f"Best AUC\n({BEST_N})",         GREEN),
    (f"{BEST['f1']:.4f}",        f"Best F1 Score\n({BEST_N})",    GOLD),
    (f"{len(FEAT)}",             "Features\nEngineered",           PURPLE),
]
for i,(val,lbl,clr) in enumerate(model_kpis):
    ax = fig2.add_subplot(gs2[0,i])
    kpi_box(ax, val, lbl, clr)

# ── ROW 1: Model Comparison + ROC ────────────────────────
ax_cmp = fig2.add_subplot(gs2[1, :2])
styled_ax(ax_cmp, "Model Performance Comparison (Validation Set)")
names = list(RES.keys())
accs  = [RES[n]["accuracy"]  for n in names]
aucs  = [RES[n]["roc_auc"]   for n in names]
f1s   = [RES[n]["f1"]        for n in names]
x     = np.arange(len(names)); w3 = 0.26
ax_cmp.bar(x-w3,   accs, w3, color=ACCENT,  alpha=0.85, label="Accuracy", edgecolor=BG)
ax_cmp.bar(x,      aucs, w3, color=GREEN,   alpha=0.85, label="ROC-AUC",  edgecolor=BG)
ax_cmp.bar(x+w3,   f1s,  w3, color=GOLD,   alpha=0.85, label="F1-Score", edgecolor=BG)
ax_cmp.set_xticks(x)
ax_cmp.set_xticklabels([n.replace(" ","\n") for n in names], fontsize=7)
ax_cmp.set_ylim(0.5, 1.02)
ax_cmp.legend(fontsize=8); ax_cmp.grid(axis="y", alpha=0.25, linestyle="--")
ax_cmp.axhline(0.8, color=RED, linestyle=":", linewidth=1.2, alpha=0.6, label="0.8 threshold")
# Highlight best
best_idx = names.index(BEST_N)
ax_cmp.axvspan(best_idx-0.45, best_idx+0.45, alpha=0.06, color=ACCENT, zorder=0)
ax_cmp.text(best_idx, 1.015, "★ BEST", ha="center", fontsize=7.5,
            color=ACCENT, fontweight="bold")

# ROC Curves
ax_roc = fig2.add_subplot(gs2[1, 2:])
styled_ax(ax_roc, "ROC Curves — All Models",
          "False Positive Rate", "True Positive Rate")
ax_roc.plot([0,1],[0,1],"--", color=T3, linewidth=1)
for i,(name,res) in enumerate(RES.items()):
    fpr,tpr,_ = roc_curve(y_val, res["proba"])
    lw   = 2.8 if name==BEST_N else 1.2
    alph = 1.0 if name==BEST_N else 0.6
    ax_roc.plot(fpr, tpr, linewidth=lw, alpha=alph,
                color=MODEL_COLORS[i % len(MODEL_COLORS)],
                label=f"{name} ({res['roc_auc']:.3f})")
ax_roc.legend(fontsize=7, loc="lower right")
ax_roc.fill_between([0,1],[0,1], alpha=0.04, color=T3)

# ── ROW 2: Feature Importance + Confusion Matrix ─────────
ax_fi = fig2.add_subplot(gs2[2, :2])
styled_ax(ax_fi, "Top Feature Importances (Random Forest)",
          "Importance Score", "")
top14 = FIMP.head(14)
clrs_fi = []
for feat in top14.index:
    if   "Credit"  in feat: clrs_fi.append(GOLD)
    elif "Income"  in feat or "income" in feat: clrs_fi.append(GREEN)
    elif "Loan"    in feat or "EMI"    in feat: clrs_fi.append(RED)
    elif "Debt"    in feat or "LTI"    in feat: clrs_fi.append(ORANGE)
    else: clrs_fi.append(PURPLE)
bars = ax_fi.barh(range(len(top14)), top14.values, color=clrs_fi,
                  height=0.65, edgecolor=BG)
ax_fi.set_yticks(range(len(top14)))
ax_fi.set_yticklabels(top14.index, fontsize=8)
ax_fi.invert_yaxis()
ax_fi.grid(axis="x", alpha=0.2, linestyle="--")
for bar, val in zip(bars, top14.values):
    ax_fi.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
               f"{val:.3f}", va="center", fontsize=7.5, color=T1)
# Legend
legend_els = [
    Line2D([0],[0],color=GOLD,  lw=6, label="Credit History"),
    Line2D([0],[0],color=GREEN, lw=6, label="Income Features"),
    Line2D([0],[0],color=RED,   lw=6, label="Loan Features"),
    Line2D([0],[0],color=ORANGE,lw=6, label="Risk Ratios"),
    Line2D([0],[0],color=PURPLE,lw=6, label="Demographics"),
]
ax_fi.legend(handles=legend_els, fontsize=7, loc="lower right")

# Confusion matrix
ax_cm = fig2.add_subplot(gs2[2, 2])
ax_cm.set_facecolor(CARD)
for sp in ax_cm.spines.values(): sp.set_color(BORDER)
cm = BEST["cm"]
im = ax_cm.imshow(cm, cmap="Blues", aspect="auto", vmin=0)
ax_cm.set_xticks([0,1]); ax_cm.set_yticks([0,1])
ax_cm.set_xticklabels(["Pred: Default","Pred: Approved"], fontsize=8)
ax_cm.set_yticklabels(["Actual: Default","Actual: Approved"], fontsize=8)
ax_cm.set_title(f"Confusion Matrix\n({BEST_N})", color=T1, fontsize=10, fontweight="bold")
labels_cm = [["TN","FP"],["FN","TP"]]
for i in range(2):
    for j in range(2):
        clr = "white" if cm[i,j] > cm.max()/2 else T1
        ax_cm.text(j, i, f"{cm[i,j]}\n{labels_cm[i][j]}", ha="center", va="center",
                   fontsize=14, fontweight="bold", color=clr)

# Per-class metrics
ax_met = fig2.add_subplot(gs2[2, 3])
ax_met.set_facecolor(CARD)
for sp in ax_met.spines.values(): sp.set_color(BORDER)
ax_met.axis("off")
ax_met.set_title("Per-Class Metrics\n(Best Model)", color=T1, fontsize=10, fontweight="bold")
cr_dict = classification_report(y_val, BEST["preds"],
                                 target_names=["Default","Approved"],
                                 output_dict=True)
rows = [
    ["Metric",       "Default", "Approved"],
    ["Precision",    f"{cr_dict['Default']['precision']:.3f}",
                     f"{cr_dict['Approved']['precision']:.3f}"],
    ["Recall",       f"{cr_dict['Default']['recall']:.3f}",
                     f"{cr_dict['Approved']['recall']:.3f}"],
    ["F1-Score",     f"{cr_dict['Default']['f1-score']:.3f}",
                     f"{cr_dict['Approved']['f1-score']:.3f}"],
    ["Support",      str(cr_dict['Default']['support']),
                     str(cr_dict['Approved']['support'])],
    ["Accuracy",     f"{cr_dict['accuracy']:.3f}", "—"],
]
tbl = ax_met.table(cellText=rows[1:], colLabels=rows[0],
                   cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2.0)
for (r,c), cell in tbl.get_celld().items():
    cell.set_edgecolor(BORDER)
    if r == 0:
        cell.set_facecolor(ACCENT); cell.set_text_props(color=BG, fontweight="bold")
    elif c == 1:
        cell.set_facecolor(RED+"33"); cell.set_text_props(color=T1)
    elif c == 2:
        cell.set_facecolor(GREEN+"33"); cell.set_text_props(color=T1)
    else:
        cell.set_facecolor(PANEL); cell.set_text_props(color=T2)

# ── ROW 3: CV Distribution + Prob Distribution + Income-LTI scatter ──
ax_cv = fig2.add_subplot(gs2[3, 0])
styled_ax(ax_cv, "5-Fold CV Score Distribution", "","CV Accuracy")
cv_data = []
for name, res in RES.items():
    cv_data.append(cross_val_score(res["model"], Xtr_s, y_tr, cv=cv5, scoring="accuracy"))
bp2 = ax_cv.boxplot(cv_data, patch_artist=True,
                    medianprops=dict(color=GOLD, linewidth=2),
                    whiskerprops=dict(color=T2),
                    capprops=dict(color=T2),
                    flierprops=dict(marker=".", color=T3, alpha=0.5))
for i, patch in enumerate(bp2["boxes"]):
    patch.set_facecolor(MODEL_COLORS[i % len(MODEL_COLORS)]+"66")
    patch.set_edgecolor(MODEL_COLORS[i % len(MODEL_COLORS)])
ax_cv.set_xticklabels([n.split()[0] for n in RES.keys()], rotation=45, fontsize=7)
ax_cv.grid(axis="y", alpha=0.25, linestyle="--")

# Probability distribution
ax_prob = fig2.add_subplot(gs2[3, 1])
styled_ax(ax_prob, "Predicted Probability\nDistribution",
          "Default Probability","Density")
true_l = y_val.values; prob_v = BEST["proba"]
ax_prob.hist(prob_v[true_l==1], bins=20, color=GREEN, alpha=0.7,
             density=True, label="Actual Approved", edgecolor=BG)
ax_prob.hist(prob_v[true_l==0], bins=20, color=RED,   alpha=0.7,
             density=True, label="Actual Default", edgecolor=BG)
ax_prob.axvline(0.5, color=GOLD, linestyle="--", linewidth=1.8, label="Threshold 0.5")
ax_prob.legend(fontsize=7.5)

# Scatter: Total Income vs Loan Amount coloured by prediction
ax_scat = fig2.add_subplot(gs2[3, 2:])
styled_ax(ax_scat, "Applicant Income vs Loan Amount\n(Coloured by Actual Loan Status)",
          "Total Income (₹)","Loan Amount (₹K)")
for status, clr, lbl, mk in [("Y",GREEN,"Approved","o"),("N",RED,"Default","x")]:
    sub = train[train["Loan_Status"]==status]
    ax_scat.scatter(sub["TotalIncome"].clip(0,60000),
                    sub["LoanAmount"].clip(0,500),
                    c=clr, alpha=0.45, s=18, label=lbl,
                    marker=mk, edgecolors="none")
ax_scat.legend(fontsize=8)
ax_scat.set_xlim(0,60000); ax_scat.set_ylim(0,500)
ax_scat.text(0.97, 0.05,
             "Higher income + lower loan\n→ Lower default risk",
             transform=ax_scat.transAxes, ha="right", fontsize=8,
             color=GOLD, style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL,
                       edgecolor=GOLD, alpha=0.85))

# ── ROW 4: Key Insights Text Panel ───────────────────────
ax_ins = fig2.add_subplot(gs2[4, :])
ax_ins.set_facecolor(PANEL)
for sp in ax_ins.spines.values(): sp.set_color(ACCENT); sp.set_linewidth(1.5)
ax_ins.axis("off")

insights = [
    ("🏆", GOLD,  "Best Model",
     f"{BEST_N} achieves AUC={BEST['roc_auc']:.4f}, Accuracy={BEST['accuracy']:.2%}, F1={BEST['f1']:.4f}"),
    ("💳", RED,   "#1 Risk Factor",
     f"Credit History is the strongest predictor ({FIMP.get('Credit_History',0):.1%} importance). "
     f"No-history applicants default {cr_dr.get(0.0,0):.0f}% vs {cr_dr.get(1.0,0):.0f}% with history"),
    ("💰", GREEN, "Income Effect",
     f"Low-income (<₹2.5K) applicants have significantly higher default rates. "
     f"Self-employed vs salaried: {emp_dr.get('Self-Employed',0):.1f}% vs {emp_dr.get('Salaried',0):.1f}% default"),
    ("🏙", ACCENT,"Location Risk",
     f"Property area matters: {area_dr.index[0]} has highest default rate ({area_dr.iloc[0]:.1f}%). "
     f"Semiurban applicants tend to have better repayment"),
    ("📊", PURPLE,"LTI Insight",
     f"Median Loan-to-Income ratio for defaults ({med_N:.2f}) is higher than approved ({med_Y:.2f}). "
     f"LTI > 5 is a strong default signal"),
]

col_w = 0.195
for i, (icon, clr, title, text) in enumerate(insights):
    xpos = 0.01 + i * col_w
    ax_ins.text(xpos, 0.88, f"{icon} {title}", transform=ax_ins.transAxes,
                fontsize=9.5, color=clr, fontweight="bold", va="top")
    ax_ins.text(xpos, 0.68, text, transform=ax_ins.transAxes,
                fontsize=7.8, color=T2, va="top", wrap=True,
                bbox=None)
    if i < 4:
        ax_ins.axvline(xpos + col_w - 0.01, color=BORDER, linewidth=1,
                       ymin=0.1, ymax=0.9)

ax_ins.set_title("  KEY INSIGHTS & ACTIONABLE FINDINGS", color=ACCENT,
                 fontsize=11, fontweight="bold", loc="left", pad=10)

# Footer
fig2.text(0.5, 0.01,
          f"Model trained on {len(train)} applications  |  "
          f"Validation set: {len(y_val)} applications  |  "
          f"Test predictions saved to loan_predictions.csv",
          ha="center", fontsize=9, color=T3)

fig2.savefig("loan_dashboard_page2.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("✅  Page 2 saved")

print("\n" + "═"*60)
print(f"  CLASSIFICATION REPORT — {BEST_N}")
print("═"*60)
print(classification_report(y_val, BEST["preds"],
                             target_names=["Default (0)","Approved (1)"]))
print(f"\n📋  Top 5 Features:")
print(FIMP.head(5).to_string())
print("\n✅  All done!  →  loan_dashboard_page1.png  +  loan_dashboard_page2.png")
