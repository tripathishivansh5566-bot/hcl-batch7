"""
Telecom Churn & Customer Behaviour
End-to-end analysis for the IBM Telco Customer Churn dataset.

Run from the project root:
    python src/telecom_churn.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv

Optional:
    python src/telecom_churn.py --data data/your_file.csv --retention-cost 20 --save-rate 0.20

The retention-impact calculation is an illustrative scenario, not a causal estimate.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end telecom churn analysis")
    parser.add_argument("--data", default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--retention-cost", type=float, default=20.0,
                        help="Assumed cost to contact/retain one at-risk customer")
    parser.add_argument("--save-rate", type=float, default=0.20,
                        help="Assumed fraction of contacted at-risk customers retained")
    return parser.parse_args()


def ensure_dirs(root: Path):
    (root / "outputs" / "charts").mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download the IBM Telco Customer Churn CSV and place it in the data folder."
        )
    df = pd.read_csv(path)
    # Standardize column names while preserving expected names.
    df.columns = df.columns.str.strip()
    return df


def audit_data(df: pd.DataFrame, out: Path) -> dict:
    audit = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_by_column": df.isna().sum().astype(int).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    if "customerID" in df:
        audit["duplicate_customer_ids"] = int(df["customerID"].duplicated().sum())
        audit["unique_customer_ids"] = int(df["customerID"].nunique(dropna=True))
    if "Churn" in df:
        audit["churn_counts"] = df["Churn"].astype(str).value_counts(dropna=False).to_dict()
        audit["churn_percent"] = (
            df["Churn"].astype(str).value_counts(normalize=True, dropna=False) * 100
        ).round(2).to_dict()

    # Capture distinct categorical values for a basic category audit.
    categorical = df.select_dtypes(include=["object", "category"]).columns
    audit["categorical_values"] = {
        col: sorted(df[col].dropna().astype(str).str.strip().unique().tolist())[:100]
        for col in categorical
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2, default=str))
    return audit


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = df.columns.str.strip()

    # Trim whitespace in text fields and normalize blank strings to missing.
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA, " ": pd.NA})

    # Customer ID is an identifier, not a model feature.
    if "customerID" in df:
        df = df.drop_duplicates(subset=["customerID"], keep="first")
    else:
        df = df.drop_duplicates()

    # Mixed numeric billing values occur in the IBM file as strings.
    if "TotalCharges" in df:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Convert known binary fields to consistent Yes/No strings where applicable.
    for col in ["Churn", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        if col in df:
            df[col] = df[col].astype("string").str.strip()

    # Churn is the target; rows without a target cannot be used for supervised analysis.
    if "Churn" in df:
        df = df[df["Churn"].isin(["Yes", "No"])].copy()

    # TotalCharges is missing for some new customers; preserve rows and impute with
    # the logical value 0 only when tenure is zero, otherwise median.
    if "TotalCharges" in df:
        zero_tenure = df.get("tenure", pd.Series(index=df.index, dtype=float)).eq(0)
        df.loc[df["TotalCharges"].isna() & zero_tenure, "TotalCharges"] = 0
        if df["TotalCharges"].isna().any():
            df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if "tenure" in data:
        data["TenureGroup"] = pd.cut(
            pd.to_numeric(data["tenure"], errors="coerce"),
            bins=[-1, 12, 24, 48, 72],
            labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"]
        )

    if "MonthlyCharges" in data:
        data["MonthlyCharges"] = pd.to_numeric(data["MonthlyCharges"], errors="coerce")
        data["MonthlyChargeGroup"] = pd.cut(
            data["MonthlyCharges"], bins=3, labels=["Low", "Medium", "High"],
            duplicates="drop"
        )

    if {"TotalCharges", "tenure"}.issubset(data.columns):
        tenure = pd.to_numeric(data["tenure"], errors="coerce").replace(0, np.nan)
        data["AvgChargePerTenureMonth"] = data["TotalCharges"] / tenure
        data["AvgChargePerTenureMonth"] = data["AvgChargePerTenureMonth"].replace(
            [np.inf, -np.inf], np.nan
        )

    # A transparent count of optional services. "No internet service" and "No"
    # are both treated as not subscribed.
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    available = [c for c in service_cols if c in data.columns]
    if available:
        data["OptionalServiceCount"] = sum(
            data[c].astype("string").eq("Yes").astype(int) for c in available
        )

    return data


def bootstrap_mean_ci(values, n_boot=2000, confidence=0.95, seed=RANDOM_STATE):
    """Percentile bootstrap confidence interval for a sample mean."""
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return {"n": 0, "mean": None, "lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(x, size=len(x), replace=True)
        means[i] = np.mean(sample)
    alpha = 1 - confidence
    lower, upper = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "lower": float(lower),
        "upper": float(upper)
    }


def proportion_ci(successes, total, confidence=0.95):
    """Wilson score interval for a proportion."""
    if total == 0:
        return (None, None)
    from math import sqrt
    # 95% normal critical value; generalized confidence uses a normal approximation.
    z = 1.959963984540054 if confidence == 0.95 else 1.959963984540054
    p = successes / total
    denom = 1 + z*z/total
    center = (p + z*z/(2*total)) / denom
    margin = z * sqrt((p*(1-p) + z*z/(4*total))/total) / denom
    return max(0.0, center-margin), min(1.0, center+margin)


def churn_rate_table(df, group_col):
    if group_col not in df or "Churn" not in df:
        return pd.DataFrame()
    tmp = df.groupby(group_col, observed=False)["Churn"].agg(
        customers="size",
        churned=lambda s: (s == "Yes").sum(),
        churn_rate=lambda s: (s == "Yes").mean()
    ).reset_index()
    tmp["churn_rate_pct"] = (tmp["churn_rate"] * 100).round(2)
    return tmp.sort_values("churn_rate", ascending=False)


def statistical_analysis(df, out: Path):
    results = {}
    if "Churn" in df:
        churn = df[df["Churn"] == "Yes"]
        retained = df[df["Churn"] == "No"]
        for col in ["tenure", "MonthlyCharges", "TotalCharges", "OptionalServiceCount"]:
            if col in df:
                results[col] = {
                    "churned": bootstrap_mean_ci(churn[col]),
                    "retained": bootstrap_mean_ci(retained[col]),
                    "mean_difference_churned_minus_retained": (
                        float(pd.to_numeric(churn[col], errors="coerce").mean()
                              - pd.to_numeric(retained[col], errors="coerce").mean())
                    )
                }

        n = len(df)
        yes = int((df["Churn"] == "Yes").sum())
        low, high = proportion_ci(yes, n)
        results["overall_churn_rate"] = {
            "churned_customers": yes, "total_customers": n,
            "rate": yes / n if n else None,
            "wilson_95_ci": [low, high]
        }

    (out / "statistical_comparison.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    return results


def segment_analysis(df, out: Path):
    segment_cols = [
        "Contract", "PaymentMethod", "InternetService", "TechSupport",
        "OnlineSecurity", "PaperlessBilling", "TenureGroup",
        "MonthlyChargeGroup", "SeniorCitizen", "Partner", "Dependents"
    ]
    summaries = []
    for col in segment_cols:
        table = churn_rate_table(df, col)
        if not table.empty:
            table.insert(0, "segment", col)
            table = table.rename(columns={col: "segment_value"})
            summaries.append(table)
            safe = "".join(ch if ch.isalnum() else "_" for ch in col)
            table.to_csv(out / f"segment_{safe}.csv", index=False)

    if summaries:
        all_segments = pd.concat(summaries, ignore_index=True)
        all_segments.to_csv(out / "all_segment_churn_rates.csv", index=False)
    else:
        all_segments = pd.DataFrame()
    return all_segments


def make_charts(df, out: Path):
    chart_dir = out / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme()

    if "Churn" in df:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x="Churn")
        plt.title("Customer Churn Count")
        plt.tight_layout()
        plt.savefig(chart_dir / "churn_count.png", dpi=150)
        plt.close()

    for col in ["Contract", "PaymentMethod", "InternetService", "TechSupport"]:
        if col in df and "Churn" in df:
            rates = churn_rate_table(df, col)
            plt.figure(figsize=(9, 5))
            sns.barplot(data=rates, x=col, y="churn_rate_pct")
            plt.ylabel("Churn rate (%)")
            plt.title(f"Churn Rate by {col}")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(chart_dir / f"churn_by_{col}.png", dpi=150)
            plt.close()

    if {"tenure", "Churn"}.issubset(df.columns):
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=df, x="Churn", y="tenure")
        plt.title("Tenure Distribution by Churn")
        plt.tight_layout()
        plt.savefig(chart_dir / "tenure_by_churn.png", dpi=150)
        plt.close()

    if {"MonthlyCharges", "Churn"}.issubset(df.columns):
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
        plt.title("Monthly Charges by Churn")
        plt.tight_layout()
        plt.savefig(chart_dir / "monthly_charges_by_churn.png", dpi=150)
        plt.close()


def leakage_audit(df):
    # customerID is an identifier and must not be used as a predictive feature.
    # Churn is the target. No post-outcome fields should be introduced.
    excluded = ["customerID", "Churn"]
    notes = [
        "customerID is an identifier; exclude it from model inputs.",
        "Churn is the target; never include it among predictors.",
        "Do not use fields collected after cancellation or retention intervention.",
        "TotalCharges is accumulated over tenure; it may encode exposure/time and "
        "should be interpreted carefully, but it is not automatically target leakage.",
        "This dataset is observational and cross-sectional; associations do not prove causation."
    ]
    return {"excluded_from_predictors": excluded, "checks": notes}


def build_risk_bands(df):
    """Rule-based, interpretable screening bands; rules are illustrative."""
    risk = pd.Series(0, index=df.index, dtype=int)
    reasons = pd.Series("", index=df.index, dtype="string")

    def add_rule(condition, points, label):
        nonlocal risk, reasons
        condition = condition.fillna(False)
        risk.loc[condition] += points
        reasons.loc[condition] = reasons.loc[condition].fillna("").astype(str).apply(
            lambda x: (x + "; " if x else "") + label
        )

    if "tenure" in df:
        add_rule(pd.to_numeric(df["tenure"], errors="coerce") <= 12, 2, "tenure <= 12 months")
    if "Contract" in df:
        add_rule(df["Contract"].eq("Month-to-month"), 2, "month-to-month contract")
    if "TechSupport" in df:
        add_rule(df["TechSupport"].eq("No"), 1, "no tech support")
    if "OnlineSecurity" in df:
        add_rule(df["OnlineSecurity"].eq("No"), 1, "no online security")
    if "PaymentMethod" in df:
        add_rule(df["PaymentMethod"].eq("Electronic check"), 1, "electronic check payment")

    bands = pd.cut(
        risk, bins=[-1, 1, 3, 10], labels=["Low", "Medium", "High"]
    )
    result = pd.DataFrame({
        "risk_score": risk,
        "risk_band": bands.astype("string"),
        "risk_reasons": reasons.replace("", "No rule-based risk flags")
    }, index=df.index)
    return result


def run_baseline_model(df, out: Path):
    if "Churn" not in df:
        return {"status": "Skipped: target column Churn not found."}

    y = df["Churn"].map({"No": 0, "Yes": 1})
    # Remove target, identifier and accumulated TotalCharges from the primary
    # baseline to reduce leakage/exposure concerns. Compare alternatives separately.
    excluded = [c for c in ["Churn", "customerID", "TotalCharges"] if c in df]
    X = df.drop(columns=excluded).copy()

    # Exclude columns with all missing values.
    X = X.dropna(axis=1, how="all")
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    if X.empty or y.nunique() < 2:
        return {"status": "Skipped: insufficient predictors or target classes."}

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical)
        ],
        remainder="drop"
    )

    model = Pipeline([
        ("prep", preprocessor),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced"))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )
    metrics = {
        "model": "Logistic Regression (baseline)",
        "predictors": X.columns.tolist(),
        "test_rows": int(len(y_test)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "threshold": 0.5,
        "caution": "Baseline association model, not proof of causation or a production-ready system."
    }
    (out / "baseline_model_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def write_report(df, audit, stats, segments, model_metrics, risk_bands, args, out: Path):
    churn_n = int((df["Churn"] == "Yes").sum()) if "Churn" in df else 0
    total_n = len(df)
    churn_rate = churn_n / total_n if total_n else 0
    report = [
        "# Telecom Churn & Customer Behaviour — Analysis Report",
        "",
        "## 1. Executive summary",
        f"- Customers analysed: **{total_n:,}**",
        f"- Churned customers: **{churn_n:,}**",
        f"- Overall churn rate: **{churn_rate:.1%}**",
        "- This report describes associations in historical data; it does not establish causation.",
        "",
        "## 2. Data quality",
        f"- Duplicate rows in source: {audit.get('duplicate_rows', 'N/A')}",
        f"- Duplicate customer IDs in source: {audit.get('duplicate_customer_ids', 'N/A')}",
        "- `TotalCharges` was converted to numeric; blank/invalid values were handled.",
        "- Customer IDs were retained in the analytical export but excluded from model predictors.",
        "",
        "## 3. Statistical comparisons",
    ]

    for col, values in stats.items():
        if col == "overall_churn_rate":
            ci = values.get("wilson_95_ci", [None, None])
            if ci[0] is not None:
                report.append(
                    f"- Overall churn proportion 95% Wilson interval: "
                    f"{ci[0]:.1%} to {ci[1]:.1%}."
                )
        else:
            c = values["churned"]
            r = values["retained"]
            if c["mean"] is not None and r["mean"] is not None:
                report.append(
                    f"- **{col}** mean — churned: {c['mean']:.2f} "
                    f"(bootstrap 95% CI {c['lower']:.2f}–{c['upper']:.2f}); "
                    f"retained: {r['mean']:.2f} "
                    f"(bootstrap 95% CI {r['lower']:.2f}–{r['upper']:.2f})."
                )

    report += ["", "## 4. Churn segmentation"]
    if not segments.empty:
        for group in ["Contract", "PaymentMethod", "InternetService", "TechSupport"]:
            sub = segments[segments["segment"] == group]
            if not sub.empty:
                best = sub.sort_values("churn_rate", ascending=False).iloc[0]
                report.append(
                    f"- Highest observed churn segment for **{group}**: "
                    f"{best['segment_value']} ({best['churn_rate_pct']:.1f}%, "
                    f"n={int(best['customers'])})."
                )
    report += [
        "",
        "See `all_segment_churn_rates.csv` and the charts folder for full comparisons.",
        "",
        "## 5. Risk bands",
        risk_bands["risk_band"].value_counts(dropna=False).to_frame("customers").to_markdown(),
        "",
        "Risk bands use transparent, hand-built screening rules. They are not calibrated probabilities.",
        "",
        "## 6. Baseline classification",
    ]
    if "roc_auc" in model_metrics:
        report.append(f"- Logistic regression ROC-AUC: **{model_metrics['roc_auc']:.3f}**.")
        report.append(f"- Average precision: **{model_metrics['average_precision']:.3f}**.")
        report.append("- See `baseline_model_metrics.json` for additional metrics.")
    else:
        report.append(f"- {model_metrics.get('status', 'Not run')}")
    report += [
        "",
        "## 7. Leakage and interpretation",
        "- `customerID` and `Churn` were excluded from model predictors.",
        "- `TotalCharges` was excluded from the baseline model because it is strongly related to tenure/exposure.",
        "- Avoid using variables recorded after the churn event.",
        "- Segment differences are associations. A randomized retention experiment is needed to estimate causal impact.",
        "",
        "## 8. Illustrative retention-impact scenario",
        f"- Assumed high-risk customers contacted: {int((risk_bands['risk_band'] == 'High').sum()):,}",
        f"- Assumed save rate among contacted customers: {args.save_rate:.0%}",
        f"- Assumed retention cost per contact: {args.retention_cost:.2f} currency units",
        f"- Estimated customers retained under assumptions: "
        f"{int((risk_bands['risk_band'] == 'High').sum() * args.save_rate):,}",
        f"- Estimated contact cost: "
        f"{(risk_bands['risk_band'] == 'High').sum() * args.retention_cost:.2f} currency units",
        "- This is a scenario calculation, not a forecast. Replace assumptions with company data.",
        "",
        "## 9. Recommended next actions",
        "1. Prioritize customer interviews and service reviews for high-risk segments.",
        "2. Test contract-flexibility, onboarding and support improvements with controlled experiments.",
        "3. Track retention, complaint resolution and offer acceptance over time.",
        "4. Validate any predictive model on later time periods before operational use.",
        "",
        "## 10. Limitations",
        "- Dataset is historical and may not represent the current telecom market.",
        "- Missing-value handling and rule-based risk thresholds are analytical choices.",
        "- Model performance depends on the train/test split and available features.",
        "- No financial lifetime-value or actual campaign-cost data was supplied."
    ]
    (out / "final_report.md").write_text("\n".join(report), encoding="utf-8")


def main():
    args = parse_args()
    root = Path.cwd()
    out = root / "outputs"
    ensure_dirs(root)

    raw = load_data(Path(args.data))
    audit = audit_data(raw, out)
    cleaned = clean_data(raw)
    featured = add_features(cleaned)

    # Save the clean analytical table.
    featured.to_csv(out / "clean_analytical_table.csv", index=False)

    stats = statistical_analysis(featured, out)
    segments = segment_analysis(featured, out)
    make_charts(featured, out)

    leakage = leakage_audit(featured)
    (out / "leakage_audit.json").write_text(json.dumps(leakage, indent=2))

    risk = build_risk_bands(featured)
    risk_output = pd.concat(
        [featured[["customerID"]].reset_index(drop=True) if "customerID" in featured else pd.DataFrame(index=featured.index),
         risk.reset_index(drop=True)],
        axis=1
    )
    risk_output.to_csv(out / "customer_risk_bands.csv", index=False)

    model_metrics = run_baseline_model(featured, out)
    write_report(featured, audit, stats, segments, model_metrics, risk, args, out)

    print("\nAnalysis completed.")
    print(f"Customers analysed: {len(featured):,}")
    if "Churn" in featured:
        print(f"Churn rate: {(featured['Churn'].eq('Yes').mean()):.1%}")
    print(f"Outputs saved in: {out.resolve()}")


if __name__ == "__main__":
    main()
