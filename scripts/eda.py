#!/usr/bin/env python3
"""Write the exploratory report and its figures from the IBM sample."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn.config import FIGURES_DIR, REPORTS_DIR  # noqa: E402
from churn.data import load_raw  # noqa: E402
from churn.features import prepare_features  # noqa: E402
from churn.plotting import NAVY, SLATE, TERRACOTTA, TEAL, apply_style, save_figure, style_axes  # noqa: E402

apply_style()


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def rate_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(column, observed=True)
        .agg(customers=("churn", "size"), churn_rate=("churn", "mean"))
        .sort_values("churn_rate", ascending=False)
    )
    return grouped


def plot_rates(table: pd.DataFrame, title: str, xlabel: str, path: Path, rank: bool = True) -> None:
    ordered = table.sort_values("churn_rate", ascending=True) if rank else table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(ordered) + 1.6))
    colors = [TERRACOTTA if rate == ordered["churn_rate"].max() else NAVY for rate in ordered["churn_rate"]]
    ax.barh(ordered.index.astype(str), ordered["churn_rate"], color=colors, zorder=2)
    for label, rate, customers in zip(ordered.index.astype(str), ordered["churn_rate"], ordered["customers"]):
        ax.text(rate + 0.008, label, f"{pct(rate)}  (n={int(customers)})", va="center", color=SLATE, fontsize=9)
    style_axes(ax, grid="x")
    ax.set_xlim(0, min(1.0, ordered["churn_rate"].max() + 0.22))
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    save_figure(fig, path)


def write_report(raw: pd.DataFrame, figures: Path) -> str:
    labeled = raw.copy()
    labeled["churn"] = (labeled["Churn"] == "Yes").astype(int)
    features, _ = prepare_features(raw)
    labeled["TotalCharges_num"] = features["TotalCharges"].to_numpy()
    labeled["addon_count"] = features["addon_count"].to_numpy()

    n_rows = len(labeled)
    churn_rate = float(labeled["churn"].mean())
    n_churn = int(labeled["churn"].sum())
    blank_bills = int((raw["TotalCharges"].astype(str).str.strip() == "").sum())
    tenure_zero = int((labeled["tenure"] == 0).sum())

    contract = rate_table(labeled, "Contract")
    internet = rate_table(labeled, "InternetService")
    payment = rate_table(labeled, "PaymentMethod")
    paperless = rate_table(labeled, "PaperlessBilling")

    tenure_bins = pd.cut(
        labeled["tenure"],
        bins=[-0.1, 6, 12, 24, 48, 72],
        labels=["0–6 months", "7–12 months", "13–24 months", "25–48 months", "49–72 months"],
    )
    tenure = rate_table(labeled.assign(tenure_bin=tenure_bins), "tenure_bin")
    tenure = tenure.reindex(list(tenure_bins.cat.categories))

    internet_users = labeled[labeled["InternetService"] != "No"]
    security = rate_table(internet_users, "OnlineSecurity")
    support = rate_table(internet_users, "TechSupport")

    month_to_month = contract.loc["Month-to-month", "churn_rate"]
    two_year = contract.loc["Two year", "churn_rate"]
    fiber = internet.loc["Fiber optic", "churn_rate"]
    no_internet = internet.loc["No", "churn_rate"]
    early = tenure.loc["0–6 months", "churn_rate"]
    late = tenure.loc["49–72 months", "churn_rate"]
    echeck = payment.loc["Electronic check", "churn_rate"]
    card = payment.loc["Credit card (automatic)", "churn_rate"]

    charges = labeled.groupby("Churn")["MonthlyCharges"].median()
    tenure_med = labeled.groupby("Churn")["tenure"].median()

    # Plots
    balance = pd.DataFrame(
        {"customers": [n_rows - n_churn, n_churn]},
        index=["Stayed", "Churned"],
    )
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.bar(balance.index, balance["customers"], color=[NAVY, TERRACOTTA], zorder=2)
    for label, count in balance["customers"].items():
        ax.text(label, count + 60, f"{int(count):,}\n{pct(count / n_rows)}", ha="center", va="bottom", color=SLATE)
    style_axes(ax)
    ax.set_ylim(0, balance["customers"].max() * 1.22)
    ax.set_ylabel("Customers")
    ax.set_title("About one in four customers churned")
    save_figure(fig, figures / "01_class_balance.png")

    plot_rates(contract, "Churn rate by contract", "Share who churned", figures / "02_churn_by_contract.png")
    plot_rates(
        tenure,
        "Churn rate by tenure",
        "Share who churned",
        figures / "03_churn_by_tenure.png",
        rank=False,
    )
    plot_rates(internet, "Churn rate by internet service", "Share who churned", figures / "04_churn_by_internet.png")
    plot_rates(payment, "Churn rate by payment method", "Share who churned", figures / "05_churn_by_payment.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for label, color in (("No", NAVY), ("Yes", TERRACOTTA)):
        subset = labeled.loc[labeled["Churn"] == label, "MonthlyCharges"]
        ax.hist(subset, bins=30, density=True, alpha=0.55, color=color, label="Churned" if label == "Yes" else "Stayed")
    style_axes(ax)
    ax.set_xlabel("Monthly charges")
    ax.set_ylabel("Density")
    ax.set_title("Churned customers sit toward higher monthly charges")
    ax.legend(frameon=False)
    save_figure(fig, figures / "06_monthly_charges.png")

    addon = (
        labeled.groupby("addon_count")
        .agg(customers=("churn", "size"), churn_rate=("churn", "mean"))
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.bar(addon.index.astype(str), addon["churn_rate"], color=TEAL, zorder=2)
    style_axes(ax)
    ax.set_xlabel("Number of add-on services (security, backup, protection, support, streaming)")
    ax.set_ylabel("Churn rate")
    ax.set_title("More add-ons, lower observed churn")
    save_figure(fig, figures / "07_addon_count.png")

    lines = [
        "# Exploratory analysis: who leaves",
        "",
        "Question for this pass: before fitting a model, which account patterns show up more often among customers who churn?",
        "",
        "The plots below are computed from the IBM Telco Customer Churn sample (7,043 customers). Associations are descriptive. Contract, fiber, and payment method are chosen together, so a high churn rate on one slice is not by itself a reason to change that product.",
        "",
        "## Data check",
        "",
        f"- {n_rows:,} customers, 21 raw columns, one row per `customerID`.",
        f"- Target `Churn`: {n_churn:,} left ({pct(churn_rate)}) and {n_rows - n_churn:,} stayed ({pct(1 - churn_rate)}). A model that always predicts “stay” is already {pct(1 - churn_rate)} accurate, so accuracy alone is a weak headline.",
        f"- `TotalCharges` is text. {blank_bills} values are blank, and all {tenure_zero} of them are customers with `tenure` 0 who have not been billed. None of those new accounts are labeled as churned. The training pipeline sets those bills to 0 instead of filling them with the typical customer’s total.",
        "- There are no other missing cells. `SeniorCitizen` is already 0/1. Service fields use “No internet service” or “No phone service” rather than nulls.",
        "",
        "![Class balance](figures/01_class_balance.png)",
        "",
        "## Contract and tenure",
        "",
        f"Month-to-month customers churn at {pct(month_to_month)}. Two-year customers churn at {pct(two_year)}. The early-tenure bin (0–6 months) churns at {pct(early)}; customers who have already stayed 49–72 months churn at {pct(late)}.",
        "",
        "Median tenure is "
        f"{tenure_med['Yes']:.0f} months among churners and {tenure_med['No']:.0f} months among stayers. "
        "The people who leave are concentrated in short, flexible contracts. That can mean the contract causes them to leave, or that people who already expect to leave refuse a long contract. The model can use the pattern either way; a retention offer cannot treat it as a proven lever without an experiment.",
        "",
        "| Contract | Customers | Churn rate |",
        "| --- | ---: | ---: |",
    ]
    for label, row in contract.iterrows():
        lines.append(f"| {label} | {int(row['customers']):,} | {pct(row['churn_rate'])} |")
    lines += [
        "",
        "![Churn rate by contract](figures/02_churn_by_contract.png)",
        "",
        "| Tenure | Customers | Churn rate |",
        "| --- | ---: | ---: |",
    ]
    for label, row in tenure.iterrows():
        lines.append(f"| {label} | {int(row['customers']):,} | {pct(row['churn_rate'])} |")
    lines += [
        "",
        "![Churn rate by tenure](figures/03_churn_by_tenure.png)",
        "",
        "## Service and bill",
        "",
        f"Fiber-optic customers churn at {pct(fiber)}, against {pct(no_internet)} for customers with no internet service. Fiber also carries a higher monthly price, so this is tangled with the bill. Median monthly charges are ${charges['Yes']:.2f} for churners and ${charges['No']:.2f} for stayers.",
        "",
        "| Internet service | Customers | Churn rate |",
        "| --- | ---: | ---: |",
    ]
    for label, row in internet.iterrows():
        lines.append(f"| {label} | {int(row['customers']):,} | {pct(row['churn_rate'])} |")
    lines += [
        "",
        "![Churn rate by internet service](figures/04_churn_by_internet.png)",
        "",
        "![Monthly charges by churn](figures/06_monthly_charges.png)",
        "",
        "Among customers who have internet, missing protection lines up with leaving: "
        f"OnlineSecurity = No churns at {pct(security.loc['No', 'churn_rate'])} versus "
        f"{pct(security.loc['Yes', 'churn_rate'])} when the add-on is present. "
        f"TechSupport = No churns at {pct(support.loc['No', 'churn_rate'])} versus "
        f"{pct(support.loc['Yes', 'churn_rate'])} with support. "
        "Streaming shows a weaker gap. The engineered `addon_count` (how many of the six add-ons are “Yes”) falls as churn falls, which is the same pattern in one column. It is still confounded with tenure and contract: long-term customers collect more add-ons.",
        "",
        "![Churn rate by add-on count](figures/07_addon_count.png)",
        "",
        "## Payment",
        "",
        f"Electronic check is the noisy payment method: churn {pct(echeck)}, compared with {pct(card)} for automatic credit card. Paperless billing is {pct(paperless.loc['Yes', 'churn_rate'])} versus {pct(paperless.loc['No', 'churn_rate'])} for paper bills. Both are markers worth handing to the model, not proof that the payment rail itself causes churn.",
        "",
        "| Payment method | Customers | Churn rate |",
        "| --- | ---: | ---: |",
    ]
    for label, row in payment.iterrows():
        lines.append(f"| {label} | {int(row['customers']):,} | {pct(row['churn_rate'])} |")
    lines += [
        "",
        "![Churn rate by payment method](figures/05_churn_by_payment.png)",
        "",
        "## What goes into the model",
        "",
        "The classifier matrix keeps the raw service, contract, and demographic fields, plus two row-local features:",
        "",
        "- `addon_count`: number of add-on columns equal to Yes.",
        "- `avg_monthly_spend`: `TotalCharges / tenure` when tenure is positive, otherwise the listed `MonthlyCharges`.",
        "",
        "`customerID` is dropped. Medians, scales, and one-hot levels are not computed here; the training script fits them inside a scikit-learn pipeline after the split.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    report = write_report(raw, FIGURES_DIR)
    destination = REPORTS_DIR / "eda.md"
    destination.write_text(report)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
