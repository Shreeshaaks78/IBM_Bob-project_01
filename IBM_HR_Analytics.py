"""
=============================================================================
IBM_HR_Analytics.py
=============================================================================
Standalone, self-contained analysis script for the IBM HR Attrition dataset.

Dataset : IBM_HR_Attrition_500.csv  (500 rows × 11 columns)
Run     : python IBM_HR_Analytics.py

Output
------
  • Formatted terminal summaries (data quality, KPIs, grouped tables, insights)
  • Matplotlib / Seaborn charts saved to  ./charts/  and displayed interactively
  • Plotly interactive charts opened in the default browser

Dependencies (all in requirements.txt):
  pandas, numpy, matplotlib, seaborn, plotly
=============================================================================
"""

# =============================================================================
# SECTION 1 – IMPORTS
# =============================================================================

import os
import sys
import warnings
import textwrap

import numpy  as np
import pandas as pd
import matplotlib
import matplotlib.pyplot   as plt
import matplotlib.ticker   as mticker
import seaborn             as sns
import plotly.express      as px
import plotly.graph_objects as go
from   plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Use non-interactive Agg backend so charts can also be saved without a display
matplotlib.use("Agg")

# --------------------------------------------------------------------------- #
# Global style settings
# --------------------------------------------------------------------------- #
PALETTE   = {"Yes": "#EF4444", "No": "#22C55E"}   # attrition colour map
NAVY      = "#1E3A5F"
BLUE      = "#3B82F6"
AMBER     = "#F59E0B"
CHARTS_DIR = "charts"                              # folder for saved PNGs

os.makedirs(CHARTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
plt.rcParams.update({
    "figure.facecolor": "#F8FAFC",
    "axes.facecolor"  : "#F8FAFC",
    "axes.titlesize"  : 13,
    "axes.titleweight": "bold",
    "axes.titlecolor" : NAVY,
    "axes.labelcolor" : "#374151",
    "xtick.color"     : "#374151",
    "ytick.color"     : "#374151",
})

DATASET_PATH = "IBM_HR_Attrition_500.csv"


# =============================================================================
# SECTION 2 – DATA LOADING & CLEANING
# =============================================================================

def load_and_clean(path: str = DATASET_PATH) -> tuple[pd.DataFrame, dict]:
    """
    Load the IBM HR Attrition CSV, normalise column names, remove duplicates,
    impute missing values, and derive helper columns.

    Returns
    -------
    df   : cleaned DataFrame
    meta : dict with quality metrics
    """
    # ------------------------------------------------------------------
    # 2.1  Load raw CSV
    # ------------------------------------------------------------------
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Dataset not found: {path}")

    df = pd.read_csv(path)

    # ------------------------------------------------------------------
    # 2.2  Normalise column names
    #      The CSV header may contain spaces (e.g. "Job Role"); unify all.
    # ------------------------------------------------------------------
    df.columns = df.columns.str.strip()
    rename_map = {
        "Job Role"          : "JobRole",
        "Job Satisfaction"  : "JobSatisfaction",
        "Monthly Income"    : "MonthlyIncome",
        "Over Time"         : "OverTime",
        "Distance From Home": "DistanceFromHome",
        "Total Working Years": "TotalWorkingYears",
        "Work Life Balance" : "WorkLifeBalance",
        "Years At Company"  : "YearsAtCompany",
    }
    df = df.rename(columns=rename_map)

    # ------------------------------------------------------------------
    # 2.3  Remove exact duplicate rows
    # ------------------------------------------------------------------
    rows_before      = len(df)
    df               = df.drop_duplicates()
    duplicates_removed = rows_before - len(df)

    # ------------------------------------------------------------------
    # 2.4  Handle missing values
    #      Numeric  → fill with column median  (robust to outliers)
    #      Categorical → fill with column mode
    # ------------------------------------------------------------------
    numeric_cols     = ["Age", "DistanceFromHome", "JobSatisfaction",
                        "MonthlyIncome", "TotalWorkingYears",
                        "WorkLifeBalance", "YearsAtCompany"]
    categorical_cols = ["Attrition", "Department", "JobRole", "OverTime"]

    missing_before = df.isnull().sum().sum()

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("nan", np.nan)
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])

    missing_filled = missing_before - df.isnull().sum().sum()

    # ------------------------------------------------------------------
    # 2.5  Enforce correct dtypes
    # ------------------------------------------------------------------
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # ------------------------------------------------------------------
    # 2.6  Derive helper columns
    # ------------------------------------------------------------------
    # Binary attrition flag (for numeric aggregation)
    df["AttritionBinary"] = (df["Attrition"] == "Yes").astype(int)

    # Age buckets
    df["AgeBucket"] = pd.cut(
        df["Age"],
        bins=[17, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "55+"],
    )

    # Salary tier
    df["SalaryTier"] = pd.cut(
        df["MonthlyIncome"],
        bins=[0, 3000, 7000, 12000, 999999],
        labels=["Low (<3k)", "Mid (3-7k)", "High (7-12k)", "Senior (>12k)"],
    )

    meta = {
        "total_rows"        : len(df),
        "total_cols"        : len(df.columns),
        "duplicates_removed": duplicates_removed,
        "missing_filled"    : int(missing_filled),
    }
    return df, meta


# =============================================================================
# SECTION 3 – EDA FUNCTIONS
# =============================================================================

def print_section(title: str):
    """Print a styled section divider to the terminal."""
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_data_quality(df: pd.DataFrame, meta: dict):
    """Print data quality / cleaning summary."""
    print_section("DATA QUALITY REPORT")
    rows = [
        ("Total records after cleaning", meta["total_rows"]),
        ("Total columns (orig + derived)", meta["total_cols"]),
        ("Duplicate rows removed"        , meta["duplicates_removed"]),
        ("Missing values filled"         , meta["missing_filled"]),
        ("Remaining nulls"               , int(df.isnull().sum().sum())),
    ]
    for label, value in rows:
        print(f"  {label:<38}: {value}")


def compute_summary_stats(df: pd.DataFrame) -> dict:
    """
    Compute top-level KPI statistics from the cleaned DataFrame.

    Returns a dict of scalar metrics.
    """
    total          = len(df)
    attrited       = int(df["AttritionBinary"].sum())
    retained       = total - attrited
    attrition_rate = round(attrited / total * 100, 2)

    return {
        "total_employees"    : total,
        "attrition_count"    : attrited,
        "retention_count"    : retained,
        "attrition_rate_pct" : attrition_rate,
        "avg_age"            : round(df["Age"].mean(), 1),
        "median_age"         : int(df["Age"].median()),
        "avg_monthly_income" : round(df["MonthlyIncome"].mean(), 2),
        "median_income"      : int(df["MonthlyIncome"].median()),
        "avg_tenure"         : round(df["YearsAtCompany"].mean(), 1),
        "avg_total_exp"      : round(df["TotalWorkingYears"].mean(), 1),
        "avg_job_sat"        : round(df["JobSatisfaction"].mean(), 2),
        "avg_wlb"            : round(df["WorkLifeBalance"].mean(), 2),
        "avg_distance"       : round(df["DistanceFromHome"].mean(), 1),
        "pct_overtime"       : round((df["OverTime"] == "Yes").mean() * 100, 1),
    }


def print_summary_stats(stats: dict):
    """Pretty-print the KPI summary to the terminal."""
    print_section("KEY METRICS SUMMARY  (500 Employees)")
    groups = [
        ("ATTRITION", [
            ("Total Employees",        stats["total_employees"]),
            ("Left (Attrition = Yes)", stats["attrition_count"]),
            ("Stayed (Attrition = No)",stats["retention_count"]),
            ("Attrition Rate",        f"{stats['attrition_rate_pct']}%"),
        ]),
        ("DEMOGRAPHICS", [
            ("Average Age",            f"{stats['avg_age']} yrs"),
            ("Median Age",             f"{stats['median_age']} yrs"),
            ("Avg Monthly Income",    f"${stats['avg_monthly_income']:,.2f}"),
            ("Median Monthly Income", f"${stats['median_income']:,}"),
            ("Avg Company Tenure",    f"{stats['avg_tenure']} yrs"),
            ("Avg Total Experience",  f"{stats['avg_total_exp']} yrs"),
        ]),
        ("SATISFACTION & LIFESTYLE", [
            ("Avg Job Satisfaction",   f"{stats['avg_job_sat']} / 4"),
            ("Avg Work-Life Balance",  f"{stats['avg_wlb']} / 4"),
            ("Avg Distance from Home", f"{stats['avg_distance']} km"),
            ("Employees on OverTime",  f"{stats['pct_overtime']}%"),
        ]),
    ]
    for group_name, items in groups:
        print(f"\n  -- {group_name} --")
        for label, value in items:
            print(f"     {label:<32}: {value}")


def print_distribution(df: pd.DataFrame):
    """Print count / percentage distributions for categorical columns."""
    print_section("CATEGORICAL DISTRIBUTIONS")
    for col in ["Department", "JobRole", "OverTime", "Attrition"]:
        counts = df[col].value_counts()
        pcts   = (counts / len(df) * 100).round(1)
        print(f"\n  {col}:")
        for val in counts.index:
            bar = "#" * int(pcts[val] / 2)
            print(f"    {str(val):<30} {counts[val]:>4}  ({pcts[val]:>5.1f}%)  {bar}")


# =============================================================================
# SECTION 4 – GROUPED ANALYSIS FUNCTIONS
# =============================================================================

def group_by_department(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary DataFrame grouped by Department.
    Columns: Total, Attrited, AttritionRate%, AvgAge, AvgIncome,
             AvgTenure, AvgSatisfaction.
    """
    g = df.groupby("Department").agg(
        Total          =("AttritionBinary", "count"),
        Attrited       =("AttritionBinary", "sum"),
        AvgAge         =("Age",             "mean"),
        AvgIncome      =("MonthlyIncome",   "mean"),
        AvgTenure      =("YearsAtCompany",  "mean"),
        AvgSatisfaction=("JobSatisfaction", "mean"),
    ).reset_index()
    g["AttritionRate%"] = (g["Attrited"] / g["Total"] * 100).round(1)
    for col in ["AvgAge", "AvgIncome", "AvgTenure", "AvgSatisfaction"]:
        g[col] = g[col].round(1)
    return g.sort_values("AttritionRate%", ascending=False)


def group_by_jobrole(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary DataFrame grouped by JobRole.
    Columns: Total, Attrited, AttritionRate%, AvgIncome,
             AvgSatisfaction, AvgWLB.
    """
    g = df.groupby("JobRole").agg(
        Total          =("AttritionBinary", "count"),
        Attrited       =("AttritionBinary", "sum"),
        AvgIncome      =("MonthlyIncome",   "mean"),
        AvgSatisfaction=("JobSatisfaction", "mean"),
        AvgWLB         =("WorkLifeBalance", "mean"),
    ).reset_index()
    g["AttritionRate%"] = (g["Attrited"] / g["Total"] * 100).round(1)
    for col in ["AvgIncome", "AvgSatisfaction", "AvgWLB"]:
        g[col] = g[col].round(1)
    return g.sort_values("AttritionRate%", ascending=False)


def group_by_overtime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary DataFrame grouped by OverTime status.
    Columns: Total, Attrited, AttritionRate%, AvgIncome, AvgSatisfaction.
    """
    g = df.groupby("OverTime").agg(
        Total          =("AttritionBinary", "count"),
        Attrited       =("AttritionBinary", "sum"),
        AvgIncome      =("MonthlyIncome",   "mean"),
        AvgSatisfaction=("JobSatisfaction", "mean"),
    ).reset_index()
    g["AttritionRate%"] = (g["Attrited"] / g["Total"] * 100).round(1)
    for col in ["AvgIncome", "AvgSatisfaction"]:
        g[col] = g[col].round(1)
    return g


def print_grouped_tables(df: pd.DataFrame):
    """Print all three grouped analysis tables to the terminal."""
    print_section("GROUPED ANALYSIS TABLES")

    def pretty_table(title: str, frame: pd.DataFrame):
        print(f"\n  {title}")
        print("  " + "-" * 68)
        # Header
        cols = list(frame.columns)
        header = "  ".join(f"{c:<22}" if i == 0 else f"{c:>10}" for i, c in enumerate(cols))
        print("  " + header)
        print("  " + "-" * 68)
        for _, row in frame.iterrows():
            line_parts = []
            for i, (col, val) in enumerate(zip(cols, row)):
                if i == 0:
                    line_parts.append(f"{str(val):<22}")
                elif isinstance(val, float):
                    line_parts.append(f"{val:>10.1f}")
                else:
                    line_parts.append(f"{str(val):>10}")
            print("  " + "  ".join(line_parts))

    pretty_table("Department Summary", group_by_department(df))
    pretty_table("Job Role Summary",   group_by_jobrole(df))
    pretty_table("OverTime Summary",   group_by_overtime(df))


# =============================================================================
# SECTION 5 – VISUALISATION FUNCTIONS
# =============================================================================

# ---------------------------------------------------------------------------
# 5.1  Matplotlib / Seaborn charts  (saved as PNG + shown interactively)
# ---------------------------------------------------------------------------

def _save_show(fig: plt.Figure, filename: str):
    """Save figure to charts/ directory and display it."""
    path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  [chart saved] {path}")
    plt.show()
    plt.close(fig)


def chart_attrition_pie(df: pd.DataFrame):
    """Pie chart – overall attrition split (Matplotlib)."""
    counts = df["Attrition"].value_counts()
    colors = [PALETTE.get(k, "#CBD5E1") for k in counts.index]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    ax.set_title("Overall Attrition Split", pad=20)
    fig.patch.set_facecolor("#F8FAFC")
    _save_show(fig, "01_attrition_pie.png")


def chart_dept_attrition_bar(df: pd.DataFrame):
    """Grouped bar chart – attrition count by Department (Seaborn)."""
    grp = (
        df.groupby(["Department", "Attrition"])
        .size()
        .reset_index(name="Count")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=grp, x="Department", y="Count",
        hue="Attrition", palette=PALETTE, ax=ax,
        edgecolor="white", linewidth=0.8,
    )
    ax.set_title("Attrition Count by Department")
    ax.set_xlabel("Department")
    ax.set_ylabel("Number of Employees")
    ax.legend(title="Attrition")
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1, str(int(h)),
                ha="center", va="bottom", fontsize=9,
            )
    _save_show(fig, "02_dept_attrition_bar.png")


def chart_jobrole_attrition_rate(df: pd.DataFrame):
    """Horizontal bar chart – attrition rate % by Job Role (Matplotlib)."""
    g = group_by_jobrole(df).sort_values("AttritionRate%")
    colors = [
        "#EF4444" if r >= 20 else "#F59E0B" if r >= 10 else "#22C55E"
        for r in g["AttritionRate%"]
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(g["JobRole"], g["AttritionRate%"], color=colors, edgecolor="white")
    for bar, val in zip(bars, g["AttritionRate%"]):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val}%", va="center", fontsize=9,
        )
    ax.set_xlabel("Attrition Rate (%)")
    ax.set_title("Attrition Rate (%) by Job Role")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    _save_show(fig, "03_jobrole_attrition_rate.png")


def chart_overtime_attrition(df: pd.DataFrame):
    """Grouped bar + annotation – attrition by OverTime status (Seaborn)."""
    grp = (
        df.groupby(["OverTime", "Attrition"])
        .size()
        .reset_index(name="Count")
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        data=grp, x="OverTime", y="Count",
        hue="Attrition", palette=PALETTE, ax=ax,
        edgecolor="white",
    )
    ax.set_title("Attrition by OverTime Status")
    ax.set_xlabel("Works OverTime?")
    ax.set_ylabel("Number of Employees")
    ax.legend(title="Attrition")
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.5, str(int(h)),
                ha="center", va="bottom", fontsize=10,
            )
    _save_show(fig, "04_overtime_attrition_bar.png")


def chart_age_bucket_attrition(df: pd.DataFrame):
    """Stacked bar – attrition by age bucket (Matplotlib)."""
    grp = (
        df.groupby(["AgeBucket", "Attrition"], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    grp.plot(
        kind="bar", stacked=True, ax=ax,
        color=[PALETTE.get(c, "#CBD5E1") for c in grp.columns],
        edgecolor="white", width=0.65,
    )
    ax.set_title("Attrition Distribution by Age Group")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Number of Employees")
    ax.legend(title="Attrition")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    _save_show(fig, "05_age_bucket_attrition.png")


def chart_salary_tier_attrition(df: pd.DataFrame):
    """Stacked bar – attrition by salary tier (Matplotlib)."""
    order = ["Low (<3k)", "Mid (3-7k)", "High (7-12k)", "Senior (>12k)"]
    grp = (
        df.groupby(["SalaryTier", "Attrition"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(order)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    grp.plot(
        kind="bar", stacked=True, ax=ax,
        color=[PALETTE.get(c, "#CBD5E1") for c in grp.columns],
        edgecolor="white", width=0.65,
    )
    ax.set_title("Attrition Distribution by Salary Tier")
    ax.set_xlabel("Salary Tier")
    ax.set_ylabel("Number of Employees")
    ax.legend(title="Attrition")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    _save_show(fig, "06_salary_tier_attrition.png")


def chart_satisfaction_heatmap(df: pd.DataFrame):
    """
    Seaborn heatmap – mean attrition rate (%) by
    JobSatisfaction (rows) × WorkLifeBalance (cols).
    """
    pivot = df.pivot_table(
        values="AttritionBinary",
        index="JobSatisfaction",
        columns="WorkLifeBalance",
        aggfunc="mean",
    ) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot.round(1),
        annot=True, fmt=".1f", cmap="RdYlGn_r",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Attrition Rate (%)"},
        ax=ax,
    )
    ax.set_title("Attrition Rate (%) – Job Satisfaction vs Work-Life Balance")
    ax.set_xlabel("Work-Life Balance  (1=Bad … 4=Best)")
    ax.set_ylabel("Job Satisfaction   (1=Low … 4=High)")
    _save_show(fig, "07_satisfaction_wlb_heatmap.png")


def chart_income_distribution(df: pd.DataFrame):
    """
    Overlapping KDE histogram – monthly income distribution
    split by attrition status (Seaborn).
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, group in df.groupby("Attrition"):
        sns.kdeplot(
            group["MonthlyIncome"], ax=ax,
            label=f"Attrition={label}",
            color=PALETTE[label], fill=True, alpha=0.35, linewidth=2,
        )
    ax.set_title("Monthly Income Distribution by Attrition")
    ax.set_xlabel("Monthly Income ($)")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(title="Attrition")
    _save_show(fig, "08_income_distribution_kde.png")


def chart_tenure_scatter(df: pd.DataFrame):
    """
    Scatter – years at company vs monthly income,
    coloured by attrition (Matplotlib).
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, group in df.groupby("Attrition"):
        ax.scatter(
            group["YearsAtCompany"], group["MonthlyIncome"],
            c=PALETTE[label], label=f"Attrition={label}",
            alpha=0.55, s=40, edgecolors="white", linewidth=0.3,
        )
    ax.set_title("Tenure vs Monthly Income (by Attrition)")
    ax.set_xlabel("Years at Company")
    ax.set_ylabel("Monthly Income ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y:,.0f}"))
    ax.legend(title="Attrition")
    _save_show(fig, "09_tenure_vs_income_scatter.png")


def chart_correlation_heatmap(df: pd.DataFrame):
    """
    Seaborn correlation heatmap for all numeric columns
    (useful for detecting multicollinearity and linear attrition drivers).
    """
    numeric_df = df.select_dtypes(include=[np.number]).drop(
        columns=["AttritionBinary"], errors="ignore"
    )
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle mask
    sns.heatmap(
        corr, mask=mask,
        annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap – Numeric Features")
    _save_show(fig, "10_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# 5.2  Plotly interactive charts  (open in browser)
# ---------------------------------------------------------------------------

def plotly_jobrole_attrition(df: pd.DataFrame):
    """Plotly interactive horizontal bar – attrition rate by Job Role."""
    g = group_by_jobrole(df).sort_values("AttritionRate%", ascending=True)
    fig = px.bar(
        g, x="AttritionRate%", y="JobRole", orientation="h",
        color="AttritionRate%", color_continuous_scale="Reds",
        text=g["AttritionRate%"].astype(str) + "%",
        title="[Interactive] Attrition Rate (%) by Job Role",
        labels={"AttritionRate%": "Attrition Rate (%)"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, paper_bgcolor="#F8FAFC")
    fig.show()


def plotly_satisfaction_heatmap(df: pd.DataFrame):
    """Plotly interactive heatmap – attrition rate by Satisfaction × WLB."""
    pivot = df.pivot_table(
        values="AttritionBinary",
        index="JobSatisfaction",
        columns="WorkLifeBalance",
        aggfunc="mean",
    ) * 100

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values.round(1),
        x=[f"WLB {c}" for c in pivot.columns],
        y=[f"Sat {r}" for r in pivot.index],
        colorscale="RdYlGn_r",
        text=pivot.values.round(1),
        texttemplate="%{text}%",
        colorbar=dict(title="Attrition %"),
    ))
    fig.update_layout(
        title="[Interactive] Attrition Rate % – Job Satisfaction vs Work-Life Balance",
        xaxis_title="Work-Life Balance (1=Bad … 4=Best)",
        yaxis_title="Job Satisfaction (1=Low … 4=High)",
        paper_bgcolor="#F8FAFC",
    )
    fig.show()


def plotly_income_box(df: pd.DataFrame):
    """Plotly interactive box plot – income by Department and Attrition."""
    fig = px.box(
        df, x="Department", y="MonthlyIncome",
        color="Attrition", color_discrete_map=PALETTE,
        points="all", notched=True,
        title="[Interactive] Monthly Income by Department (coloured by Attrition)",
        labels={"MonthlyIncome": "Monthly Income ($)"},
    )
    fig.update_layout(paper_bgcolor="#F8FAFC")
    fig.show()


def plotly_scatter_tenure_income(df: pd.DataFrame):
    """Plotly interactive scatter – tenure vs income, hover shows role."""
    fig = px.scatter(
        df, x="YearsAtCompany", y="MonthlyIncome",
        color="Attrition", color_discrete_map=PALETTE,
        size="TotalWorkingYears",
        hover_data=["JobRole", "Department", "Age", "OverTime"],
        title="[Interactive] Tenure vs Monthly Income (bubble = Total Working Years)",
        opacity=0.75,
    )
    fig.update_layout(paper_bgcolor="#F8FAFC")
    fig.show()


# =============================================================================
# SECTION 6 – BUSINESS INSIGHTS
# =============================================================================

def compute_insights(df: pd.DataFrame, stats: dict) -> list[dict]:
    """
    Derive 7 data-driven insights from the cleaned DataFrame.
    Each insight is a dict with keys: title, finding, recommendation.
    """
    insights = []

    # ── 1. OverTime impact ─────────────────────────────────────────────
    ot_yes  = df[df["OverTime"] == "Yes"]["AttritionBinary"].mean() * 100
    ot_no   = df[df["OverTime"] == "No"]["AttritionBinary"].mean()  * 100
    insights.append({
        "title": "OverTime is the strongest single attrition driver",
        "finding": (
            f"Employees on OverTime leave at {ot_yes:.1f}% vs {ot_no:.1f}% "
            f"for non-OverTime staff — a {ot_yes - ot_no:.1f} percentage-point gap."
        ),
        "recommendation": (
            "Enforce overtime caps; introduce compensatory leave; "
            "track burnout scores monthly."
        ),
    })

    # ── 2. Low-salary vulnerability ────────────────────────────────────
    low_sal = df[df["SalaryTier"] == "Low (<3k)"]["AttritionBinary"].mean() * 100
    insights.append({
        "title": "Low-income employees leave at the highest rate",
        "finding": (
            f"Employees earning <$3k/month have a {low_sal:.1f}% attrition rate — "
            f"the highest of any salary band."
        ),
        "recommendation": (
            "Conduct annual salary benchmarking and introduce merit-based raises "
            "for the lowest-compensated roles."
        ),
    })

    # ── 3. Young-worker churn ──────────────────────────────────────────
    y18 = df[df["AgeBucket"] == "18-25"]["AttritionBinary"].mean() * 100
    insights.append({
        "title": "Early-career employees (18-25) churn most",
        "finding": f"The 18-25 age group has a {y18:.1f}% attrition rate.",
        "recommendation": (
            "Create structured onboarding, mentorship programmes, and clear "
            "promotion timelines for early-career hires."
        ),
    })

    # ── 4. Job satisfaction signal ─────────────────────────────────────
    low_sat = df[df["JobSatisfaction"] == 1]["AttritionBinary"].mean() * 100
    insights.append({
        "title": "Score-1 job satisfaction is a strong exit signal",
        "finding": (
            f"Employees with the lowest job satisfaction score (1) leave at "
            f"{low_sat:.1f}%."
        ),
        "recommendation": (
            "Run quarterly engagement surveys; mandate a 30-day action plan "
            "whenever a team averages below 2.5/4."
        ),
    })

    # ── 5. Commute distance ────────────────────────────────────────────
    far = df[df["DistanceFromHome"] > 20]["AttritionBinary"].mean() * 100
    insights.append({
        "title": "Long commutes increase attrition risk",
        "finding": (
            f"Employees living >20 km from the office leave at {far:.1f}%."
        ),
        "recommendation": (
            "Offer hybrid/remote work options or commute allowances "
            "for employees beyond 15 km."
        ),
    })

    # ── 6. Work-life balance ───────────────────────────────────────────
    bad_wlb = df[df["WorkLifeBalance"] == 1]["AttritionBinary"].mean() * 100
    insights.append({
        "title": "Poor work-life balance (score 1) drives exits",
        "finding": (
            f"Employees rating WLB as 1 (bad) leave at {bad_wlb:.1f}%."
        ),
        "recommendation": (
            "Introduce flexible hours, enforce annual leave quotas, "
            "and add well-being programmes."
        ),
    })

    # ── 7. Highest-attrition job role ──────────────────────────────────
    role_rates = df.groupby("JobRole")["AttritionBinary"].mean() * 100
    top_role   = role_rates.idxmax()
    top_rate   = role_rates.max()
    insights.append({
        "title": f"Role-specific risk: '{top_role}' needs urgent attention",
        "finding": (
            f"'{top_role}' has the highest role-level attrition at {top_rate:.1f}%."
        ),
        "recommendation": (
            f"Design targeted incentives, career-pathing, and recognition "
            f"programmes specific to the {top_role} track."
        ),
    })

    return insights


def print_insights(insights: list[dict]):
    """Pretty-print data-driven insights and HR recommendations."""
    print_section("DATA-DRIVEN BUSINESS INSIGHTS & HR RECOMMENDATIONS")
    for i, ins in enumerate(insights, 1):
        title = ins["title"]
        finding = textwrap.fill(ins["finding"],       width=64, subsequent_indent="           ")
        rec     = textwrap.fill(ins["recommendation"], width=64, subsequent_indent="           ")
        print(f"\n  [{i}] {title}")
        print(f"      Finding : {finding}")
        print(f"      Action  : {rec}")

    print()
    print("  OVERALL HR STRATEGY SUMMARY")
    print("  " + "-" * 68)
    strategies = [
        "1. Manage overtime — cap hours, introduce comp-time, monitor burnout.",
        "2. Benchmark pay — raise low-band salaries; add performance bonuses.",
        "3. Early-career programme — mentorship + clear promotion timelines.",
        "4. Close the feedback loop — act on survey data within 30 days.",
        "5. Flexible working — hybrid/remote for long-commute employees.",
        "6. Role-specific plans — target highest-attrition roles with tailored programmes.",
        "7. Lead indicators — track WLB, satisfaction, and OT hours as early warnings.",
    ]
    for s in strategies:
        print(f"    {s}")


# =============================================================================
# SECTION 7 – MAIN EXECUTION BLOCK
# =============================================================================

def main():
    print()
    print("=" * 72)
    print("   IBM HR Analytics – Employee Attrition Analysis")
    print("   Dataset: IBM_HR_Attrition_500.csv")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Step 1: Load & clean data
    # ------------------------------------------------------------------
    print("\n[1/6]  Loading and cleaning dataset ...")
    df, meta = load_and_clean(DATASET_PATH)
    print_data_quality(df, meta)

    # ------------------------------------------------------------------
    # Step 2: EDA – summary statistics
    # ------------------------------------------------------------------
    print("\n[2/6]  Computing summary statistics ...")
    stats = compute_summary_stats(df)
    print_summary_stats(stats)

    # ------------------------------------------------------------------
    # Step 3: Distributions
    # ------------------------------------------------------------------
    print("\n[3/6]  Categorical distributions ...")
    print_distribution(df)

    # ------------------------------------------------------------------
    # Step 4: Grouped analysis tables
    # ------------------------------------------------------------------
    print("\n[4/6]  Grouped analysis (Department / JobRole / OverTime) ...")
    print_grouped_tables(df)

    # ------------------------------------------------------------------
    # Step 5: Visualisations
    # ------------------------------------------------------------------
    print("\n[5/6]  Generating charts ...")
    print(f"       Static charts (Matplotlib/Seaborn) saved to ./{CHARTS_DIR}/")
    print("       Plotly charts will open in your default browser.\n")

    # Matplotlib / Seaborn
    chart_attrition_pie(df)
    chart_dept_attrition_bar(df)
    chart_jobrole_attrition_rate(df)
    chart_overtime_attrition(df)
    chart_age_bucket_attrition(df)
    chart_salary_tier_attrition(df)
    chart_satisfaction_heatmap(df)
    chart_income_distribution(df)
    chart_tenure_scatter(df)
    chart_correlation_heatmap(df)

    # Plotly (interactive browser tabs)
    plotly_jobrole_attrition(df)
    plotly_satisfaction_heatmap(df)
    plotly_income_box(df)
    plotly_scatter_tenure_income(df)

    # ------------------------------------------------------------------
    # Step 6: Business insights & recommendations
    # ------------------------------------------------------------------
    print("\n[6/6]  Deriving business insights ...")
    insights = compute_insights(df, stats)
    print_insights(insights)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("   Analysis complete.")
    print(f"   {len(os.listdir(CHARTS_DIR))} chart(s) saved to ./{CHARTS_DIR}/")
    print("   Run  `streamlit run app.py`  for the interactive dashboard.")
    print("   Run  `python generate_report.py`  to regenerate the Word report.")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
