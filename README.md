# 📊 IBM HR Analytics – Employee Attrition Analysis

> A full-stack data analytics project that identifies the key drivers of employee
> attrition using the IBM HR dataset, delivered as an interactive Streamlit dashboard,
> a standalone analysis script, and a fully exported Word project report.

---

## 📁 Project Structure

```
IBM BOB project/
├── IBM_HR_Attrition_500.csv           # Source dataset (500 rows × 11 columns)
├── app.py                             # Streamlit interactive dashboard (main UI)
├── data_loader.py                     # Data loading, cleaning & validation module
├── analytics.py                       # Aggregation, grouping & Plotly chart functions
├── IBM_HR_Analytics.py                # Standalone self-contained analysis script
├── generate_report.py                 # Word report generator (python-docx)
├── IBM_HR_Analytics_Project_Report.docx  # Auto-generated project report
├── charts/                            # PNG chart exports from IBM_HR_Analytics.py
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## 📂 Dataset Information

| Property | Detail |
|---|---|
| **File** | `IBM_HR_Attrition_500.csv` |
| **Rows** | 500 |
| **Columns** | 11 |
| **Target variable** | `Attrition` (Yes / No) |
| **Attrition rate** | ~15.6% (78 of 500 employees) |

### Column Reference

| Column | Type | Description |
|---|---|---|
| `Age` | Integer | Employee age in years |
| `Attrition` | Categorical | Whether employee left — `Yes` / `No` |
| `Department` | Categorical | Department name (HR / R&D / Sales) |
| `DistanceFromHome` | Integer | Commute distance in km |
| `JobRole` | Categorical | Job title (9 unique roles) |
| `JobSatisfaction` | Integer (1–4) | Self-reported satisfaction score |
| `MonthlyIncome` | Integer | Monthly salary in USD |
| `OverTime` | Categorical | Whether employee works overtime — `Yes` / `No` |
| `TotalWorkingYears` | Integer | Total years of work experience |
| `WorkLifeBalance` | Integer (1–4) | Self-reported work-life balance score |
| `YearsAtCompany` | Integer | Years employed at current company |

---

## 🛠 Tech Stack

| Layer | Technology | Version |
|---|---|---|
| **Language** | Python | 3.10+ |
| **Dashboard** | Streamlit | 1.64.0 |
| **Data Wrangling** | Pandas | 2.2.2 |
| **Numerical** | NumPy | 1.26.4 |
| **Interactive Charts** | Plotly (Express + Graph Objects) | 5.22.0 |
| **Static Charts** | Matplotlib + Seaborn | 3.9.0 / 0.13.2 |
| **Word Report** | python-docx | 1.1.2 |
| **Chart Export** | Kaleido | 0.2.1 |
| **Excel Support** | openpyxl | 3.1.3 |

---

## ⚙️ Setup & Run Instructions

### 1. Prerequisites

- Python **3.10 or higher**
- `pip` package manager

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Interactive Dashboard

```bash
streamlit run app.py
```

Opens automatically at **http://localhost:8501**.  
Use the sidebar filters (Department, Job Role, OverTime, Age, Income) to slice the
data in real time — all charts, KPIs, and insight cards update instantly.

### 4. Run the Standalone Analysis Script

```bash
python IBM_HR_Analytics.py
```

Runs a complete terminal analysis and saves **10 PNG charts** to `./charts/`.  
Also opens **4 interactive Plotly charts** in your default browser.

### 5. Regenerate the Word Report

```bash
python generate_report.py
```

Creates (or overwrites) `IBM_HR_Analytics_Project_Report.docx` with full analysis,
summary tables, insights, and HR strategy recommendations.

---

## 🧩 Module Overview

### `data_loader.py`

| Step | Action |
|---|---|
| Load | `pd.read_csv()` with automatic column-name normalisation |
| Clean | Removes duplicate rows |
| Impute | Numeric nulls → column median; categorical nulls → column mode |
| Type-fix | Enforces integer dtype on all numeric columns |
| Derive | `AttritionBinary` (0/1), `AgeBucket` (5 bands), `SalaryTier` (4 bands) |
| Safety | `summary_stats()` is guarded against empty DataFrames (no ZeroDivisionError) |

### `analytics.py`

- **14 Plotly figures** — pie, donut, grouped/stacked/horizontal bars, violin,
  scatter (bubble), box, KDE histogram, heatmap
- **3 summary tables** — Department, JobRole, OverTime (used in dashboard + report)
- **`derive_insights()`** — generates 7 data-driven findings with HR recommendations;
  all subset calculations are guarded against empty slices and NaN values
- Shared `_apply_layout()` helper enforces transparent backgrounds and dark-label
  axes (`#0F172A`) on every chart for high-contrast readability

### `app.py` — Streamlit Dashboard

- **Sidebar** — live filters: Department, Job Role, OverTime, Age range, Income range
- **7 KPI cards** — Total Employees, Attrited, Retained, Attrition Rate, Avg Income,
  Avg Tenure, Avg Job Satisfaction (all update on filter change)
- **10-tab layout**:

| Tab | Contents |
|---|---|
| 🔵 Overview | Overall attrition pie + department bar |
| 🏢 Department | Grouped bar, avg income bar, summary table |
| 💼 Job Role | Attrition rate bar, satisfaction box plot, summary table |
| ⏰ OverTime | Grouped bar, income violin plot, summary table |
| 👤 Age & Tenure | Age-bucket stacked bar, tenure vs income scatter |
| 😊 Satisfaction | Job Sat × WLB heatmap, WLB distribution pie |
| 💰 Salary | Salary-tier stacked bar, income KDE histogram |
| 🚗 Distance | Attrition rate by commute distance bucket |
| 📋 Data Tables | Searchable, filterable raw data + CSV download |
| 💡 Business Insights | 7 dynamic insight cards + 7-point HR strategy panel |

### `IBM_HR_Analytics.py` — Standalone Script

Self-contained, no Streamlit dependency. Outputs:
- Formatted terminal summaries (quality report, KPIs, distributions, grouped tables)
- 10 Matplotlib/Seaborn charts saved to `./charts/`
- 4 Plotly interactive charts (open in browser)
- 7 business insights with recommendations

### `generate_report.py` — Word Report Generator

Produces a styled 13-section `.docx` report:
- Cover page, Table of Contents, Executive Summary
- Dataset overview with column reference table
- Methodology (data cleaning steps)
- 8 analysis sections with embedded summary tables
- Business insights & HR strategy action plan
- Conclusion

---

## 📈 Key Insights (Dataset-derived)

| # | Insight |
|---|---|
| 1 | **OverTime is the #1 driver** — overtime employees leave at ~2× the rate of non-OT staff |
| 2 | **Low salary = high risk** — the `<$3k/month` tier has the highest attrition % |
| 3 | **Early-career churn** — employees aged 18–25 leave most frequently |
| 4 | **Satisfaction score 1 is a strong exit signal** — lowest-rated employees leave at a significantly elevated rate |
| 5 | **Commute distance matters** — employees >20 km from office show higher attrition |
| 6 | **WLB score 1 correlates with leaving** — poor work-life balance is a leading indicator |
| 7 | **Role-specific risk** — certain job roles have structurally high attrition needing targeted plans |

---

## 🏢 Business Recommendations

1. **Manage overtime** — cap hours, introduce comp-time, track burnout monthly.
2. **Benchmark pay** — raise lowest salary bands; add merit bonuses.
3. **Early-career programme** — structured onboarding, mentorship, clear promotions for 18–25s.
4. **Close the feedback loop** — act on survey data within 30 days of collection.
5. **Flexible working** — hybrid/remote options for employees commuting >15 km.
6. **Role-specific retention plans** — targeted incentives for highest-attrition roles.
7. **Lead indicators dashboard** — monitor WLB, satisfaction, and OT hours as early warnings.

---

## 🐛 Known Fixes Applied

| Issue | Fix |
|---|---|
| `StreamlitDuplicateElementId` | Unique `key=` added to all 16 `st.plotly_chart()` calls |
| `ZeroDivisionError` in `summary_stats()` | Guarded with `if total > 0` check |
| `ZeroDivisionError` / `NaN` in `derive_insights()` | New `_safe_rate()` helper guards all subset `.mean()` calls |
| Plotly dark background on white page | `paper_bgcolor` / `plot_bgcolor` set to `rgba(0,0,0,0)` |
| Low-contrast chart text | Shared `_apply_layout()` enforces `#0F172A` on all axes, ticks, labels |
| Sidebar text invisible on light bg | CSS selectors added for dropdown values, options, and slider ticks |

---

## 📄 License

This project is for educational and analytical purposes. The dataset is derived from
the IBM HR Analytics Employee Attrition & Performance sample dataset.
