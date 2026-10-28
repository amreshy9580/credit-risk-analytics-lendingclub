# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:47:47 2026

@author: DELL
"""

import numpy as np 
import pandas as pd
file_path = r"C:\Users\DELL\.spyder-py3\archive\loan\loan.csv"
df = pd.read_csv(file_path, low_memory = False)
print("Dataset Successfully Loaded")
print("Number of rows and columns:" , df.shape) 
# Display the first 5 rows of the dataset
print(df.head())
# Display the last 5 rows 
print(df.tail())
# Display all 74 columns name as python list 
print(list(df.columns))
# Display the technical Summary of the DataFrame
df.info(verbose = True, show_counts = True)
# Select key numerical Columns and view their summary
key_columns = ['loan_amnt', 'int_rate', 'annual_inc']
print(df[key_columns].describe())
# Separate numerical and categorical column names 
num_cols = df.select_dtypes(include = ['number']).columns
cat_cols = df.select_dtypes(include = ['object']).columns
# Print the counts of each column type
print("Total Columns Count :", df.shape[1])
print("numerical Columns Count:", len(num_cols))
print("Categorical Coumns Count :", len(cat_cols))
# 1. Define our essential credit-risk columns
risk_cols = [
    'loan_amnt', 'term', 'int_rate', 'grade', 
    'annual_inc', 'dti', 'delinq_2yrs', 'loan_status'
]

# 2. Create a smaller, focused DataFrame
df_risk = df[risk_cols]

# 3. Verify dimensions and inspect the top 3 rows
print("Focused DataFrame shape:", df_risk.shape)
print(df_risk.head(3))
# 1. Count missing values per column
missing_counts = df_risk.isnull().sum()

# 2. Calculate percentage of missing values per column
missing_percent = (missing_counts / len(df_risk)) * 100

# 3. Combine both into a small summary DataFrame
missing_summary = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing %': missing_percent
})

print(missing_summary)
# Check for duplicate rows across the dataset
duplicate_rows = df.duplicated().sum()
print("Total duplicate rows:", duplicate_rows)
# Inspect the data types of df_risk
print(df_risk.dtypes)

#  Step 12: Basic Filtering with Comparison Operators

# 1. Create a filter condition for large loans (> $30,000)
condition_large_loan = df_risk['loan_amnt'] > 30000

# 2. Apply the condition to filter df_risk
large_loans = df_risk[condition_large_loan]

# 3. Print the number of large loans found and view the top 3
print("Total loans over $30,000:", len(large_loans))
print(large_loans[['loan_amnt', 'grade', 'annual_inc', 'loan_status']].head(3))

# 1. Create a filter condition for high int loans (>20%)
condition_large_int = df_risk['int_rate'] > 20

# 2. Apply the condition to filter df_risk
high_interest_loans = df_risk[condition_large_int]

# 3. Print the number of high interest loans loans found and view the top 3
print("Total loans 20 % interest rate:", len(high_interest_loans))
print(high_interest_loans[['loan_amnt', 'grade', 'annual_inc', 'loan_status']].head(3))

# Filter for large loans (> $25,000) issued to Grade 'A' (safest) borrowers
condition_safe_and_large = (df_risk['grade'] == 'A') & (df_risk['loan_amnt'] > 25000)

safe_large_loans = df_risk[condition_safe_and_large]

print("Total Grade 'A' loans over $25,000:", len(safe_large_loans))
print(safe_large_loans[['loan_amnt', 'grade', 'int_rate', 'loan_status']].head(3))

# Filter for delinq 2 years > 0 and loan ammount > 20000
condition_unsafe_and_large = (df_risk['delinq_2yrs'] > 0 ) & (df_risk['loan_amnt'] > 20000)

risky_delinq_loans = df_risk[condition_unsafe_and_large]

print("Total delinq 2 years > 0 and > 20000 loans", len(risky_delinq_loans))
print(risky_delinq_loans[['loan_amnt', 'grade', 'int_rate', 'loan_status']].head(3))

# 1. Sort the loans by interest rate from highest to lowest
sorted_by_rate = df_risk.sort_values(by='int_rate', ascending=False)

# 2. Display the top 5 loans with the highest interest rates
print(sorted_by_rate[['int_rate', 'grade', 'loan_amnt', 'loan_status']].head(5))

# 1. Sort the loans by income from highest to lowest
sorted_by_income = df_risk.sort_values(by='annual_inc', ascending=False)

# 2. Display the top 5 loans with the highest income 
print(sorted_by_income[['annual_inc', 'loan_amnt', 'grade']].head(5))

# Step 15: Frequency Analysis with .value_counts()

# 1. Count the raw number of loans in each loan status
status_counts = df_risk['loan_status'].value_counts()
print("Loan Status Raw Counts:")
print(status_counts)

# 2. Calculate the percentage distribution of each status
status_percentages = df_risk['loan_status'].value_counts(normalize=True) * 100
print("\nLoan Status Percentages:")
print(status_percentages.round(2))

# 1. Count the raw number of loans in each grade
grade_counts = df_risk['grade'].value_counts()
print("Grades Raw Counts:")
print(grade_counts)

# 2. Calculate the percentage distribution of each status
grade_percentages = df_risk['grade'].value_counts(normalize=True) * 100
print("\ngrade Percentages:")
print(grade_percentages.round(2))

# 1. Calculate the average interest rate for each credit grade
avg_rate_by_grade = df_risk.groupby('grade')['int_rate'].mean().round(2)

# 2. Display the result
print("Average Interest Rate by Grade:")
print(avg_rate_by_grade)

avg_loan_by_grade = df_risk.groupby('grade')['loan_amnt'].mean().round(2)
print("Average loan amount by grade")
print(avg_loan_by_grade)

grade_stats = df_risk.groupby('grade')['loan_amnt'].agg(['count', 'mean', 'median']).round(2)
print("Loan Amount Summary Statstistics by Grade:")
print(grade_stats)

rate_range_by_grade = df_risk.groupby('grade')['int_rate'].agg(['min', 'mean', 'max']).round(2)
print("Interest Rate by grade ")
print(rate_range_by_grade)

print(df.shape)
print(df_risk.shape)
print(df_risk.columns)

# 1. Calculate metrics on loan_amnt using NumPy
loan_mean = np.mean(df_risk['loan_amnt'])
loan_median = np.median(df_risk['loan_amnt'])
loan_std = np.std(df_risk['loan_amnt'])

# 2. Print the results rounded to 2 decimal places
print("--- NumPy Loan Amount Statistics ---")
print("Mean loan amount:   $", round(loan_mean, 2))
print("Median loan amount: $", round(loan_median, 2))
print("Standard Deviation: $", round(loan_std, 2))

# NumPy calculations ignoring NaN values
income_mean = np.nanmean(df_risk['annual_inc'])
income_median = np.nanmedian(df_risk['annual_inc'])

print("Mean Annual Income:   $", round(income_mean, 2))
print("Median Annual Income: $", round(income_median, 2))

# Visualizing Loan Distribution with Matplotlib & Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")

# 1. Create a plot figure
plt.figure(figsize=(9, 4))

# 2. Plot histogram with KDE for loan amounts
sns.histplot(df_risk['loan_amnt'], bins=35, kde=True, color='teal')

# 3. Add titles and labels
plt.title('Distribution of Loan Amounts (LendingClub)', fontsize=13, pad=10)
plt.xlabel('Loan Amount ($)', fontsize=11)
plt.ylabel('Number of Loans', fontsize=11)

# 4. Render the plot
plt.tight_layout()
plt.show()

# Visualizing Interest Rate Spreads by Grade using Boxplots
#Step 20 Boxplot 
plt.figure(figsize=(10, 5))

grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

sns.boxplot(
    x='grade', 
    y='int_rate', 
    data=df_risk, 
    order=grade_order, 
    hue='grade',         # Explicitly binds the color palette to the grade
    legend=False,        # Suppresses the redundant side legend
    palette='Spectral'
)

plt.title('Interest Rate Distribution across Credit Grades', fontsize=13, pad=10)
plt.xlabel('LendingClub Grade (Low Risk → High Risk)', fontsize=11)
plt.ylabel('Interest Rate (%)', fontsize=11)

plt.tight_layout()
plt.show()


# 1. Filter for closed loans only to avoid censoring bias
closed_mask = df_risk['loan_status'].isin(['Fully Paid', 'Charged Off'])
df_closed = df_risk[closed_mask].copy()

# 2. Create binary default indicator (1 = Default/Charged Off, 0 = Fully Paid)
df_closed['is_default'] = (df_closed['loan_status'] == 'Charged Off').astype(int)

# 3. Calculate historical default rate (%) by grade
default_by_grade = (
    df_closed.groupby('grade')['is_default']
    .mean()
    .mul(100)
    .round(2)
    .reset_index(name='default_rate_pct')
)

print("--- Default Rate (%) by Grade ---")
print(default_by_grade)

# 4. Plot Default Rates by Grade (Warning-Free)
plt.figure(figsize=(9, 4.5))
grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

ax = sns.barplot(
    x='grade', 
    y='default_rate_pct', 
    data=default_by_grade, 
    order=grade_order, 
    hue='grade',         # Binds palette to grade
    legend=False,        # Suppresses extra legend
    palette='Reds'
)

# Annotate exact percentages above each bar
for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.2f}%", 
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='bottom', 
        fontsize=10, xytext=(0, 3), 
        textcoords='offset points'
    )

plt.title('Historical Default Rate by LendingClub Grade (Closed Loans)', fontsize=13, pad=12)
plt.xlabel('Credit Grade', fontsize=11)
plt.ylabel('Default Rate (%)', fontsize=11)
plt.ylim(0, default_by_grade['default_rate_pct'].max() + 5)

plt.tight_layout()
plt.show()

# Comparing Borrower Profiles & Visualizing Correlations

# 1. Compare key metrics between Non-Defaulters (0) and Defaulters (1)
profile_comparison = (
    df_closed.groupby('is_default')[['int_rate', 'dti', 'annual_inc', 'loan_amnt']]
    .mean()
    .round(2)
)
print("--- Average Borrower Profiles: Paid (0) vs Charged Off (1) ---")
print(profile_comparison)

# 2. Select numeric risk columns and calculate Pearson correlation
numeric_risk_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'is_default']
corr_matrix = df_closed[numeric_risk_cols].corr()

# 3. Plot the Correlation Heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    corr_matrix, 
    annot=True,            # Display the correlation coefficient inside each cell
    fmt='.2f',             # Format numbers to 2 decimal places
    cmap='vlag',           # Diverging colormap: blue for negative, red for positive
    vmin=-0.3, vmax=0.3,   # Color scale limits centered around 0
    linewidths=0.8,        # White border separating cells
    cbar_kws={'label': 'Pearson Correlation'}
)

plt.title('Correlation Matrix: Credit Risk Indicators & Default', fontsize=13, pad=12)
plt.tight_layout()
plt.show()


# Loan Term Impact (term) on Default Risk

# 1. Group by term to calculate loan volume, default rate, average amount, and average rate
term_analysis = df_closed.groupby('term').agg(
    total_loans=('is_default', 'count'),
    default_rate_pct=('is_default', lambda x: round(np.mean(x) * 100, 2)),
    avg_loan_amnt=('loan_amnt', 'mean'),
    avg_int_rate=('int_rate', 'mean')
).round(2)

print("--- Loan Term Risk Breakdown ---")
print(term_analysis)

# 2. Visualizing Default Rate by Term
plt.figure(figsize=(7, 4.5))

ax = sns.barplot(
    x='term', 
    y='is_default', 
    data=df_closed, 
    estimator=lambda x: np.mean(x) * 100,
    errorbar=None,
    hue='term',
    legend=False,
    palette='Blues_d'
)

# Annotate percentages on top of bars
for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.2f}%", 
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='bottom', 
        fontsize=11, xytext=(0, 3), 
        textcoords='offset points'
    )

plt.title('Default Rate (%) by Loan Term (Closed Loans)', fontsize=13, pad=12)
plt.xlabel('Loan Term', fontsize=11)
plt.ylabel('Default Rate (%)', fontsize=11)
plt.ylim(0, term_analysis['default_rate_pct'].max() + 5)

plt.tight_layout()
plt.show()


# Continuous Feature Binning with pd.cut() (DTI Risk Bands)

# 1. Define bin thresholds and readable tier labels
dti_bins = [0, 12, 20, 30, 100]
dti_labels = ['Low (<12%)', 'Moderate (12-20%)', 'High (20-30%)', 'Critical (>30%)']

# 2. Create the binned categorical feature
df_closed['dti_bucket'] = pd.cut(
    df_closed['dti'], 
    bins=dti_bins, 
    labels=dti_labels, 
    right=False
)

# 3. Calculate loan count and default rate by DTI tier
dti_summary = df_closed.groupby('dti_bucket', observed=False).agg(
    total_loans=('is_default', 'count'),
    default_rate_pct=('is_default', lambda x: round(np.mean(x) * 100, 2))
).reset_index()

print("--- Default Rate by Debt-to-Income (DTI) Band ---")
print(dti_summary)

# 4. Plot DTI Tiers vs. Default Rate
plt.figure(figsize=(8, 4.5))

ax = sns.barplot(
    x='dti_bucket', 
    y='default_rate_pct', 
    data=dti_summary, 
    hue='dti_bucket',
    legend=False,
    palette='Oranges'
)

# Annotate percentages above bars
for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.2f}%", 
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='bottom', 
        fontsize=10, xytext=(0, 3), 
        textcoords='offset points'
    )

plt.title('Default Rate by Debt-to-Income (DTI) Tier', fontsize=13, pad=12)
plt.xlabel('DTI Risk Band', fontsize=11)
plt.ylabel('Default Rate (%)', fontsize=11)
plt.ylim(0, dti_summary['default_rate_pct'].max() + 5)

plt.tight_layout()
plt.show()

# Multi-Dimensional Risk Matrix (pd.pivot_table & Heatmap) 

# 1. Create a 2D Pivot Table of default rates (%) by Grade and Term
risk_matrix = pd.pivot_table(
    data=df_closed,
    values='is_default',
    index='grade',
    columns='term',
    aggfunc=lambda x: np.mean(x) * 100
).round(2)

print("--- Default Rate (%) Matrix: Grade vs. Term ---")
print(risk_matrix)

# 2. Visualize the Risk Matrix with a Heatmap
plt.figure(figsize=(8, 5))

sns.heatmap(
    risk_matrix, 
    annot=True, 
    fmt='.2f', 
    cmap='YlOrRd',       # Yellow (low risk) to Red (high risk)
    cbar_kws={'label': 'Default Rate (%)'},
    linewidths=0.5
)

plt.title('Default Rate Risk Matrix: Grade vs. Term (Closed Loans)', fontsize=13, pad=12)
plt.xlabel('Loan Term', fontsize=11)
plt.ylabel('Credit Grade', fontsize=11)

plt.tight_layout()
plt.show()

# Credit Bureau History — Clean vs. Derogatory Files (delinq_2yrs)

# 1. Create a binary segment: Clean (0 past lates) vs. Derogatory (1+ past lates)
df_closed['credit_history'] = np.where(
    df_closed['delinq_2yrs'] == 0, 
    'Clean (0 lates)', 
    'Derogatory (1+ lates)'
)

# 2. Compare volume, default rate, and interest rate between the two groups
history_summary = df_closed.groupby('credit_history', observed=False).agg(
    total_loans=('is_default', 'count'),
    default_rate_pct=('is_default', lambda x: round(np.mean(x) * 100, 2)),
    avg_int_rate=('int_rate', 'mean'),
    avg_annual_inc=('annual_inc', 'mean')
).round(2).reset_index()

print("--- Past Delinquency vs. Default Performance ---")
print(history_summary)

# 3. Visualize the comparison
plt.figure(figsize=(7, 4.5))

ax = sns.barplot(
    x='credit_history', 
    y='default_rate_pct', 
    data=history_summary, 
    hue='credit_history',
    legend=False,
    palette=['#2b5c8f', '#d95f02']
)

for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.2f}%", 
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='bottom', 
        fontsize=11, xytext=(0, 3), 
        textcoords='offset points'
    )

plt.title('Default Rate: Clean Credit vs. Prior Delinquency (Past 2 Yrs)', fontsize=13, pad=12)
plt.xlabel('Credit Bureau History', fontsize=11)
plt.ylabel('Default Rate (%)', fontsize=11)
plt.ylim(0, history_summary['default_rate_pct'].max() + 5)

plt.tight_layout()
plt.show()

# Portfolio Financial Loss Summary (Executive Deliverable)

# 1. Create a helper column measuring actual dollar loss per loan
df_closed['dollar_loss'] = df_closed['is_default'] * df_closed['loan_amnt']

# 2. Portfolio Dollar Summary by Grade
portfolio_summary = df_closed.groupby('grade').agg(
    total_loans=('loan_amnt', 'count'),
    total_funded_m=('loan_amnt', lambda x: round(x.sum() / 1e6, 2)),
    total_loss_m=('dollar_loss', lambda x: round(x.sum() / 1e6, 2)),
    default_rate_pct=('is_default', lambda x: round(np.mean(x) * 100, 2))
).reset_index()

# 3. Calculate Loss Rate as % of Funded Capital
portfolio_summary['loss_rate_pct'] = (
    (portfolio_summary['total_loss_m'] / portfolio_summary['total_funded_m']) * 100
).round(2)

print("\n=================== EXECUTIVE PORTFOLIO SUMMARY ===================")
print(portfolio_summary.to_string(index=False))

# 4. Plot Total Dollars Lost by Grade (in Millions)
plt.figure(figsize=(9, 4.5))

ax = sns.barplot(
    x='grade', 
    y='total_loss_m', 
    data=portfolio_summary, 
    order=grade_order, 
    hue='grade',
    legend=False,
    palette='flare'
)

for p in ax.patches:
    ax.annotate(
        f"${p.get_height():.1f}M", 
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='bottom', 
        fontsize=10, xytext=(0, 3), 
        textcoords='offset points'
    )

plt.title('Total Capital Lost to Defaults ($ Millions) by Grade', fontsize=13, pad=12)
plt.xlabel('Credit Grade', fontsize=11)
plt.ylabel('Total Charged-Off Capital ($ Millions)', fontsize=11)
plt.ylim(0, portfolio_summary['total_loss_m'].max() * 1.15)

plt.tight_layout()
plt.show()

# Borrower Leverage — Loan-to-Income (LTI) Ratio

# 1. Feature Engineering: Calculate Loan-to-Income (LTI) ratio
df_closed['lti_ratio'] = (df_closed['loan_amnt'] / df_closed['annual_inc']).round(3)

# 2. Bin LTI into practical underwriting leverage tiers
lti_bins = [0, 0.15, 0.30, 0.50, np.inf]
lti_labels = ['Low Leverage (<15%)', 'Moderate (15-30%)', 'High (30-50%)', 'Extreme (>50%)']

df_closed['lti_tier'] = pd.cut(
    df_closed['lti_ratio'], 
    bins=lti_bins, 
    labels=lti_labels, 
    right=False
)

# 3. Summarize default rates across LTI tiers
lti_summary = df_closed.groupby('lti_tier', observed=False).agg(
    total_loans=('is_default', 'count'),
    default_rate_pct=('is_default', lambda x: round(np.mean(x) * 100, 2)),
    avg_loan_amnt=('loan_amnt', 'mean')
).reset_index()

print("--- Default Rate by Loan-to-Income (LTI) Leverage Tier ---")
print(lti_summary)

# 4. Visualize Leverage vs. Default Risk
plt.figure(figsize=(8, 4.5))

ax = sns.barplot(
    x='lti_tier', 
    y='default_rate_pct', 
    data=lti_summary, 
    hue='lti_tier',
    legend=False,
    palette='Purples'
)

for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.2f}%", 
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='bottom', 
        fontsize=10, xytext=(0, 3), 
        textcoords='offset points'
    )

plt.title('Default Rate by Loan-to-Income (LTI) Leverage Tier', fontsize=13, pad=12)
plt.xlabel('Borrower Leverage Tier', fontsize=11)
plt.ylabel('Default Rate (%)', fontsize=11)
plt.ylim(0, lti_summary['default_rate_pct'].max() + 5)

plt.tight_layout()
plt.show()

# Final Underwriting Takeaways & Data Export
# 1. Export the cleaned, enriched dataset for modeling or reporting
export_filename = "lending_club_risk_processed.csv"
df_closed.to_csv(export_filename, index=False)
print(f"\n[SUCCESS] Enriched dataset exported successfully to: {export_filename}")

# 2. Print Project Executive Summary Findings
print("\n" + "="*60)
print("     LENDINGCLUB CREDIT RISK ANALYSIS: CORE TAKEAWAYS")
print("="*60)
print("1. RISK-BASED PRICING IS EMPIRICALLY JUSTIFIED:")
print("   - Grade A defaults at ~6% (avg rate: ~7.2%).")
print("   - Grade G defaults at ~42% (avg rate: ~25.6%).")
print("   - The 7x default multiplier is directly offset by the interest rate spread.")
print("\n2. TERM RISK OFTEN OUTWEIGHS CREDIT GRADE:")
print("   - 60-month loans default at ~29%, compared to ~14% for 36-month loans.")
print("   - A Grade B borrower on 60 months carries higher default risk than a Grade C borrower on 36 months.")
print("\n3. CAPITAL EXPOSURE CONCENTRATION:")
print("   - Grade C generates the largest absolute dollar write-offs due to origination volume.")
print("   - High DTI (>30%) and extreme leverage (>50% LTI) consistently elevate default probability.")
print("="*60)
