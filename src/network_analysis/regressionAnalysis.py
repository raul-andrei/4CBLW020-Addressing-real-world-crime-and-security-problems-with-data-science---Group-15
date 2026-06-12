import pandas as pd
from pathlib import Path
from src.network_analysis.scores import aggregate_lsoas_to_wards
from duckdb import connect
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE_DIR = Path(__file__).resolve().parent.parent.parent
# load scores dataframe generated from "scores" file
scores_df = pd.read_parquet(BASE_DIR / "data" / "ward_brokerage_scores.parquet")
# so they match
years = (scores_df['period'] - 1) // 12
months = (scores_df['period'] - 1) % 12 + 1
scores_df['period'] = years.astype(str) + '-' + months.astype(str).str.zfill(2)

conn = connect(BASE_DIR / "data" / "crimes.db")
sql = """
SELECT lsoa_code, crime_type, month_num, year
FROM crimes
WHERE year BETWEEN 2017 AND 2026
"""

df = conn.execute(sql).df()
df['period'] = df['year'].astype(str) + '-' + df['month_num'].astype(str).str.zfill(2)
df = df.drop(['year', 'month_num'], axis=1)

# print("Scores DataFrame:")
# print(scores_df.head())

new_df = aggregate_lsoas_to_wards(df)

# print("\nMain DataFrame:")
# print(new_df[['WD21CD', 'period']].head())

# add total crime per month column
total_crime = new_df.groupby(['WD21CD', 'period'], as_index=False)['count'].sum()
total_crime = total_crime.rename(columns={'count': 'total_crime_in_month'})
df = pd.merge(new_df, total_crime, on=['WD21CD', 'period'], how='left')

# print(df.head())

#Create the "Future Violence" column that stores count per month

violence = df[df['crime_type'] == 'Violence and sexual offences']
violence_per_month = violence.groupby(['WD21CD', 'period'], as_index=False)['count'].sum()
violence_per_month = violence_per_month.rename(columns={'count': 'violence_count'})

#### merge dataframes again
df = pd.merge(df, violence_per_month, on=['WD21CD', 'period'], how='left')

# fill in zeros instead of NaN so regression works
df['violence_count'] = df['violence_count'].fillna(0)

scores_df['period'] = scores_df['period'].astype(str)

# final final final merge - crime data with average betweenness
final_df = pd.merge(df, scores_df, on=['WD21CD', 'period'], how='left')
final_df['avg_betweenness'] = final_df['avg_betweenness'].fillna(0)

print(final_df.head())
print(final_df.columns.tolist())

# finally run the regression
reg_df = final_df[['WD21CD', 'period', 'total_crime_in_month', 'violence_count', 'avg_betweenness']].drop_duplicates()


reg_df = reg_df.sort_values(by=['WD21CD', 'period'])

# Shift the violence count up by one to represent next month's violence
reg_df['violence_next_month'] = reg_df.groupby('WD21CD')['violence_count'].shift(-1)

# Drop the last month in the dataset since it has no "next month" to predict
reg_df = reg_df.dropna(subset=['violence_next_month'])

equation = "violence_next_month  ~ avg_betweenness + total_crime_in_month"

model = smf.glm(formula=equation,
                data=reg_df,
                family=sm.families.NegativeBinomial(alpha=1.0)).fit()

print(model.summary())

# we don't control for ward fixed effects, meaning part of the signal could reflect that high-crime wards just structurally have both high brokerage and high violence
# rather than brokerage genuinely predicting change over time — we flag this as a limitation and suggest demeaning as future work
print(reg_df['avg_betweenness'].describe())
