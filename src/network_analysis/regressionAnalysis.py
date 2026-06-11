import pandas as pd
from pathlib import Path
from src.network_analysis.scores import aggregate_lsoas_to_wards
from duckdb import connect
import statsmodels as smf

BASE_DIR = Path(__file__).resolve().parent.parent.parent
# load scores dataframe generated from "scores" file
scores_df = pd.read_parquet(BASE_DIR / "data" / "ward_brokerage_scores.parquet")


conn = connect(BASE_DIR / "data" / "crimes.db")
sql = """
SELECT lsoa_code, crime_type, month_num, year
FROM crimes
WHERE year BETWEEN 2017 AND 2026
"""

df = conn.execute(sql).df()

df['period'] = df['year'].astype(str) + '-' + df['month_num'].astype(str).str.zfill(2)
df = df.drop(['year', 'month_num'], axis=1)

# print(df.head())
new_df = aggregate_lsoas_to_wards(df)


# print(new_df.columns.tolist())
# print(new_df.head())

# add total crime per month column and  total crime for specific crime type column
total_crime = new_df.groupby(['WD21CD', 'period'], as_index=False)['count'].sum()
total_crime = total_crime.rename(columns={'count': 'total_crime_in_month'})
df = pd.merge(new_df, total_crime, on=['WD21CD', 'period'], how='left')

# print(df.head())

#Create the "Future Violence" column that stores count per month

violence = df[df['crime_type'] == 'Violence and sexual offences']
violence_per_month = violence.groupby(['WD21CD', 'period'], as_index=False)['count'].sum()
violence_per_month = violence_per_month.rename(columns={'count': 'violence_count'})

#### merge dataframes again
final_df = pd.merge(df, violence_per_month, on=['WD21CD', 'period'], how='left')
print(final_df.head())

# fill in zeros instead of NaN so regression works
df['violence_per_month'] = df['violence_per_month'].fillna(0)

# finally run the regression

equation = "violence_per_month ~ brokerageScore + total_crime_in_month"

model = smf.glm(formula=equation,
                data=df,
                family=smf.families.NegativeBinomial(alpha=1.0)).fit()

print(model.summary())

