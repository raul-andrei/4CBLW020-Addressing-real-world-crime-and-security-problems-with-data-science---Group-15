import pandas as pd
from pathlib import Path
from code.network_analysis.scores import aggregate_lsoas_to_wards

# load scores dataframe generated from "scores" file
scores_df = pd.read_parquet("ward_brokerage_scores.parquet")


# load the crime dataframe
script_dir = Path(__file__).parent.resolve()
file_path = script_dir.parent / "data" / "uk_crime_full_cleaned.parquet"
# read the file containing all crimes per lsoas and aggragte them by wards
df = pd.read_parquet(file_path)
df = df.rename(columns={'LSOA code': 'lsoa_code'})
df = df.rename(columns={'Month': 'period'})
df = df.rename(columns={'Crime type': 'crime_type'})
new_df = aggregate_lsoas_to_wards(df)


# Print all column names
print(new_df.columns.tolist())

# add total crime per month column and  total crime for specific crime type column
new_df = new_df({"Total crime"} : kn, {"Total violence" : kn})
new_df["Total crime"] =
new_df["Total violence"] =

def get_crime_count(crime_type, area, timeColumn: str) -> list:
    '''
    Given an area and a crime type, the function returns a list of crime counts for each month given.
    '''
    filteredData = df[(df['LSOA name'] == area) & (df['Crime type'] == crime_type)]
    all_months = sorted(df[timeColumn].unique())
    crimeCount = filteredData[timeColumn].value_counts()
    crimeCount = crimeCount.reindex(all_months, fill_value=0)

    return list(crimeCount)

df = df.sort_values(by=['LSOA name', 'Month'])
violenceCount = get_crime_count('Violence', area, 'Month')

#Create the "Future Violence" column
# We group by LSOA so we don't accidentally shift one neighborhood's data into another
df['violenceNextMonth'] = df.groupby('LSOA name')['violenceCount'].shift(-1)

df = df.dropna(subset=['violenceNextMonth'])

#### merge both dataframes
# finally run the regression

equation = "violenceNextMonth ~ brokerageScore + totalCrimeCount"

model = smf.glm(formula=equation,
                data=df,
                family=smf.families.NegativeBinomial(alpha=1.0)).fit()

print(model.summary())
