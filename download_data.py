import pandas as pd
import os

RAW = "data/raw"

print("Downloading JHU COVID-19 confirmed cases...")
url1 = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
df1 = pd.read_csv(url1)
df1.to_csv(f"{RAW}/jhu_confirmed_global.csv", index=False)
print(f"  Saved → {RAW}/jhu_confirmed_global.csv  |  Shape: {df1.shape}")

print("Downloading JHU COVID-19 deaths...")
url2 = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
df2 = pd.read_csv(url2)
df2.to_csv(f"{RAW}/jhu_deaths_global.csv", index=False)
print(f"  Saved → {RAW}/jhu_deaths_global.csv  |  Shape: {df2.shape}")

print("Downloading Our World in Data COVID-19 dataset...")
url3 = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
df3 = pd.read_csv(url3)
df3.to_csv(f"{RAW}/owid_covid_data.csv", index=False)
print(f"  Saved → {RAW}/owid_covid_data.csv  |  Shape: {df3.shape}")

print("\nAll datasets downloaded successfully!")