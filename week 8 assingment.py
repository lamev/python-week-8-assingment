# Project Overview
# This project will analyze and visualize global COVID-19 data to track cases, deaths, recoveries, and vaccination progress across countries and over time.

# Project Setup
# 1. Create Project Structure
mkdir covid19-tracker
cd covid19-tracker
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install Required Packages
pip install pandas numpy matplotlib seaborn plotly jupyter

# 3. Create Jupyter Notebook
jupyter notebook

# Data Collection
# Option 1: Use API (Recommended)
import pandas as pd
import requests

# Fetch latest data from COVID-19 API
url = "https://disease.sh/v3/covid-19/countries"
response = requests.get(url)
data = response.json()

# Convert to DataFrame
covid_df = pd.DataFrame(data)

# Option 2: Use CSV Dataset
# Download dataset from: https://github.com/CSSEGISandData/COVID-19
covid_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/MM-DD-YYYY.csv')

# Data Analysis Implementation
# 1. Basic Data Exploration
# Display first 5 rows
print(covid_df.head())

# Get summary statistics
print(covid_df.describe())

# Check for missing values
print(covid_df.isnull().sum())

# 2. Data Cleaning
# python
# Handle missing values
covid_df.fillna(0, inplace=True)

# Convert date columns to datetime
covid_df['updated'] = pd.to_datetime(covid_df['updated'], unit='ms')

# Select relevant columns
covid_df = covid_df[['country', 'cases', 'deaths', 'recovered', 'active', 'tests', 'population']]

# 3. Key Visualizations
# Cases by Country (Matplotlib)
import matplotlib.pyplot as plt

top_10 = covid_df.sort_values('cases', ascending=False).head(10)
plt.figure(figsize=(12,6))
plt.barh(top_10['country'], top_10['cases'])
plt.title('Top 10 Countries by COVID-19 Cases')
plt.xlabel('Total Cases')
plt.ylabel('Country')
plt.show()

# Cases Over Time (Plotly)
import plotly.express as px

# Assuming you have time series data
fig = px.line(time_series_df, x='date', y='cases', color='country',
              title='COVID-19 Cases Over Time')
fig.show()


# Mortality Rate Analysis (Seaborn)
import seaborn as sns

covid_df['mortality_rate'] = (covid_df['deaths'] / covid_df['cases']) * 100

plt.figure(figsize=(12,6))
sns.scatterplot(data=covid_df, x='cases', y='mortality_rate', hue='continent')
plt.title('Mortality Rate vs Total Cases')
plt.xscale('log')
plt.show()


# Advanced Features
# 1. Interactive Dashboard

from ipywidgets import interact

@interact(country=covid_df['country'].unique())
def show_country_stats(country):
    country_data = covid_df[covid_df['country'] == country]
    print(f"Cases: {country_data['cases'].values[0]}")
    print(f"Deaths: {country_data['deaths'].values[0]}")
    print(f"Recovery Rate: {(country_data['recovered']/country_data['cases']).values[0]*100:.2f}%")


    # 2. Vaccination Progress

    # Add vaccination data
vaccine_df = pd.read_csv('vaccination-data.csv')
merged_df = pd.merge(covid_df, vaccine_df, on='country')

# Plot vaccination vs cases
plt.figure(figsize=(10,6))
sns.regplot(data=merged_df, x='people_vaccinated_per_hundred', y='cases_per_million')
plt.title('Vaccination Rate vs Cases per Million')
plt.show()


# Project Structure
covid19-tracker/
├── data/                   # Raw data files
├── notebooks/              # Jupyter notebooks
│   └── COVID-19_Analysis.ipynb
├── scripts/                # Python scripts
│   └── data_processing.py
├── visualizations/         # Saved plots
├── README.md               # Project documentation
└── requirements.txt        # Dependencies

# ----Running the Project-------
# Activate virtual environment

# Launch Jupyter Notebook

# Run analysis cells sequentially

# Export visualizations as needed