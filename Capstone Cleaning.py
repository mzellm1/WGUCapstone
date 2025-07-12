import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = '/Users/matthewzellmer/Desktop/WGU/Capstone/Capstone analysis/Enterprise_GenAI_Adoption_Impact 2.csv'
df = pd.read_csv(file_path, encoding='utf-8', delimiter=',')


# Getting an overview of dataset
print('Dataset Shape:', df.shape)
print('Dataset Columns:', df.columns.tolist())

print(df.head())

# Dropping unused columns
columns_to_drop = ['Employee Sentiment', 'New Roles Created', 'Adoption Year', 'Country']
df = df.drop(columns=columns_to_drop)
print("Remaining columns:", df.columns.tolist())

# Changing company names to a unique integer starting from 1, column named changed to reflect that
df = df.rename(columns={'Company Name': 'Company ID'})
df = df.rename(columns={'Productivity Change (%)': 'Productivity Change'})
df['Company ID'] = range(1, len(df) + 1)

# Confirmation of change
print(df[['Company ID']].head())

# Checking missing/null values in columns
print("Missing values per column:")
print(df.isnull().sum())

# Nothing showing as missing or null, running another check in rows for confirmation
missing_rows = df[df.isnull().any(axis=1)]
print(f"\nRows with missing values: {len(missing_rows)}")
print(missing_rows.head())
# No missing values found in remaining dataset

print(df.info())
print(df[['Productivity Change']].describe())
print(df[['Training Hours Provided']].describe())
print(df[['Number of Employees Impacted']].describe())
# No obvious incorrect data or outliers in the dataset