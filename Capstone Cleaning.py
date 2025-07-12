from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Finding and adding average training hours per employee to the dataset
df['Avg Training Hours per Employee'] = df['Training Hours Provided'] / df['Number of Employees Impacted']
print(df[['Training Hours Provided', 'Number of Employees Impacted', 'Avg Training Hours per Employee']].head())


# Scatterplot to begin analysis on training hours and productivity change
sns.scatterplot(x='Avg Training Hours per Employee', y='Productivity Change', data=df)
plt.title('Average Training Hours vs Productivity Change')
plt.xlabel('Average Training Hours Provided')
plt.ylabel('Productivity Change')
plt.show()
# Original scatterplot was too noisy visually, created an average training hours for clearer analysis

# Finding correlation between the average training hours and productivity
correlation = df['Avg Training Hours per Employee'].corr(df['Productivity Change'])
print(f"Correlation: {correlation:.2f}")

# Fitting a polynomial regression line

X = df[['Avg Training Hours per Employee']]
y = df['Productivity Change']

# Polynomial transformation
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Fit the model
model = LinearRegression()
model.fit(X_poly, y)

# Predict for plotting
x_range = np.linspace(X.min(), X.max(), 300)
x_range_poly = poly.transform(x_range)
y_pred = model.predict(x_range_poly)

# Plot with curve
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Avg Training Hours per Employee', y='Productivity Change', data=df, alpha=0.1)
plt.plot(x_range, y_pred, color='red', label='Polynomial Fit')
plt.title('Average Training Hours vs Productivity Change (with curve)')
plt.legend()
plt.show()

# Finding peak productivity for training hours
a = model.coef_[2]
b = model.coef_[1]
peak = -b / (2 * a)
print(f"Estimated peak productivity occurs at ~{peak:.2f} avg training hours per employee")

# Export the DataFrame to a CSV file to use in Tableau
#output_path = '/Users/matthewzellmer/Desktop/WGU/Capstone/Capstone analysis/cleaned_ai_productivity_data.csv'
#df.to_csv(output_path, index=False)
#print(f"✅ DataFrame exported to: {output_path}")
# Leaving the export code as a comment to not export every time I run this code