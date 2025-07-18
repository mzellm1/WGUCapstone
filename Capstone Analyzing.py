from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cleaned_dataset.csv')

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
x_range = pd.DataFrame(np.linspace(X.min().iloc[0], X.max().iloc[0], 300), columns=['Avg Training Hours per Employee'])
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

# Visual for peak productivity training
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Avg Training Hours per Employee', y='Productivity Change', data=df, alpha=0.1)
plt.plot(x_range, y_pred, color='red', label='Polynomial Fit')
plt.axvline(x=peak, color='green', linestyle='--', label=f'Peak at ~{peak:.1f} hrs')
plt.title('Training Hours vs Productivity Change (Peak at ~41.7 hrs)')
plt.xlabel('Avg Training Hours per Employee')
plt.ylabel('Productivity Change')
plt.legend()
plt.tight_layout()
plt.show()

# Regression summary for R-squared and P-Value
X = df[['Avg Training Hours per Employee']]
y = df['Productivity Change']
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

X_poly_df = pd.DataFrame(X_poly, columns=['Intercept', 'X', 'X^2'])

# Add constant to avoid duplicate intercept
model = sm.OLS(y, X_poly_df).fit()

print(model.summary())