from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cleaned_dataset.csv')

print(df[['Productivity Change']].describe())

# Calculating average productivity increase by business size
bins = [0, 5000, 10000, 15000, float('inf')]
labels = ['Small (≤5k)', 'Medium (5k–10k)', 'Large (10k–15k)', 'Enterprise (15k–20k+)']

df['Business Size'] = pd.cut(df['Number of Employees Impacted'], bins=bins, labels=labels)


avg_productivity_by_size = df.groupby('Business Size', observed=False)['Productivity Change'].mean().reset_index()
avg_productivity_by_size['Productivity Change'] = avg_productivity_by_size['Productivity Change'].round(2)

print(avg_productivity_by_size)

sns.barplot(x='Business Size', y='Productivity Change', data=avg_productivity_by_size)
plt.title('Average Productivity Change by Business Size')
plt.ylabel('Avg Productivity Change (%)')
plt.xlabel('Business Size')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Plotting the productivity average increase
min_val = df['Productivity Change'].min()
max_val = df['Productivity Change'].max()
mean_val = df['Productivity Change'].mean()
plt.figure(figsize=(12, 4))
plt.hlines(1, min_val, max_val, color='gray')
plt.plot(min_val, 1, 'bo', label=f'Min: {min_val:.2f}')
plt.plot(mean_val, 1, 'go', label=f'Avg: {mean_val:.2f}')
plt.plot(max_val, 1, 'ro', label=f'Max: {max_val:.2f}')
plt.legend()
plt.yticks([])
plt.title('Productivity Change: Min, Average, and Max')
plt.xlabel('Productivity Change (%)')
plt.tight_layout()
plt.show()