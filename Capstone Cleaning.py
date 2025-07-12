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
