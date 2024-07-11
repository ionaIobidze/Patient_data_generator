import pandas as pd
from scipy.stats import ks_2samp

# Load real and synthetic data
real_data = pd.read_csv('data/test_preprocessed.csv')
synthetic_data = pd.read_csv('data/synthetic_data.csv')

# Evaluate the similarity
for column in real_data.columns:
    stat, p_value = ks_2samp(real_data[column], synthetic_data[column])
    print(f"Feature: {column}, KS Statistic: {stat}, p-value: {p_value}")
