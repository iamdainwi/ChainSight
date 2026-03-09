import pandas as pd

df = pd.read_csv('datasets/DataCoSupplyChainDataset.csv', encoding='latin-1')

print("Shape: ", df.shape)
print("\n Columns: ", df.columns)
print("\n Frist 5 Rows: ", df.head())
print("\n Missing Values: ", df.isnull().sum())