import pandas as pd

df = pd.read_csv('datasets/DataCoSupplyChainDataset.csv', encoding='latin-1')

columns_needed = [
    "order date (DateOrders)",
    "Order Item Quantity",
    "Product Name",
    "Category Name",
    "Sales",
    "Order Item Total",
    "Order Region",
    "Shipping Mode"
]

df = df[columns_needed]

df.rename(columns={'Order Item Quantity': 'quantity'}, inplace=True)
df.rename(columns={'Order Item Total': 'total'}, inplace=True)
df.rename(columns={'Order Region': 'region'}, inplace=True)
df.rename(columns={'Shipping Mode': 'mode'}, inplace=True)
df.rename(columns={'order date (DateOrders)': 'order_date'}, inplace=True)

df['order_date'] = pd.to_datetime(df['order_date'])

df.dropna(inplace=True)

print("Cleaned shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nDate range:", df["order_date"].min(), "to", df["order_date"].max())
print("\nSample:\n", df.head())