import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/DataCoSupplyChainDataset.csv", encoding="latin-1")

df.rename(columns={"order date (DateOrders)":"order_date",
                   "Order Item Quantity":"quantity",
                   }, inplace=True)

df["order_date"] = pd.to_datetime(df["order_date"])

# Group by date — sum all quantities ordered per day
daily_demand = df.groupby(df["order_date"].dt.date)["quantity"].sum().reset_index()
daily_demand.columns = ["date", "quantity"]
daily_demand["date"] = pd.to_datetime(daily_demand["date"])
daily_demand = daily_demand.sort_values("date")

print("Total days:", len(daily_demand))
print("\nSample:\n", daily_demand.head(10))

# Plot the demand over time
plt.figure(figsize=(14, 5))
plt.plot(daily_demand["date"], daily_demand["quantity"], color="steelblue", linewidth=0.8)
plt.title("Daily Demand Over Time (2015–2018)")
plt.xlabel("Date")
plt.ylabel("Total Quantity Ordered")
plt.tight_layout()
plt.savefig("daily_demand.png")
plt.show()

print("\nPlot saved as daily_demand.png")