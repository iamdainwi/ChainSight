import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("datasets/DataCoSupplyChainDataset.csv", encoding="latin-1")
df.rename(columns={"order date (DateOrders)": "order_date", "Order Item Quantity": "quantity"}, inplace=True)
df["order_date"] = pd.to_datetime(df["order_date"])

daily_demand = df.groupby(df["order_date"].dt.date)["quantity"].sum().reset_index()
daily_demand.columns = ["date", "quantity"]
daily_demand["date"] = pd.to_datetime(daily_demand["date"])
daily_demand = daily_demand.sort_values("date").reset_index(drop=True)

# Monthly demand trend
monthly = daily_demand.set_index("date").resample("ME")["quantity"].sum().reset_index()

# ARIMA forecast
split = int(len(daily_demand) * 0.9)
train = daily_demand["quantity"][:split]
test  = daily_demand["quantity"][split:]
model_fit = ARIMA(train, order=(5, 1, 0)).fit()
forecast = model_fit.forecast(steps=len(test))
mae = mean_absolute_error(test, forecast)
accuracy = 100 - (mae / test.mean() * 100)

# Top 5 products by quantity
top_products = (
    df.groupby("Product Name")["quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

# Demand by region
region_demand = df.groupby("Order Region")["quantity"].sum().sort_values(ascending=False).head(6)

# ---- Dashboard Layout ----
fig = plt.figure(figsize=(18, 12))
fig.suptitle("ChainSight — Supply Chain Demand Forecasting Dashboard",
             fontsize=18, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. Full daily demand
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(daily_demand["date"], daily_demand["quantity"], color="steelblue", linewidth=0.7)
ax1.set_title("Daily Demand (2015–2018)")
ax1.set_xlabel("Date")
ax1.set_ylabel("Quantity")

# 2. Forecast vs Actual
ax2 = fig.add_subplot(gs[1, :2])
ax2.plot(daily_demand["date"][split:], test.values, label="Actual", color="green")
ax2.plot(daily_demand["date"][split:], forecast.values, label="Forecast",
         color="red", linestyle="--")
ax2.set_title(f"ARIMA Forecast vs Actual  |  MAE: {mae:.2f}  |  Accuracy: {accuracy:.2f}%")
ax2.set_xlabel("Date")
ax2.set_ylabel("Quantity")
ax2.legend()

# 3. Monthly trend
ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(monthly["date"].dt.strftime("%b %Y"), monthly["quantity"], color="steelblue")
ax3.set_title("Monthly Demand Trend")
ax3.set_xlabel("")
ax3.set_ylabel("Quantity")
ax3.tick_params(axis="x", rotation=90)

# 4. Top 5 products
ax4 = fig.add_subplot(gs[1, 2])
ax4.barh(top_products.index, top_products.values, color="darkorange")
ax4.set_title("Top 5 Products by Demand")
ax4.set_xlabel("Total Quantity")
ax4.invert_yaxis()

plt.savefig("chainsight_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("Dashboard saved as chainsight_dashboard.png")