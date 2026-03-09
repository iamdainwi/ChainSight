import pandas as pd
import matplotlib.pyplot as plt
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

# Split: 90% train, 10% test
split = int(len(daily_demand) * 0.9)
train = daily_demand["quantity"][:split]
test  = daily_demand["quantity"][split:]

print(f"Training on {len(train)} days, testing on {len(test)} days")

# Train ARIMA model
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Forecast for test period
forecast = model_fit.forecast(steps=len(test))

# Accuracy
mae = mean_absolute_error(test, forecast)
accuracy = 100 - (mae / test.mean() * 100)
print(f"MAE: {mae:.2f}")
print(f"Forecast Accuracy: {accuracy:.2f}%")

# Plot
plt.figure(figsize=(14, 5))
plt.plot(daily_demand["date"][:split], train, label="Training Data", color="steelblue")
plt.plot(daily_demand["date"][split:], test, label="Actual Demand", color="green")
plt.plot(daily_demand["date"][split:], forecast, label="Forecasted Demand", color="red", linestyle="--")
plt.title("ChainSight — Supply Chain Demand Forecast (ARIMA)")
plt.xlabel("Date")
plt.ylabel("Quantity")
plt.legend()
plt.tight_layout()
plt.savefig("forecast_result.png")
plt.show()

print("Plot saved as forecast_result.png")