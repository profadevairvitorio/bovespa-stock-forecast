import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("Bovespa - PETR3-LIMPA.csv")

cols_to_convert = ["Open", "High", "Low", "Close"]
df[cols_to_convert] = df[cols_to_convert].replace(",", ".", regex=True).astype(float)
df["Volume"] = df["Volume"].astype(int)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

df["MA_5"] = df["Close"].rolling(window=5).mean()
df["MA_20"] = df["Close"].rolling(window=20).mean()
df["Volatility_5"] = df["Close"].rolling(window=5).std()
df["Daily_Return"] = df["Close"].pct_change()

df = df.dropna().reset_index(drop=True)

features = ["Open", "High", "Low", "Volume", "MA_5", "MA_20", "Volatility_5", "Daily_Return"]
target = "Close"
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE da Regressão Linear para previsão do Close: {rmse:.4f}")

plt.figure(figsize=(14, 6))
plt.plot(df["Date"].iloc[y_test.index], y_test, label="Valor Real (Close)", color="blue")
plt.plot(df["Date"].iloc[y_test.index], y_pred, label="Previsão (Regressão Linear)", color="orange")
plt.title("Previsão do Preço de Fechamento (Close) - PETR3")
plt.xlabel("Data")
plt.ylabel("Preço (R$)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
