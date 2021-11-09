import math
import datetime as dt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv("Data/pair1.csv")

y = df.drop("Time", axis="columns")
X = df.drop("pair1", axis="columns")

tss = TimeSeriesSplit(
    n_splits=5,
    gap=48,
    max_train_size=10000,
    test_size=1000,
)

all_splits = list(tss.split(X, y))
train_0, test_0 = all_splits[0]

df_train = df.iloc[train_0]
X_train = X.iloc[train_0]
y_train = y.iloc[train_0]

# arima = auto_arima(y_train.to_numpy(), start_p=1, start_q=1, max_p=3, max_q=3, start_P=0, start_Q=1, max_P=3,
#                    max_Q=3, m=7, stepwise=True, seasonal=True, information_criterion='aic', trace=True, d=1, D=1,
#                    error_action='warn', suppress_warnings=True, random_state=20)
# arima.summary()

model = SARIMAX(df['pair1'], order=(3, 1, 0), seasonal_order=(3, 1, 0, 7))

result = model.fit()
# result.summary()

start = test_0[0]
end = test_0[len(test_0) - 1]

predictions = result.predict(start, end, typ='levels').rename("Predictions")
y_test = y.iloc[test_0]
X_test = X.iloc[test_0]
df_test = df.iloc[test_0]

x_plot_axis = [dt.datetime.strptime(date, "%d.%m.%Y %H:%M") for date in df.iloc[test_0]["Time"]]
print(x_plot_axis)
print(predictions.to_numpy())

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.plot(x_plot_axis, y_test, label="pair1")
plt.plot(x_plot_axis, predictions.to_numpy(), label="predictions")
plt.legend()
plt.show()

# how to measure accuracy???
mse = mean_squared_error(y_test, predictions)
rmse = math.sqrt(mse)
print('RMSE: %f' % rmse)
