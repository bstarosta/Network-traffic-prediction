import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor


def process_dataframe(dataframe):
    time_series = dataframe["Time"]
    date_times = [dt.datetime.strptime(date, "%d.%m.%Y %H:%M") for date in time_series]
    weekdays = [date.weekday() for date in date_times]
    months = [date.month for date in date_times]
    hours = [date.timetuple().tm_hour for date in date_times]
    minutes = [date.timetuple().tm_min for date in date_times]
    dataframe["weekday"] = weekdays
    dataframe["months"] = months
    dataframe["hours"] = hours
    dataframe["minutes"] = minutes
    return dataframe.drop("Time", axis="columns")


SERIES_NAME = "pair1"

df = pd.read_csv("Data/" + SERIES_NAME + ".csv")
processed_data = process_dataframe(df)
X = processed_data.drop(SERIES_NAME, axis="columns")
y = processed_data[SERIES_NAME]

ts_cv = TimeSeriesSplit(
    n_splits=5,
    gap=48,
    max_train_size=10000,
    test_size=1000,
)

all_splits = list(ts_cv.split(X, y))
train_0, test_0 = all_splits[0]

X_train = X.iloc[train_0]
y_train = y.iloc[train_0]

X_test = X.iloc[test_0]
y_test = y.iloc[test_0]

x_plot_axis = [dt.datetime.strptime(date, "%d.%m.%Y %H:%M") for date in df.iloc[test_0]["Time"]]
print(len(x_plot_axis))

mlp_regressor = MLPRegressor(random_state=42, max_iter=1500).fit(X_train, y_train)
predictor_y = mlp_regressor.predict(X_test)
print(mlp_regressor.score(X_test, y_test))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.plot(x_plot_axis, y_test, label="Actual traffic")
plt.plot(x_plot_axis, predictor_y, label="MLPPredictor")
plt.gcf().autofmt_xdate()
plt.legend()
plt.show()
