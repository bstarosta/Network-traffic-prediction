import datetime as dt

import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

from DataLoader import DataLoader
from ErrorCalculator import ErrorCalculator

FILE_NAMES = ["pair1.csv", "pair2.csv", "pair3.csv"]
MODEL_ORDERS = [(3, 1, 0), (2, 1, 1), (3, 1, 0)]
MODEL_SEASONAL_ORDERS = [(3, 1, 0, 7), (3, 1, 0, 7), (3, 1, 0, 7)]
TEST_SIZE = 5000

results = [None] * len(FILE_NAMES)

for i in range(len(FILE_NAMES)):
    df = DataLoader.get_standardized_data(FILE_NAMES[i])
    y = df.drop("Time", axis="columns")
    X = df.drop("Count", axis="columns")
    y_test = y.iloc[-TEST_SIZE:]

    # arima = auto_arima(y_train.to_numpy(), start_p=1, start_q=1, max_p=3, max_q=3, start_P=0, start_Q=1, max_P=3,
    #                    max_Q=3, m=7, stepwise=True, seasonal=True, information_criterion='aic', trace=True, d=1, D=1,
    #                    error_action='warn', suppress_warnings=True, random_state=20)
    # arima.summary()

    model = SARIMAX(y, order=MODEL_ORDERS[i], seasonal_order=MODEL_SEASONAL_ORDERS[i])

    result = model.fit()

    start = y_test.index[0]
    end = y_test.index[len(y_test) - 1]

    predictions = result.predict(start, end, typ='levels').rename("Predictions")

    x_plot_axis = [dt.datetime.strptime(date, "%d.%m.%Y %H:%M") for date in df.iloc[-TEST_SIZE:]["Time"]]

    plt.title("ARIMA " + FILE_NAMES[i])
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.plot(x_plot_axis, y_test, label="Actual traffic")
    plt.plot(x_plot_axis, predictions.to_numpy(), label="ARIMA")
    plt.legend()
    plt.show()

    mape = ErrorCalculator.calculate_mape(y_test, predictions)
    rmse = ErrorCalculator.calculate_rmse(y_test, predictions)
    results[i] = mape

with open('Results/arima_results.txt', 'w') as f:
    print('MAPE:', results, file=f)
