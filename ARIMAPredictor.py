import datetime as dt

import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX

from DataLoader import DataLoader
from ErrorCalculator import ErrorCalculator

FILE_NAMES = ["pair1.csv", "pair2.csv", "pair3.csv"]
MODEL_ORDERS = [(3, 1, 0), (2, 1, 1), (3, 1, 0)]
MODEL_SEASONAL_ORDERS = [(3, 1, 0, 7), (3, 1, 0, 7), (3, 1, 0, 7)]

results = [None] * len(FILE_NAMES)

for i in range(len(FILE_NAMES)):
    df = DataLoader.get_standardized_data(FILE_NAMES[i])
    y = df.drop("Time", axis="columns")
    X = df.drop("Count", axis="columns")

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

    model = SARIMAX(df['Count'], order=MODEL_ORDERS[i], seasonal_order=MODEL_SEASONAL_ORDERS[i])

    result = model.fit()
    # result.summary()

    start = test_0[0]
    end = test_0[len(test_0) - 1]

    predictions = result.predict(start, end, typ='levels').rename("Predictions")
    y_test = y.iloc[test_0]
    X_test = X.iloc[test_0]
    df_test = df.iloc[test_0]

    x_plot_axis = [dt.datetime.strptime(date, "%d.%m.%Y %H:%M") for date in df.iloc[test_0]["Time"]]

    plt.title("ARIMA " + FILE_NAMES[i])
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.plot(x_plot_axis, y_test, label="Actual traffic")
    plt.plot(x_plot_axis, predictions.to_numpy(), label="ARIMA")
    plt.legend()
    plt.show()

    print(y_test)
    print(predictions)

    # how to measure accuracy???
    mape = ErrorCalculator.calculate_mape(y_test, predictions)
    rmse = ErrorCalculator.calculate_rmse(y_test, predictions)
    # print('MAPE: %f' % mape)
    # print('RMSE: %f' % rmse)
    results[i] = mape

print(results)
with open('Results/arima_results.txt', 'w') as f:
    print('MAPE:', results, file=f)
