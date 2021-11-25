import datetime as dt

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from tabulate import tabulate

from DataLoader import DataLoader
from ErrorCalculator import ErrorCalculator
from NetworkTrafficDataProcessor import NetworkTrafficDataProcessor

FILE_NAMES = ["pair1.csv", "pair2.csv", "pair3.csv"]
HIDDEN_LAYER_STRUCTURES = [(30,), (50,), (300, 200), (50, 30), (300, 200, 50), (50, 30, 15),
                           (300, 20, 150, 100), (50, 30, 20, 15)]
TEST_SIZE = 5000

results = np.zeros((len(FILE_NAMES), len(HIDDEN_LAYER_STRUCTURES)))

for i in range(len(FILE_NAMES)):

    df = DataLoader.get_standardized_data(FILE_NAMES[i])
    processed_data = NetworkTrafficDataProcessor.add_features(df)

    X = processed_data.drop("Count", axis="columns")
    y = processed_data["Count"]

    X_train = X.iloc[:-TEST_SIZE]
    y_train = y.iloc[:-TEST_SIZE]

    X_test = X.iloc[-TEST_SIZE:]
    y_test = y.iloc[-TEST_SIZE:]

    x_plot_axis = [dt.datetime.strptime(date, "%d.%m.%Y %H:%M") for date in df.iloc[-TEST_SIZE:]["Time"]]

    for j in range(len(HIDDEN_LAYER_STRUCTURES)):

        mlp_regressor = MLPRegressor(random_state=24, hidden_layer_sizes=HIDDEN_LAYER_STRUCTURES[j], max_iter=1500).fit(X_train, y_train)
        predictor_y = mlp_regressor.predict(X_test)

        mape = ErrorCalculator.calculate_mape(y_test, predictor_y)
        rmse = ErrorCalculator.calculate_rmse(y_test, predictor_y)

        results[i][j] = mape

        plt.title("MLP %s " % (HIDDEN_LAYER_STRUCTURES[j],) + FILE_NAMES[i])
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.plot(x_plot_axis, y_test, label="Actual traffic")
        plt.plot(x_plot_axis, predictor_y, label="MLPPredictor")
        plt.gcf().autofmt_xdate()
        plt.legend()
        plt.show()

rows = np.array([FILE_NAMES]).T

for i in range(len(HIDDEN_LAYER_STRUCTURES)):
    tuple_string = [str(value) for value in HIDDEN_LAYER_STRUCTURES[i]]
    HIDDEN_LAYER_STRUCTURES[i] = ",".join(tuple_string)

headers = HIDDEN_LAYER_STRUCTURES
mape_errors = np.concatenate((rows, results), axis=1)
mape_errors = tabulate(mape_errors, headers, floatfmt=".4f")
print(mape_errors)

with open('Results/mape_errors.txt', 'w') as f:
    print('MAPE:\n', mape_errors, file=f)
