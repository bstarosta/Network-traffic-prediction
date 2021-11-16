import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from tabulate import tabulate

from DataLoader import DataLoader
from ErrorCalculator import ErrorCalculator
from NetworkTrafficDataProcessor import NetworkTrafficDataProcessor


FILE_NAMES = ["pair1.csv", "pair2.csv", "pair3.csv"]
HIDDEN_LAYER_STRUCTURES = [(300,), (50,), (300, 200), (50, 30), (300, 200, 4), (50, 30, 15),
                           (300, 20, 150, 100), (50, 30, 3, 2)]

results = np.zeros((len(FILE_NAMES), len(HIDDEN_LAYER_STRUCTURES)))

for i in range(len(FILE_NAMES)):

    df = DataLoader.get_standardized_data(FILE_NAMES[i])
    processed_data = NetworkTrafficDataProcessor.add_features(df)

    X = processed_data.drop("Count", axis="columns")
    y = processed_data["Count"]

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

    for j in range(len(HIDDEN_LAYER_STRUCTURES)):

        mlp_regressor = MLPRegressor(random_state=42, hidden_layer_sizes=HIDDEN_LAYER_STRUCTURES[j], max_iter=1500).fit(X_train, y_train)
        predictor_y = mlp_regressor.predict(X_test)

        mape = ErrorCalculator.calculate_mape(y_test, predictor_y)
        rmse = ErrorCalculator.calculate_rmse(y_test, predictor_y)
        # print(mlp_regressor.score(X_test, y_test))
        #         # print('MAPE: %f' % mape)
        #         # print('RMSE: %f' % rmse)

        results[i][j] = mape

        plt.title("MLP %s " % (HIDDEN_LAYER_STRUCTURES[j],) + FILE_NAMES[i])
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.plot(x_plot_axis, y_test, label="Actual traffic")
        plt.plot(x_plot_axis, predictor_y, label="MLPPredictor")
        plt.gcf().autofmt_xdate()
        plt.legend()
        plt.show()


print(results)

# headers = list(clfs.keys())
# names_column = np.array(headers)[np.newaxis].T
# advantage_table = tabulate(np.concatenate((names_column, advantage), axis=1), headers)

rows = np.array([FILE_NAMES]).T
# headers = list(map(''.join, HIDDEN_LAYER_STRUCTURES))
# headers = np.zeros(len(HIDDEN_LAYER_STRUCTURES))

for i in range(len(HIDDEN_LAYER_STRUCTURES)):
    tuple_string = [str(value) for value in HIDDEN_LAYER_STRUCTURES[i]]
    HIDDEN_LAYER_STRUCTURES[i] = ",".join(tuple_string)

headers = HIDDEN_LAYER_STRUCTURES
mape_errors = np.concatenate((rows, results), axis=1)
mape_errors = tabulate(mape_errors, headers, floatfmt=".4f")
print(mape_errors)

with open('Results/mape_errors.txt', 'w') as f:
    print('MAPE:\n', mape_errors, file=f)
