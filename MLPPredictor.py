import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor

from DataLoader import DataLoader
from ErrorCalculator import ErrorCalculator
from NetworkTrafficDataProcessor import NetworkTrafficDataProcessor


FILE_NAMES = ["pair1.csv", "pair2.csv", "pair3.csv"]
HIDDEN_LAYER_STRUCTURES = [(15, 10, 5), (30, 20, 10), (20, 15, 10, 10), (30, 15, 10, 5)]


for file in FILE_NAMES:

    df = DataLoader.get_standardized_data(file)
    processed_data = NetworkTrafficDataProcessor.add_features(df)

    X = processed_data.drop("Count", axis="columns")
    print(X)
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
    print(len(x_plot_axis))

    for hidden_layer_structure in HIDDEN_LAYER_STRUCTURES:

        mlp_regressor = MLPRegressor(random_state=42, hidden_layer_sizes=hidden_layer_structure, max_iter=1500).fit(X_train, y_train)
        predictor_y = mlp_regressor.predict(X_test)

        mape = ErrorCalculator.calculate_mape(y_test, predictor_y)
        rmse = ErrorCalculator.calculate_rmse(y_test, predictor_y)
        print(mlp_regressor.score(X_test, y_test))
        print('MAPE: %f' % mape)
        print('RMSE: %f' % rmse)

        plt.title("MLP %s " % (hidden_layer_structure,) + file)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.plot(x_plot_axis, y_test, label="Actual traffic")
        plt.plot(x_plot_axis, predictor_y, label="MLPPredictor")
        plt.gcf().autofmt_xdate()
        plt.legend()
        plt.show()
