import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error


class ErrorCalculator:

    @staticmethod
    def calculate_mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return mean_absolute_percentage_error(y_true, y_pred)

    @staticmethod
    def calculate_rmse(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return mean_squared_error(y_true, y_pred, squared=False)
