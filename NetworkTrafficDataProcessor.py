import datetime as dt
import pandas as pd


class NetworkTrafficDataProcessor:

    @staticmethod
    def add_features(dataframe):
        time_series = dataframe["Time"]
        date_times = [dt.datetime.strptime(date, "%d.%m.%Y %H:%M") for date in time_series]
        weekdays = [date.weekday() for date in date_times]
        hours = [date.timetuple().tm_hour for date in date_times]
        minutes = [date.timetuple().tm_min for date in date_times]
        previous = dataframe["Traffic"].shift(1)
        one_hot_weekdays = pd.get_dummies(weekdays, prefix="Weekday")
        dataframe["Hours"] = hours
        dataframe["Minutes"] = minutes
        dataframe["Previous"] = previous
        dataframe = dataframe.join(one_hot_weekdays)
        dataframe = dataframe.iloc[1:, :]
        return dataframe.drop("Time", axis="columns")

