import datetime as dt


class NetworkTrafficDataProcessor:

    @staticmethod
    def add_features(dataframe):
        time_series = dataframe["Time"]
        date_times = [dt.datetime.strptime(date, "%d.%m.%Y %H:%M") for date in time_series]
        weekdays = [date.weekday() for date in date_times]
        months = [date.month for date in date_times]
        day = [date.day for date in date_times]
        hours = [date.timetuple().tm_hour for date in date_times]
        minutes = [date.timetuple().tm_min for date in date_times]
        previous = dataframe["Count"].shift(1)
        before_previous = dataframe["Count"].shift(2)
        #dataframe["day"] = day
        dataframe["Weekday"] = weekdays
        #dataframe["months"] = months
        dataframe["Hours"] = hours
        dataframe["Minutes"] = minutes
        dataframe["Previous"] = previous
        #dataframe["Before previous"] = before_previous
        dataframe = dataframe.iloc[1:, :]
        return dataframe.drop("Time", axis="columns")

