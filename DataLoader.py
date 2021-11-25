import pandas as pd


class DataLoader:

    @staticmethod
    def standardize_column_names(dataframe):
        dataframe.rename(columns={dataframe.columns[1]: "Traffic"}, inplace=True)

    @staticmethod
    def load_data(filename):
        return pd.read_csv("Data/" + filename)

    @staticmethod
    def get_standardized_data(filename):
        dataframe = DataLoader.load_data(filename)
        DataLoader.standardize_column_names(dataframe)
        return dataframe
