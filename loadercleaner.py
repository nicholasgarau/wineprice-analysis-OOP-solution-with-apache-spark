import os
import class_spark


# class Loader(object):
#     """A class that loading a CSV file"""
#
#     def __init__(self, path, filename):
#         self.path = path
#         self.filename = filename
#         self.fullpath = self.make_fullpath()
#         self.infile = None
#         self.target = None
#         self.features = None
#         self.scaled_df = None
#
#     def make_fullpath(self):
#         """Utility that generates the right path to the wanted file"""
#         return os.path.join(self.path, self.filename)
#
#     def load(self):
#         """A method that load the csv"""
#         self.infile = class_spark.SparkClass.open_csv(class_spark.selfSparkClass, self.filename)
#         self.target = self.infile.select('WinePrice')
#         self.features = [item for item in self.infile.columns if item != 'WinePrice']
#         return self.infile


class DatasetCleaner(object):
    """A class to clean a dataset from useless elements"""

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def dropnas(self):
        """A method to drop none values"""
        self.dataframe = self.dataframe.dropna()
        return self

    def dropper(self, useless_features=str):
        """A method to drop useless features"""
        self.dataframe = self.dataframe.drop(useless_features)
        return self

    def outliers_remove(self, feature, thresold):
        """A method to remove the outliers """
        self.dataframe = self.dataframe.filter(self.dataframe[feature] < thresold)
        return self

    def __str__(self):
        return f'Dataset pulito.\nPrime 10 righe:\n{self.dataframe.show(5)}'

