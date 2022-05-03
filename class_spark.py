import findspark
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame


class SparkClass(object):
    """a class to create a Spark environment
    ______
    attributes:
    path_config = the path of the variable spark_home
    ______
    methods:
    spark_start = a method to create a SparkSession

    open_csv = a method that open a csv file as spark dataframe

    """

    def __init__(self, path_config, filename):
        self.config = path_config
        self.filename = filename
        self.infile = self.open_csv()

    def spark_start(self, app_name=None):
        findspark.init(self.config)
        return SparkSession.builder.appName(app_name).getOrCreate()

    def open_csv(self):
        return self.spark_start(app_name='finalproj').read.csv(self.filename, sep=',', inferSchema=True, header=True)


class DatasetCleaner(object):
    """A class to clean a dataset from useless elements"""

    def __init__(self, dataframe:DataFrame):
        self.dataframe = dataframe

    def dropnas(self):
        """A method to drop none values"""
        self.dataframe = self.dataframe.dropna()
        return self

    def dropper(self, useless_features):
        """A method to drop useless features"""
        self.dataframe = self.dataframe.drop(useless_features)
        return self

    def outliers_remove(self, feature, thresold):
        """A method to remove the outliers """
        self.dataframe = self.dataframe.filter(self.dataframe[feature] < thresold)
        return self

    def __call__(self):
        return f'Dataset pulito.\nPrime 10 righe:\n{self.dataframe.show(7)}'

    #todo: cerca di chiudere questa