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
        self.df_clean = self.dropper()

    def spark_start(self, app_name=None):
        findspark.init(self.config)
        return SparkSession.builder.appName(app_name).getOrCreate()

    def open_csv(self):
        return self.spark_start(app_name='finalproj').read.csv(self.filename, sep=',', inferSchema=True, header=True)

    def dropnas(self):
        """A method to drop none values"""
        return self.open_csv().dropna()

    def dropper(self,useless_feat):
        """A method to drop useless features"""
        self.df_clean = self.dropnas().drop(useless_feat)
        return self

    def display(self):
        return {f'Minca dimmi che funzioni!!\n'
                f'{self.df_clean}'}

