import findspark
from pyspark.sql import SparkSession


class SparkClass(object):
    """a class to create a Spark environment
    ______
    attributes:
    path_config = the path of the variable spark_home
    ______
    methods:
    spark_start = a method to create a SparkSession

    """

    def __init__(self, path_config):
        self.config = path_config

    def spark_start(self, app_name):
        findspark.init(self.config)
        return SparkSession.builder.appName(app_name).getOrCreate()




