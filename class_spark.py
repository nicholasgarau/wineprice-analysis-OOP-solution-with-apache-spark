import findspark
from pyspark.sql import SparkSession

class SparkEnv(object):
    """
    A class that create a Spark environment
    ______
    attributes:
    app_name = the name of the session that you create
    path_config = the path of the SPARK_HOME
    ______
    methods:
    spark_start() = a method to start a Spark session

    """
    def __init__(self, app_name, path_config):
        self.app_name = app_name
        self.config = path_config


    def spark_start(self):
        findspark.init(self.config)
        spark_session = SparkSession.builder.appName(self.app_name).getOrCreate()
        return spark_session


