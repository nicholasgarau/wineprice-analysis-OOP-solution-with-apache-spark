from class_spark import SparkEnv
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, RobustScaler
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.sql.functions import udf

spark = SparkEnv('final_project', '/usr/local/Cellar/apache-spark/3.2.1/libexec').spark_start()
as_dense = udf(
    lambda v: DenseVector(v.toArray()) if v is not None else None,
    VectorUDT()
)


class Loader(object):
    """a class to create a Spark environment
    ______
    attributes:
    filename = the file's name
    ______
    methods:
    load_csv() = a method to load the file on filename's slot

    dropnas() = a method to drop none values

    dropper() = a method to drop given columns of a dataframe
        params:
            • useless_feat(list) = a list of strings or a string with the features to drop

    funnel() = a method to filter the values of a column and drop the values over the threshold
        params:
            • column(str) = the name of the column
            • max_threshold(int) = the maximum threshold of the values to keep

    reverse_funnel() = a method to filter the values of a column and drop the values under the threshold
        params:
            • column(str) = the name of the column
            • min_threshold(int) = the minimum threshold of the values to keep

    """

    def __init__(self, filename):
        self.filename = filename
        self.dataframe = None
        self.df_cleaned = None
        self.pandas_dataframe = None

    def load_csv(self):
        self.dataframe = spark.read.csv(self.filename, header=True, inferSchema=True)
        self.dataframe = self.dataframe.dropna()
        return self

    def dropnas(self):
        self.dataframe = self.dataframe.dropna()
        return self

    def drop(self, useless_feat):
        self.df_cleaned = self.dataframe.drop(*useless_feat)
        return self

    def funnel(self, column=str, max_threshold=int):
        self.df_cleaned = self.df_cleaned.filter(self.df_cleaned[column] < max_threshold)
        return self

    def reverse_funnel(self, column=str, min_threshold=int):
        self.df_cleaned = self.df_cleaned.filter(self.df_cleaned[column] > min_threshold)
        return self

    def to_pandas(self):
        self.pandas_dataframe = self.df_cleaned.toPandas()
        return self

    def show(self):
        return self.df_cleaned.show()


class Plotter(object):
    """
    A Loader sub-class implemented for data visualization:
    _____
    methods:
    to_pandas() = a method to convert a Spark dataframe to Pandas dataframe

    distribution_plotter() = a method to create a distribution plot
        params:
         • column(str): the name of the column
         • color(str): the bars' color of the plot
         • height(int): the height of the plot

    boxplotter() = a method to create a boxplot
        params:
         • column(str): the name of the column
         • color(str): the bars' color of the plot

    """

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def displot(self, column=str, color=str, height=int):
        return sns.displot(self.dataframe[column], aspect=2, color=color, height=height)

    def boxplot(self, column=str, color=str):
        return sns.boxplot(data=self.dataframe[column], color=color, orient='h')


class PipelineCreator(object):
    """
    A class that creates a Pipeline and returns a spark dataframe ready to machine learning alghoritms
    ________
    attributes:
    dataframe = a dataframe ready for the preprocessing

    ----
    methods:
    pl_generator() = a method that return a pipeline ready to fit
    pl_fitter() = a method to fit the pipeline created on pl_generator

    """

    def __init__(self, dataframe):
        self.df_raw = dataframe
        self.pipeline = self.make_pipeline()

    def make_pipeline(self):
        indexer = StringIndexer(inputCols=['Winery', 'Region', 'RegionalVariety', 'Year'],
                                outputCols=['WineryNDX', 'RegionNDX', 'RegionalVarietyNDX', 'YearNDX'],
                                handleInvalid='skip')
        encoder = OneHotEncoder(inputCols=['WineryNDX', 'YearNDX', 'RegionNDX', 'RegionalVarietyNDX'],
                                outputCols=(['WineryENC', 'YearENC', 'RegionENC', 'RegionalVarietyENC'])
                                )
        assembler = VectorAssembler(inputCols=['WineryENC', 'RegionENC', 'RegionalVarietyENC', 'YearENC',
                                               'WineRating', 'WineRatingCount'], outputCol='features',
                                    handleInvalid='skip')

        pipeline = Pipeline(stages=[indexer, encoder, assembler])
        return pipeline

    def fit_transform(self):
        return self.pipeline.fit(self.df_raw).transform(self.df_raw)


class Scaler(object):
    """
    A class to scale a Spark dataframe:
    ______
    attributes:
    dataframe = the spark dataframe that going to be scaled
    ______
    methods:
    min_max_scaler() = a method that return a scaled dataframe using MinMaxScaler from pyspark
    robust_scaler() = a method that return a scaled dataframe using RobustScaler from pyspark

    """

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def min_max_scaler(self):
        scaler = MinMaxScaler(inputCol='features', outputCol='scaledFeatures')
        scaled_df = scaler.fit(self.dataframe).transform(self.dataframe)
        return scaled_df

    def robust_scaler(self):
        scaler = RobustScaler(inputCol='features', outputCol='scaledFeatures', withCentering=False)
        scaled_df = scaler.fit(self.dataframe).transform(self.dataframe)
        return scaled_df

    def features_to_dense(self):
        self.dataframe = self.dataframe.withColumn("features", as_dense("features"))
        return self
