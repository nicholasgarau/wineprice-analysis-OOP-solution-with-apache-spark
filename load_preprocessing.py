from class_spark import SparkEnv
import seaborn as sns
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

spark = SparkEnv('final_project', '/usr/local/Cellar/apache-spark/3.2.1/libexec').spark_start()


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
        self.df_clean = None

    def load_csv(self):
        self.dataframe = spark.read.csv(self.filename, header=True, inferSchema=True)
        return self

    def dropnas(self):
        self.dataframe = self.dataframe.dropna()
        return self

    def dropper(self, useless_feat):
        self.df_clean = self.dataframe.drop(*useless_feat)
        return self

    def funnel(self, column=str, max_threshold=int):
        self.df_clean = self.df_clean.filter(self.df_clean[column] < max_threshold)
        return self

    def reverse_funnel(self, column=str, min_threshold=int):
        self.df_clean = self.df_clean.filter(self.df_clean[column] > min_threshold)
        return self


class Plotter(Loader):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pandas_dataframe = None

    def to_pandas(self):
        self.pandas_dataframe = self.df_clean.toPandas()
        return self

    def distribution_plotter(self, column=str, color=str, height=int):
        return sns.displot(self.pandas_dataframe[column], aspect=2, color=color, height=height)

    def box_plotter(self, column=str, color=str):
        return sns.boxplot(data=self.pandas_dataframe[column], color=color, orient='h')

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
        self.pipeline = self.pl_generator()

    def pl_generator(self):
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

    def pl_fitter(self):
        df_assembled = self.pipeline.fit(self.df_raw).transform(self.df_raw)
        return df_assembled



