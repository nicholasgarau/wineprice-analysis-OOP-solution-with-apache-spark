from class_spark import SparkEnv
import seaborn as sns

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



# if __name__ == '__main__':
#     df = Loader('white-wine-price-rating.csv')
#     useless_feat = ['FullName', 'VintageRating', 'VintageRatingCount', 'VintagePrice', 'VintageRatingPriceRatio',
#                     'WineRatingPriceRatio']
#     df = df.load_csv().dropnas().dropper(useless_feat).df_clean.show()
#     df_pandas = Plotter('white-wine-price-rating.csv')
#     print(df_pandas.pandas_dataframe)

