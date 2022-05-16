from class_spark import SparkEnv
import seaborn as sbs

spark = SparkEnv('final_project', '/usr/local/Cellar/apache-spark/3.2.1/libexec').spark_start()


class Loader(object):
    """a class to create a Spark environment
    ______
    attributes:
    filename = the file's name
    ______
    methods:
    load_csv = a method to load the file on filename's slot

    dropnas = a method to drop none values

    dropper = a method to drop given columns of a dataframe

    """

    def __init__(self, filename):
        self.filename = filename
        self.dataframe = None
        self.df_clean = None

    def load_csv(self):
        """a method that load the csv file"""
        self.dataframe = spark.read.csv(self.filename, header=True, inferSchema=True)
        return self

    def dropnas(self):
        """A method to drop none values"""
        self.dataframe = self.dataframe.dropna()
        return self

    def dropper(self, useless_feat):
        """A method to drop useless features"""
        self.df_clean = self.dataframe.drop(*useless_feat)
        return self


class Plotter(Loader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pandas_dataframe = self.df_clean.toPandas()

    def to_pandas(self):
        pass







if __name__ == '__main__':
    df = Loader('white-wine-price-rating.csv')
    useless_feat = ['FullName', 'VintageRating', 'VintageRatingCount', 'VintagePrice', 'VintageRatingPriceRatio', 'WineRatingPriceRatio']
    df = df.load_csv().dropnas().dropper(useless_feat).df_clean.show()
    df_pandas = Plotter('white-wine-price-rating.csv')
    print(df_pandas.pandas_dataframe)

