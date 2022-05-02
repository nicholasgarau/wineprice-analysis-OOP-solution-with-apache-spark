from pyspark.sql import SparkSession
import findspark
import os
findspark.init('/usr/local/Cellar/apache-spark/3.2.1/libexec')
spark = SparkSession.builder.appName('FinalProject').getOrCreate()


class Loader(object):
    """A class that loading a CSV file"""

    def __init__(self, path, filename):
        self.path = path
        self.filename = filename
        self.fullpath = self.make_fullpath()
        self.infile = None
        self.target = None
        self.features = None

    def make_fullpath(self):
        """Utility that generates the right path to the wanted file"""
        return os.path.join(self.path, self.filename)

    def get(self):
        "A method that print the outputs"
        print(f'Feature names: \n {self.features}')
        print(f'Data frame normalized: \n {self.scale()}')
        print(f'Target variable: \n {self.target}')
        return self

    def scale(self):
        "A method that normalize the values"
        #self.infile.iloc[:, 0:-1] = scaler.fit_transform(self.infile.iloc[:, 0:-1].to_numpy())
        return self.infile









if __name__ == '__main__':
