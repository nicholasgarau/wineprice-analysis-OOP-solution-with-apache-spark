from pyspark.sql import SparkSession
import findspark
findspark.init('/usr/local/Cellar/apache-spark/3.2.1/libexec')
spark = SparkSession.builder.appName('FinalProject').getOrCreate()

df = spark.read.csv('white-wine-price-rating.csv', header=True, inferSchema=True)

df.show(4)