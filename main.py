import os

import class_spark

'''Initializing spark session and importing file'''
path = ''
filename = 'white-wine-price-rating.csv'

spark_home = '/usr/local/Cellar/apache-spark/3.2.1/libexec'
#df = class_spark.SparkClass(spark_home, filename).open_csv()
#df.show(5)

'''Data cleaning and preprocessing'''
useless_feat = ('FullName, VintageRating, VintageRatingCount, '
                                            'VintagePrice, VintageRatingPriceRatio, WineRatingPriceRatio')
df_clean = class_spark.SparkClass(spark_home, filename).open_csv()
df_clean.dropna().drop(useless_feat)
df_clean.show()
