from load_preprocessing import PipelineCreator, Loader, Plotter
from pyspark.sql.functions import round

'''Initializing spark session and importing file (included data cleaning)'''

filename = 'white-wine-price-rating.csv'
df = Loader(filename)
useless_feat = ['FullName', 'VintageRating', 'VintageRatingCount', 'VintagePrice', 'VintageRatingPriceRatio',
                'WineRatingPriceRatio']
df.load_csv().dropnas().dropper(useless_feat).df_clean.show()

"""Plotting the dataframe for data cleaning"""

# df_pandas = Plotter(filename).load_csv().dropnas().dropper(useless_feat).to_pandas()
# print(f'Dataframe Pandas converted:\n{df_pandas.pandas_dataframe}')
# dis_plot = df_pandas.distribution_plotter('WineRating', 'red', 6)
# #print(f'distribution plot for WineRating: \n {dis_plot}')
# bp_wineratingcount = df_pandas.box_plotter('WineRatingCount', 'darkred')
# #print(f'boxplot for WineRatingCount: \n {bp_wineratingcount}')
# bp_price = df_pandas.box_plotter('WinePrice', 'green')
# #print(f'boxplot for WinePrice: \n {bp_price}')


df_no_outliers = df.load_csv().dropnas().dropper(useless_feat).funnel('WineRatingCount', 8000).funnel('WinePrice', 1000).df_clean
print(f'Observation after the data cleaning:\t  {df_no_outliers.count()}')

'''Data preprocessing '''

pl_df = PipelineCreator(dataframe=df_no_outliers)
df_post_pipeline = pl_df.pl_fitter()
df_final = df_post_pipeline.select('features', 'WinePrice')
df_final = df_final.select('features', round(df_final['WinePrice'], 2).alias('WinePrice'))

df_final.show(10)

""" Linear Regression + Random Forest """