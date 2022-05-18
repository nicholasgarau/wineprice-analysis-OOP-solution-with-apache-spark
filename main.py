import loader_plotter

'''Initializing spark session and importing file (included data cleaning)'''

filename = 'white-wine-price-rating.csv'
df = loader_plotter.Loader(filename)
useless_feat = ['FullName', 'VintageRating', 'VintageRatingCount', 'VintagePrice', 'VintageRatingPriceRatio',
                'WineRatingPriceRatio']
df.load_csv().dropnas().dropper(useless_feat).df_clean.show()

"""Plotting the dataframe for data cleaning"""

df_pandas = loader_plotter.Plotter(filename).load_csv().dropnas().dropper(useless_feat).to_pandas()
print(f'Dataframe Pandas converted:\n{df_pandas.pandas_dataframe}')
dis_plot = df_pandas.distribution_plotter('WineRating', 'red', 6)
#print(f'distribution plot for WineRating: \n {dis_plot}')
bp_wineratingcount = df_pandas.box_plotter('WineRatingCount', 'darkred')
#print(f'boxplot for WineRatingCount: \n {bp_wineratingcount}')
bp_price = df_pandas.box_plotter('WinePrice', 'green')
#print(f'boxplot for WinePrice: \n {bp_price}')


df_no_outliers = df.load_csv().dropnas().dropper(useless_feat).funnel('WineRatingCount', 8000).funnel('WinePrice', 1000)
df_no_outliers.df_clean.show()
print(f'Observation after the data cleaning:\t  {df_no_outliers.df_clean.count()}')

'''Data preprocessing '''
