from load_preprocessing import PipelineCreator, Loader, Plotter, Scaler
from pyspark.sql.functions import round
from ml_activities import linear_regressor, random_forest_regressor, subset_selection_rf
import matplotlib.pyplot as plt

'''Initializing spark session and importing file (included data cleaning)'''

filename = 'white-wine-price-rating.csv'
useless_feat = ['FullName', 'VintageRating', 'VintageRatingCount', 'VintagePrice', 'VintageRatingPriceRatio',
                'WineRatingPriceRatio']
loader = Loader(filename)
loader.load_csv().drop(useless_feat).show()

"""Plotting the dataframe for data cleaning"""

# df_plot = Plotter(loader)
# print(f'Dataframe Pandas converted:\n{df_plot.dataframe}')
# df_plot.displot('WineRating', 'red', 6)
# df_plot.boxplot('WineRatingCount', 'darkred')
# df_plot.boxplot('WinePrice', 'green')
# plt.show()


df_no_outliers = loader.funnel('WineRatingCount', 8000).funnel('WinePrice', 1000).df_cleaned
df_no_outliers.show()
print(f'Observation after the data cleaning:\t  {df_no_outliers.count()}')

'''Data preprocessing '''

pl_df = PipelineCreator(dataframe=df_no_outliers)
df_post_pipeline = pl_df.fit_transform()
df_final = df_post_pipeline.select('features', 'WinePrice')
df_final = df_final.select('features', round(df_final['WinePrice'], 2).alias('WinePrice'))

df_final.show(10)

""" Linear Regression + Random Forest (Not Scaled) """

print('LINEAR REGRESSION (NOT SCALED) \n')
linear_regressor(df_final)
print('RANDOM FOREST (NOT SCALED) \n')
rf_no_scaling = random_forest_regressor(df_final)
print(rf_no_scaling)


""" Linear Regression + Random Forest (Scaled with MinMaxScaler) """

df_minmaxscaled = Scaler(df_final).min_max_scale()
print('LINEAR REGRESSION (MINMAXSCALED SCALED) \n')
linear_regressor(df_minmaxscaled)
print('RANDOM FOREST (MINMAX SCALED) \n')
rf_no_feature_selection = random_forest_regressor(df_minmaxscaled)
print(rf_no_feature_selection)

""" Linear Regression + Random Forest (Scaled with RobustScaler) """

df_robusted = Scaler(df_final).features_to_dense().robust_scale()
print('LINEAR REGRESSION (ROBUST SCALED) \n')
linear_regressor(df_robusted)
print('RANDOM FOREST (ROBUST SCALED) \n')
random_forest_regressor(df_robusted)

""" random forest after susbset selection based on feature importance (no scaled data) """

subset_selection_rf(rf_no_scaling, df_final)
