from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor as RF
from pyspark.sql.functions import round
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt


def linear_regressor(dataframe):
    training_set, test_set = dataframe.randomSplit([0.8, 0.2])
    regressor = LinearRegression(featuresCol='features', labelCol='WinePrice', predictionCol='prediction')
    model_lm = regressor.fit(training_set)
    performance_lm = model_lm.evaluate(test_set)
    print(f"R2 measure: \t {performance_lm.r2}")
    print(f"Mean Absolute Error: \t {performance_lm.meanAbsoluteError}")
    print(f"Root Mean Squared Error: \t {performance_lm.rootMeanSquaredError}")
    predictions_lm = model_lm.transform(test_set)
    predictions_lm.select('WinePrice', round('prediction', 2).alias('prediction')).show(20)


def random_forest_regressor(dataframe):
    training_set, test_set = dataframe.randomSplit([0.8, 0.2])
    regressor_rf = RF(featuresCol='features', labelCol='WinePrice', numTrees=10, maxDepth=5)
    model_rf = regressor_rf.fit(training_set)
    predictions_rf = model_rf.transform(test_set)
    evaluator = RegressionEvaluator(labelCol="WinePrice", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions_rf)
    rf_pred = model_rf.transform(dataframe)
    rf_result = rf_pred.toPandas()
    plt.plot(rf_result.WinePrice, rf_result.prediction, 'bo')
    plt.xlabel('Price')
    plt.ylabel('Prediction')
    plt.suptitle("Model Performance RMSE: %f" % rmse)
    #plt.show()
    print(f'Feature importance: \t {model_rf.featureImportances}')

    #todo: sei arrivato alla feature importance